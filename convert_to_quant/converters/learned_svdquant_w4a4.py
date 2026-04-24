# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Learned Rounding SVDQuant W4A4 Converter.

Inherits optimizer infrastructure from BaseLearnedConverter and applies
learned rounding to the int4 residual weight after the SVD low-rank
correction has been subtracted.

Optimization variable
---------------------
All four optimizers work in quantized int4 units (~[-7, 7]):

    W_q      = round(W_residual / wscales).clamp(-7, 7)   initial grid snap
    W_dq     = W_q * wscales                              dequant (per group)
    loss     = ||U_k.T @ (W_dq - W_residual) @ Vh_k.T||_F

Gradient (original optimizer, no autograd):
    By chain rule: dL/dW_q = dL/dW_dq * dW_dq/dW_q = dL/dW_dq * wscales
    So the physical-space gradient is multiplied by wscales to move into
    W_q space before the gradient step. Matches the INT8 blockwise pattern
    in learned_rounding.py::_optimize_int8_original.

Autograd optimizers (AdamW / RAdam / Prodigy):
    The delta parameter is added to the snapped W_q_init. Gradients flow
    through the differentiable dequant (_dequantize_blockwise) which
    multiplies by wscales, so the chain rule is handled automatically.

Schedule / early stopping conventions
--------------------------------------
original optimizer:
    - Shape-aware plateau parameters (matching FP8 / INT8 original)
    - Full debug plateau logging
    - Compound early-stop with categorised messages (matching FP8 / INT8
      original in learned_rounding.py)
    - Pbar always shows worse_count (matching _optimize_int8_original)

autograd optimizers (AdamW / RAdam / Prodigy):
    - Bare lr_patience / lr_factor for plateau (no shape-aware scaling,
      matching _optimize_int8_adamw/radam/prodigy)
    - Full debug plateau logging matching learned_rounding.py
    - Simple early-stop inline checking best_loss (matching INT8 autograd)
    - Pbar shows plateau/patience in plateau mode, worse_count otherwise
      (matching FP8/MXFP8/NVFP4 autograd)
"""
from __future__ import annotations

import gc
import math
from typing import Tuple, Optional, Dict

import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import COMPUTE_DTYPE
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import info, verbose, debug
from .base_converter import BaseLearnedConverter
from .svdquant_w4a4_converter import (
    _pack_int4_weight,
    _INT4_GROUP_SIZE,
    _INT4_QMAX,
    SVDQuantW4A4Converter,
)


class LearnedSVDQuantW4A4Converter(BaseLearnedConverter):
    """SVDQuant W4A4 converter with learned rounding optimization.

    Inherits all optimizer infrastructure from BaseLearnedConverter.

    Args:
        rank: SVD rank for the low-rank LoRA correction (default 32).
        smooth_mode: "weight_only", "ones", or "external".
        smooth_alpha: Power for "weight_only" heuristic (default 0.5).
        **kwargs: All BaseLearnedConverter args (optimizer, num_iter, lr,
            lr_schedule, top_p, min_k, max_k, early_stop_*, etc.).
    """

    def __init__(
        self,
        rank: int = 32,
        smooth_mode: str = "weight_only",
        smooth_alpha: float = 0.5,
        **kwargs,
    ):
        valid_smooth = ("weight_only", "ones", "external")
        if smooth_mode not in valid_smooth:
            raise ValueError(
                f"smooth_mode must be one of {valid_smooth}. Got '{smooth_mode}'."
            )

        super().__init__(**kwargs)

        self.rank = rank
        self.smooth_mode = smooth_mode
        self.smooth_alpha = smooth_alpha
        self.group_size = _INT4_GROUP_SIZE

        # Reuse SVDQuantW4A4Converter helpers for smooth / SVD / scale setup.
        self._simple = SVDQuantW4A4Converter(
            rank=rank,
            smooth_mode=smooth_mode,
            smooth_alpha=smooth_alpha,
            device=self.device,
        )

        verbose(f"LearnedSVDQuantW4A4Converter initialized on device: {self.device}")
        verbose(f"  - SVD rank  : {self.rank}")
        verbose(
            f"  - Smooth    : {self.smooth_mode}"
            + (f" (alpha={self.smooth_alpha})" if smooth_mode == "weight_only" else "")
        )
        verbose(
            f"  - Optimizer : {self.optimizer_choice}"
            + (" (simple — no learned rounding)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            verbose(f"  - LR schedule: {self.lr_schedule}")

    # ------------------------------------------------------------------
    # Public entry point  (matches SVDQuantW4A4Converter.convert signature)
    # ------------------------------------------------------------------

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        smooth: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,   # qweight        (N, K//2) int8
        torch.Tensor,   # wscales        (K//64, N) bf16  transposed
        torch.Tensor,   # proj_down      (K, R) bf16
        torch.Tensor,   # proj_up        (N, R) bf16
        torch.Tensor,   # smooth_factor  (K,) bf16
        torch.Tensor,   # dequantized    (N, K) bf16  for bias correction
    ]:
        N, K = W_orig.shape
        label = key or f"({N}, {K})"

        if K % self.group_size != 0:
            raise ValueError(
                f"Layer {label}: K={K} not divisible by group_size={self.group_size}."
            )

        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            verbose(f"  [{label}] All-zeros tensor, skipping optimization.")
            return self._simple.convert(W_orig, key=key, smooth=smooth)

        # ---- Smooth factor -------------------------------------------------
        smooth_factor = self._simple._compute_smooth(W_float32, smooth, label)
        W_smoothed = W_float32 * smooth_factor[None, :]

        # ---- SVD low-rank correction ----------------------------------------
        effective_rank = min(self.rank, min(N, K))
        if effective_rank < self.rank:
            import logging
            logging.getLogger(__name__).warning(
                "Layer %s: rank=%d clamped to %d", label, self.rank, effective_rank
            )

        if effective_rank > 0:
            # SVD on W (original unsmoothed weight): LoRA runs on raw x at inference,
            # not x/smooth. See svdquant_w4a4_converter.py for full explanation.
            proj_up, proj_down = self._simple._compute_svd_correction(
                W_float32, effective_rank, label
            )
            W_residual = W_smoothed - (proj_up.float() @ proj_down.float().T)
        else:
            proj_up   = torch.zeros(N, 0, dtype=torch.float32, device=self.device)
            proj_down = torch.zeros(K, 0, dtype=torch.float32, device=self.device)
            W_residual = W_smoothed

        # ---- Initial per-group scales (fixed, never updated) ----------------
        G = self.group_size
        W_groups = W_residual.view(N, K // G, G)
        absmax   = W_groups.abs().amax(dim=-1)                      # (N, K/G)
        wscales  = (absmax / _INT4_QMAX).clamp(min=1e-8)            # (N, K/G) float32

        # ---- Quantize -------------------------------------------------------
        if self.no_learned_rounding:
            q_vals = self._simple_quantize(W_residual, wscales)
        else:
            q_vals = self._optimize(W_residual, wscales, label)

        # ---- Pack, dequant, output ------------------------------------------
        qweight  = _pack_int4_weight(q_vals)                         # (N, K//2) int8

        q_dq = (
            q_vals.float()
            * wscales.view(N, K // G, 1).expand_as(W_groups).reshape(N, K)
        )
        if proj_up.shape[1] > 0:
            W_dequant = q_dq + (proj_up.float() @ proj_down.float().T)
        else:
            W_dequant = q_dq

        wscales_T = wscales.T.contiguous()                           # (K/G, N)

        compute_dtype = torch.bfloat16
        result = (
            qweight.cpu(),
            wscales_T.to(compute_dtype).cpu(),
            proj_down.to(compute_dtype).cpu(),
            proj_up.to(compute_dtype).cpu(),
            smooth_factor.to(compute_dtype).cpu(),
            W_dequant.to(compute_dtype).cpu(),
        )

        del W_float32, W_smoothed, W_residual, W_groups, q_dq
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Quantize helpers
    # ------------------------------------------------------------------

    def _simple_quantize(
        self, W_residual: torch.Tensor, wscales: torch.Tensor
    ) -> torch.Tensor:
        """Round-to-nearest without optimization."""
        N, K = W_residual.shape
        G = self.group_size
        W_groups = W_residual.view(N, K // G, G)
        q_vals = (
            (W_groups / wscales.unsqueeze(-1))
            .clamp(-_INT4_QMAX, _INT4_QMAX)
            .round()
            .to(torch.int8)
        )
        return q_vals.view(N, K)

    def _dequantize_blockwise(
        self,
        W_q: torch.Tensor,
        wscales: torch.Tensor,
        N: int,
        K: int,
    ) -> torch.Tensor:
        """Differentiable dequant for the optimization loop.

        W_dq = W_q * wscales  (per group along K).
        No discretize flag needed — the optimization variable is always
        treated as continuous float32; the int4 grid snap happens only
        at init and at the final round() call.
        """
        G = self.group_size
        return (
            W_q.reshape(N, K // G, G)
            * wscales.unsqueeze(-1)
        ).view(N, K)

    # ------------------------------------------------------------------
    # Optimizer dispatcher
    # ------------------------------------------------------------------

    def _optimize(
        self,
        W_residual: torch.Tensor,
        wscales: torch.Tensor,
        label: str,
    ) -> torch.Tensor:
        """Run learned rounding; return (N, K) int8 q_vals."""
        U_k, Vh_k, _ = self._compute_svd_components(W_residual)

        if self.optimizer_choice == "original":
            q_vals = self._optimize_original(W_residual, wscales, U_k, Vh_k)
        elif self.optimizer_choice == "adamw":
            q_vals = self._optimize_adamw(W_residual, wscales, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            q_vals = self._optimize_radam(W_residual, wscales, U_k, Vh_k)
        elif self.optimizer_choice == "prodigy":
            q_vals = self._optimize_prodigy(W_residual, wscales, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        self._cleanup_tensors(U_k, Vh_k)
        return q_vals

    # ------------------------------------------------------------------
    # Original (manual gradient descent, no autograd)
    # ------------------------------------------------------------------

    def _optimize_original(
        self,
        W_residual: torch.Tensor,
        wscales: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        N, K = W_residual.shape
        G = self.group_size

        # Snap to int4 grid for a non-zero starting loss.
        W_groups = W_residual.view(N, K // G, G)
        W_q = (
            (W_groups / wscales.unsqueeze(-1))
            .clamp(-_INT4_QMAX, _INT4_QMAX)
            .round()
            .float()
            .view(N, K)
            .clone()
        )

        best_loss    = float("inf")
        best_W_q     = None
        worse_loss_counter = 0
        plateau_counter    = 0
        cooldown_counter   = 0
        curr_lr      = self.lr
        schedule     = self.lr_schedule
        M            = N   # alias for shape-aware helpers that expect (M, N)

        # Shape-aware plateau parameters (matching FP8 / INT8 original).
        aspect_ratio = max(N, K) / min(N, K)
        if schedule == "plateau" and self.lr_shape_influence > 0:
            ar_factor          = math.sqrt(aspect_ratio)
            blend              = self.lr_shape_influence
            effective_patience = self.lr_patience
            raw_factor         = self.lr_factor
            aggressive_factor  = raw_factor ** ar_factor
            effective_factor   = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor   = self.lr_factor
            effective_cooldown = self.lr_cooldown

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing SVDQuant W4A4 (Original-{schedule})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            with torch.no_grad():
                current_dq      = self._dequantize_blockwise(W_q, wscales, N, K)
                error           = current_dq - W_residual
                projected_error = U_k.T @ error @ Vh_k.T
                loss            = torch.linalg.norm(projected_error)

            current_loss = loss.item()

            # Threshold-based improvement check (matching FP8 / INT8 original).
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                    improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:
                    improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss  = current_loss
                best_W_q   = W_q.clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter    += 1

            # LR schedule.
            if schedule == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(
                        f"      [LR] Plateau {plateau_counter}/{effective_patience}"
                        " reached. Decaying."
                    )
                    if curr_lr > self.lr_min:
                        old_lr  = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                        debug(
                            f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e}"
                            f" (Factor: {effective_factor:.4f})"
                        )
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(
                            f"      [LR] Waiting: {plateau_counter}/{effective_patience}"
                            f" (Loss: {current_loss:.3e})"
                        )
            else:  # adaptive
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (N, K), self.early_stop_lr,
                )
                if lr_updated:
                    curr_lr = new_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Postfix: always show worse_count (matching INT8 original).
            pbar.set_postfix({
                "loss":        f"{current_loss:.3e}",
                "best":        f"{best_loss:.3e}",
                "lr":          f"{curr_lr:.2e}",
                "worse_count": f"{worse_loss_counter}",
            })

            # Compound early-stop with categorised messages
            # (matching FP8 / INT8 original in learned_rounding.py).
            if (
                current_loss <= self.early_stop_loss
                or curr_lr   <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if (
                    curr_lr <= self.early_stop_lr * 1.75
                    and worse_loss_counter > self.early_stop_stall * 0.95
                ):
                    info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif (
                    current_loss <= self.early_stop_loss
                    and curr_lr  <= self.early_stop_lr * 1.75
                ):
                    info("\n      - Learning Rate has bottomed out and loss is negligible. Stopping.")
                elif (
                    worse_loss_counter > self.early_stop_stall * 0.95
                    and current_loss   > self.early_stop_loss * 2
                ):
                    info("\n      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping.")
                elif curr_lr <= self.early_stop_lr:
                    info("\n      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping.")
                break

            # Gradient step.
            # dL/dW_q = dL/dW_dq * dW_dq/dW_q = grad_direction * wscales
            # (same chain rule as INT8 blockwise in _optimize_int8_original)
            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                grad_in_Wq = (
                    grad_direction.view(N, K // G, G)
                    * wscales.unsqueeze(-1)
                ).view(N, K)
                W_q = W_q - curr_lr * grad_in_Wq
                W_q = W_q.clamp(-_INT4_QMAX, _INT4_QMAX)

        pbar.close()

        final_W_q = best_W_q if best_W_q is not None else W_q
        return final_W_q.round().clamp(-_INT4_QMAX, _INT4_QMAX).to(torch.int8).view(N, K)

    # ------------------------------------------------------------------
    # AdamW
    # ------------------------------------------------------------------

    def _optimize_adamw(
        self,
        W_residual: torch.Tensor,
        wscales: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        N, K = W_residual.shape
        G = self.group_size

        # Snap to int4 grid (non-zero starting loss), keep as float32.
        W_groups  = W_residual.view(N, K // G, G)
        W_q_init  = (
            (W_groups / wscales.unsqueeze(-1))
            .clamp(-_INT4_QMAX, _INT4_QMAX)
            .round()
            .float()
            .view(N, K)
        )

        delta     = torch.zeros_like(W_q_init, requires_grad=True)
        curr_lr   = self.lr
        optimizer = AdamW([delta], lr=curr_lr)

        schedule          = self.lr_schedule
        best_loss         = float("inf")
        best_delta        = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter   = 0
        cooldown_counter  = 0

        # Bare plateau params — no shape-aware scaling (matching INT8 autograd).
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing SVDQuant W4A4 (AdamW-{schedule})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            W_q_refined = W_q_init + delta
            current_dq  = self._dequantize_blockwise(W_q_refined, wscales, N, K)
            error       = current_dq - W_residual
            projected_error = U_k.T @ error @ Vh_k.T
            loss        = torch.linalg.norm(projected_error)
            loss.backward()
            optimizer.step()

            current_loss_val   = loss.item()
            prev_worse_counter = worse_loss_counter
            improved           = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss          = current_loss_val
                best_delta         = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter    = 0
            else:
                worse_loss_counter += 1
                plateau_counter    += 1

            if schedule == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
            elif schedule == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= self.lr_patience:
                    debug(
                        f"      [LR] Plateau {plateau_counter}/{self.lr_patience}"
                        " reached. Decaying."
                    )
                    if curr_lr > self.lr_min:
                        old_lr  = curr_lr
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                        debug(
                            f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e}"
                            f" (Factor: {self.lr_factor:.4f})"
                        )
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(
                            f"      [LR] Waiting: {plateau_counter}/{self.lr_patience}"
                            f" (Loss: {current_loss_val:.3e})"
                        )
            else:  # adaptive
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (N, K), self.early_stop_lr,
                )
                if lr_updated:
                    curr_lr = new_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            if schedule == "plateau":
                pbar.set_postfix({
                    "loss":    f"{current_loss_val:.3e}",
                    "best":    f"{best_loss:.3e}",
                    "lr":      f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{self.lr_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss":        f"{current_loss_val:.3e}",
                    "best":        f"{best_loss:.3e}",
                    "lr":          f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                })

            # Simple early-stop checking best_loss (matching INT8 / MXFP8 autograd).
            if (
                best_loss          <= self.early_stop_loss
                or curr_lr         <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        final = (W_q_init + best_delta).clamp(-_INT4_QMAX, _INT4_QMAX).detach()
        return final.round().to(torch.int8).view(N, K)

    # ------------------------------------------------------------------
    # RAdam
    # ------------------------------------------------------------------

    def _optimize_radam(
        self,
        W_residual: torch.Tensor,
        wscales: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        N, K = W_residual.shape
        G = self.group_size

        W_groups  = W_residual.view(N, K // G, G)
        W_q_init  = (
            (W_groups / wscales.unsqueeze(-1))
            .clamp(-_INT4_QMAX, _INT4_QMAX)
            .round()
            .float()
            .view(N, K)
        )

        delta     = torch.zeros_like(W_q_init, requires_grad=True)
        curr_lr   = self.lr
        optimizer = RAdam([delta], lr=curr_lr)

        schedule          = self.lr_schedule
        best_loss         = float("inf")
        best_delta        = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter   = 0
        cooldown_counter  = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing SVDQuant W4A4 (RAdam-{schedule})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            W_q_refined = W_q_init + delta
            current_dq  = self._dequantize_blockwise(W_q_refined, wscales, N, K)
            error       = current_dq - W_residual
            projected_error = U_k.T @ error @ Vh_k.T
            loss        = torch.linalg.norm(projected_error)
            loss.backward()
            optimizer.step()

            current_loss_val   = loss.item()
            prev_worse_counter = worse_loss_counter
            improved           = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss          = current_loss_val
                best_delta         = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter    = 0
            else:
                worse_loss_counter += 1
                plateau_counter    += 1

            if schedule == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
            elif schedule == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= self.lr_patience:
                    debug(
                        f"      [LR] Plateau {plateau_counter}/{self.lr_patience}"
                        " reached. Decaying."
                    )
                    if curr_lr > self.lr_min:
                        old_lr  = curr_lr
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                        debug(
                            f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e}"
                            f" (Factor: {self.lr_factor:.4f})"
                        )
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(
                            f"      [LR] Waiting: {plateau_counter}/{self.lr_patience}"
                            f" (Loss: {current_loss_val:.3e})"
                        )
            else:  # adaptive
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (N, K), self.early_stop_lr,
                )
                if lr_updated:
                    curr_lr = new_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            if schedule == "plateau":
                pbar.set_postfix({
                    "loss":    f"{current_loss_val:.3e}",
                    "best":    f"{best_loss:.3e}",
                    "lr":      f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{self.lr_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss":        f"{current_loss_val:.3e}",
                    "best":        f"{best_loss:.3e}",
                    "lr":          f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                })

            if (
                best_loss          <= self.early_stop_loss
                or curr_lr         <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        final = (W_q_init + best_delta).clamp(-_INT4_QMAX, _INT4_QMAX).detach()
        return final.round().to(torch.int8).view(N, K)

    # ------------------------------------------------------------------
    # Prodigy
    # ------------------------------------------------------------------

    def _optimize_prodigy(
        self,
        W_residual: torch.Tensor,
        wscales: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        from prodigyplus.prodigy_plus_schedulefree import ProdigyPlusScheduleFree

        N, K = W_residual.shape
        G = self.group_size

        W_groups  = W_residual.view(N, K // G, G)
        W_q_init  = (
            (W_groups / wscales.unsqueeze(-1))
            .clamp(-_INT4_QMAX, _INT4_QMAX)
            .round()
            .float()
            .view(N, K)
        )

        delta     = torch.zeros_like(W_q_init, requires_grad=True)
        curr_lr   = self.lr
        optimizer = ProdigyPlusScheduleFree(
            [delta], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed
        )

        schedule          = self.lr_schedule
        best_loss         = float("inf")
        best_delta        = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter   = 0
        cooldown_counter  = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing SVDQuant W4A4 (Prodigy-{schedule})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            W_q_refined = W_q_init + delta
            current_dq  = self._dequantize_blockwise(W_q_refined, wscales, N, K)
            error       = current_dq - W_residual
            projected_error = U_k.T @ error @ Vh_k.T
            loss        = torch.linalg.norm(projected_error)
            loss.backward()
            optimizer.step()

            current_loss_val   = loss.item()
            prev_worse_counter = worse_loss_counter
            improved           = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss          = current_loss_val
                best_delta         = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter    = 0
            else:
                worse_loss_counter += 1
                plateau_counter    += 1

            if schedule == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
            elif schedule == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= self.lr_patience:
                    debug(
                        f"      [LR] Plateau {plateau_counter}/{self.lr_patience}"
                        " reached. Decaying."
                    )
                    if curr_lr > self.lr_min:
                        old_lr  = curr_lr
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                        debug(
                            f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e}"
                            f" (Factor: {self.lr_factor:.4f})"
                        )
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(
                            f"      [LR] Waiting: {plateau_counter}/{self.lr_patience}"
                            f" (Loss: {current_loss_val:.3e})"
                        )
            else:  # adaptive
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (N, K), self.early_stop_lr,
                )
                if lr_updated:
                    curr_lr = new_lr
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            if schedule == "plateau":
                pbar.set_postfix({
                    "loss":    f"{current_loss_val:.3e}",
                    "best":    f"{best_loss:.3e}",
                    "lr":      f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{self.lr_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss":        f"{current_loss_val:.3e}",
                    "best":        f"{best_loss:.3e}",
                    "lr":          f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                })

            if (
                best_loss          <= self.early_stop_loss
                or curr_lr         <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        final = (W_q_init + best_delta).clamp(-_INT4_QMAX, _INT4_QMAX).detach()
        return final.round().to(torch.int8).view(N, K)
