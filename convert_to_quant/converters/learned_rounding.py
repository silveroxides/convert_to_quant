"""
Learned rounding converter for FP8 and INT8 quantization.

This module implements advanced quantization using learned adaptive rounding
with SVD-based optimization. Inherits from BaseLearnedConverter.
"""

import gc
import math
from typing import (
    Dict,
    Optional,
    Tuple,
)

import torch
from torch.optim import (
    AdamW,
    RAdam,
)
from tqdm import tqdm

from ..comfy.quant_ops import BlockWiseINT8Layout
from ..constants import (
    COMPUTE_DTYPE,
    FP8_MAX,
    INT8_SYMMETRIC_MAX,
    SCALE_DTYPE,
    TARGET_FP8_DTYPE,
    TARGET_INT8_DTYPE,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import (
    debug,
    info,
    verbose,
)
from .base_converter import BaseLearnedConverter


class LearnedRoundingConverter(BaseLearnedConverter):
    """
    Learned rounding converter for FP8 and INT8 quantization.

    Inherits shared infrastructure from BaseLearnedConverter.
    Adds format-specific: target_format, scaling_mode, block_size.
    """

    def __init__(
        self,
        scaling_mode: str = "tensor",
        block_size: int = 64,
        target_format: str = "fp8",
        lr: float = 1.0,
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: Optional[str] = None,
        lora_ar_threshold: float = 0.0,
        convrot: bool = False,
        convrot_group_size: int = 256,
        dynamic_convrot: bool = False,
        scale_optimization: str = "fixed",
        **kwargs,
    ):
        """
        Initialize FP8/INT8 converter.

        Args:
            scaling_mode: Scale granularity ("tensor", "row", "block")
            block_size: Block size for block-wise scaling (default 64)
            target_format: Target format ("fp8" or "int8")
            convrot: Enable Hadamard rotation for INT8 row-wise quantization
            convrot_group_size: Group size for ConvRot (default 256)
            scale_optimization: Scale optimization mode (default "fixed")
            **kwargs: All other args passed to BaseLearnedConverter
        """
        super().__init__(
            lr=lr, extract_lora=extract_lora, lora_rank=lora_rank, lora_depth=lora_depth, lora_target=lora_target,
            lora_ar_threshold=lora_ar_threshold, **kwargs
        )

        self.block_size = block_size
        self.target_format = target_format
        self.convrot = convrot
        self.convrot_group_size = convrot_group_size
        self.dynamic_convrot = dynamic_convrot
        if self.dynamic_convrot:
            self.convrot = True
        self.scale_optimization = scale_optimization
        self.has_bias = True

        # INT8 defaults to block-wise scaling, but allows tensor-wise and row-wise
        if target_format == "int8" and scaling_mode not in ("tensor", "row", "block"):
            scaling_mode = "block"
        # Normalize block3d alias to block
        if scaling_mode == "block3d":
            scaling_mode = "block"
        self.scaling_mode = scaling_mode

        # Check ConvRot validity
        if self.convrot and not (self.target_format == "int8" and self.scaling_mode == "row"):
            verbose("  - WARNING: ConvRot is currently only supported for INT8 row-wise quantization. It will be ignored.")
            self.convrot = False
            self.dynamic_convrot = False

        # Set format-specific max values and dtype
        if self.target_format == "int8":
            self.target_dtype = TARGET_INT8_DTYPE
            self.f8_max_val = None
        else:
            self.target_dtype = TARGET_FP8_DTYPE
            self.f8_max_val = FP8_MAX

        verbose(f"LearnedRoundingConverter initialized on device: {self.device}")
        verbose(f"  - Target format: {self.target_format}")
        verbose(
            f"  - Using optimizer: '{self.optimizer_choice}'" +
            (" (disabled - simple quant)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            verbose(f"  - LR schedule: {self.lr_schedule}")
        verbose(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode in ("block", "block2d", "block3d"):
            verbose(f"    - Block size: {self.block_size}")
        if self.convrot:
            verbose(f"  - ConvRot (Hadamard rotation) enabled: group_size={self.convrot_group_size}")

        self.calib_scale = 1.0

    def _optimize_adamw(
        self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor
    ) -> torch.Tensor:
        """FP8 optimization using AdamW optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        W_scaled = W_float32 * scale
        if self.target_format == "int8":
            W_rounded = W_scaled.round().to(self.target_dtype).to(COMPUTE_DTYPE)
        else:
            W_rounded = W_scaled.to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        curr_lr = self.lr
        optimizer = AdamW([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Shape-aware plateau parameters
        effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(
            W_float32.shape[0], W_float32.shape[1]
        )

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (AdamW-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Schedule-appropriate postfix: show plateau counter or worse counter
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions
            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_radam(
        self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor
    ) -> torch.Tensor:
        """FP8 optimization using RAdam optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        W_scaled = W_float32 * scale
        if self.target_format == "int8":
            W_rounded = W_scaled.round().to(self.target_dtype).to(COMPUTE_DTYPE)
        else:
            W_rounded = W_scaled.to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        curr_lr = self.lr
        optimizer = RAdam([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Shape-aware plateau parameters
        effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(
            W_float32.shape[0], W_float32.shape[1]
        )

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (RAdam-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Schedule-appropriate postfix: show plateau counter or worse counter
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions
            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_prodigy(
        self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor
    ) -> torch.Tensor:
        """FP8 optimization using ProdigyPlusScheduleFree optimizer."""
        from prodigyplus.prodigy_plus_schedulefree import (
            ProdigyPlusScheduleFree,
        )

        M, N = W_float32.shape
        W_scaled = W_float32 * scale
        if self.target_format == "int8":
            W_rounded = W_scaled.round().to(self.target_dtype).to(COMPUTE_DTYPE)
        else:
            W_rounded = W_scaled.to(self.target_dtype).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        curr_lr = self.lr
        optimizer = ProdigyPlusScheduleFree([delta], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Shape-aware plateau parameters
        effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(
            W_float32.shape[0], W_float32.shape[1]
        )

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (Prodigy-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Manual LR update based on schedule
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0  # Keep inactive
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_original(
        self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor
    ) -> torch.Tensor:
        W_scaled = W_float32 * scale
        if self.target_format == "int8":
            W_rounded = W_scaled.round().to(self.target_dtype).to(COMPUTE_DTYPE)
        else:
            W_rounded = W_scaled.to(self.target_dtype).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        plateau_counter = 0  # For plateau schedule
        cooldown_counter = 0  # For plateau cooldown
        curr_lr = self.lr
        # Tensor dimensions for adaptive LR schedule
        M, N = W_float32.shape[0], W_float32.shape[1]

        schedule_name = self.lr_schedule

        # Shape-aware plateau parameters
        rows, cols = W_float32.shape
        aspect_ratio = max(rows, cols) / min(rows, cols)

        if schedule_name == "plateau" and self.lr_shape_influence > 0:
            # Scale factor based on aspect ratio, modulated by influence
            # influence=1.0: full effect, influence=0.0: no effect (use raw values)
            # Elongated tensors need MORE AGGRESSIVE decay (lower factor)
            ar_factor = math.sqrt(aspect_ratio)  # e.g., 1.0 for square, 2.0 for AR=4
            blend = self.lr_shape_influence

            # Keep patience unchanged per user feedback
            effective_patience = self.lr_patience

            # More aggressive factor for elongated tensors: factor^ar_factor makes it smaller
            # E.g., 0.92^2 = 0.846 for AR=4, 0.92^2.45 = 0.808 for AR=6
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend

            # Cooldown unchanged
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (Original-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            with torch.no_grad():
                current_dq = W_q_refined / scale
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()
            # Check if improvement exceeds threshold (supports rel/abs mode like PyTorch ReduceLROnPlateau)
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                    # Relative: significant if loss < best * (1 - threshold)
                    improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:  # 'abs'
                    # Absolute: significant if improvement > threshold
                    improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            # Store counter before potential reset (for no-reset adaptive mode)
            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss = current_loss
                best_tensor = W_q_refined.clone()
                plateau_counter = 0
                worse_loss_counter = 0
                # no-reset mode: worse_loss_counter preserved for tier calculation
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update based on schedule
            if schedule_name == "exponential":
                # ExponentialLR: lr = lr * gamma per step
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
                # ReduceLROnPlateau with cooldown (shape-aware)
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    # Debug log to track patience accumulation
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr

                # Reset counter after boost in no-reset mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Show schedule-appropriate metric in progress bar
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions (configurable thresholds)
            if current_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr * 1.75 and worse_loss_counter > self.early_stop_stall * 0.95:
                    info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif current_loss <= self.early_stop_loss and curr_lr <= self.early_stop_lr * 1.75:
                    info("\n      - Learning Rate has bottomed out and loss is negligible. Stopping.")
                elif worse_loss_counter > self.early_stop_stall * 0.95 and current_loss > self.early_stop_loss * 2:
                    info("\n      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping.")
                elif curr_lr <= self.early_stop_lr:
                    info("\n      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping.")
                break

            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * (grad_direction * scale)

        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(
        self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1, calibration_data: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        has_bias = kwargs.get("has_bias", True)
        self.has_bias = has_bias
        self._current_extra_tensors = {}

        # 1. Initialize State
        attempt = 1
        max_attempts = 10
        orig_top_p = self.top_p
        orig_max_k = self.max_k
        orig_min_k = self.min_k
        self.calib_scale = 1.0

        try:
            while True:
                try:
                    # Clear current extra tensors at start of attempt
                    self._current_extra_tensors = {}

                    W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

                    # Determine if we should optimize
                    if torch.all(W_float32 == 0):
                        verbose("  - Tensor is all zeros, skipping optimization.")
                        quantized_tensor = torch.zeros_like(W_float32, dtype=self.target_dtype)
                        dequant_scale = None

                        if W_float32.ndim == 2:
                            out_features, in_features = W_float32.shape

                            if self.target_format == "int8":
                                # INT8 uses 2D block scaling (M//block_size, N//block_size)
                                num_blocks_m = out_features // self.block_size
                                num_blocks_n = in_features // self.block_size
                                dequant_scale = torch.ones(num_blocks_m, num_blocks_n, device=self.device, dtype=SCALE_DTYPE)
                            elif self.scaling_mode == "row":
                                # Row-wise: one scale per row
                                dequant_scale = torch.ones(out_features, device=self.device, dtype=SCALE_DTYPE)
                            elif self.scaling_mode in (
                                "block", "block2d"
                            ) and out_features % self.block_size == 0 and in_features % self.block_size == 0:
                                # 2D block-wise: (M//bs, N//bs) - 'block' is primary, 'block2d' deprecated alias
                                num_blocks_m = out_features // self.block_size
                                num_blocks_n = in_features // self.block_size
                                dequant_scale = torch.ones(num_blocks_m, num_blocks_n, device=self.device, dtype=SCALE_DTYPE)
                            elif self.scaling_mode == "block3d" and in_features > 0 and in_features % self.block_size == 0:
                                # Per-row-group 3D: (out_features, num_blocks, 1)
                                num_blocks = in_features // self.block_size
                                dequant_scale = torch.ones(out_features, num_blocks, 1, device=self.device, dtype=SCALE_DTYPE)
                            else:
                                # Tensor-wise: single scale
                                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
                        else:
                            dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)

                        return quantized_tensor, dequant_scale, torch.zeros_like(W_float32), {}

                    # INT8 quantization path
                    if self.target_format == "int8":
                        if self.scaling_mode in ("tensor", "row"):
                            qdata, scale, dequantized = self._convert_int8_tensorwise(
                                W_float32, calibration_data=calibration_data
                            )
                        else:
                            qdata, scale, dequantized = self._convert_int8(W_float32)
                    else:
                        # FP8 quantization path - route based on scaling_mode
                        if self.scaling_mode == "row":
                            qdata, scale, dequantized = self._convert_fp8_rowwise(W_float32)
                        elif self.scaling_mode in ("block", "block2d"):
                            # 2D block-wise - 'block' is primary, 'block2d' is deprecated alias
                            qdata, scale, dequantized = self._convert_fp8_block2d(W_float32)
                        elif self.scaling_mode == "block3d":
                            # 3D per-row-group mode (legacy)
                            qdata, scale, dequantized = self._convert_fp8(W_float32)
                        else:
                            # 'tensor' mode
                            qdata, scale, dequantized = self._convert_fp8(W_float32)

                    # Error Correction LoRA extraction
                    extra_tensors = self._current_extra_tensors.copy()
                    self._current_extra_tensors.clear()
                    if self._should_extract_lora(key, W_orig.shape, depth):
                        lora_data = self._extract_error_lora(W_float32, dequantized)
                        if lora_data:
                            extra_tensors.update(lora_data)

                    return qdata, scale, dequantized, extra_tensors

                except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                    is_oom = isinstance(e, torch.cuda.OutOfMemoryError) or (
                        isinstance(e, RuntimeError)
                        and any(msg in str(e).lower() for msg in ["out of memory", "cuda out of memory", "oom"])
                    )
                    if not is_oom:
                        raise e

                    verbose(f"    - [OOM Warning] Out of memory during layer conversion (attempt {attempt}/{max_attempts}).")

                    # Perform aggressive cleanup
                    try:
                        del W_float32
                    except NameError:
                        pass
                    try:
                        del qdata
                    except NameError:
                        pass
                    try:
                        del scale
                    except NameError:
                        pass
                    try:
                        del dequantized
                    except NameError:
                        pass

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    # Shrink Parameters
                    self.top_p *= 0.7
                    self.max_k = int(self.max_k * 0.7)
                    self.min_k = int(self.min_k * 0.7)
                    self.calib_scale *= 0.5

                    verbose(
                        f"    - [OOM Warning] Reduced parameters: top_p={self.top_p:.4f}, max_k={self.max_k}, min_k={self.min_k}, calib_scale={self.calib_scale:.4f}"
                    )

                    # Check for fatal failure (too many attempts or params reached floor)
                    if attempt >= max_attempts or (self.max_k < 1 and self.min_k < 1 and self.top_p < 1e-4):
                        verbose(
                            f"    - [OOM Error] OOM mitigation failed (attempt {attempt}/{max_attempts}, max_k: {self.max_k}). Re-raising OOM."
                        )
                        raise e

                    attempt += 1

        finally:
            # Restore State
            self.top_p = orig_top_p
            self.max_k = orig_max_k
            self.min_k = orig_min_k
            self.calib_scale = 1.0

    def _convert_int8(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        INT8 block-wise quantization using BlockWiseINT8Layout or Lode-Wise kernels.

        INT8 block-wise quantization differs from FP8:
        - Uses symmetric quantization with range [-127, 127]
        - Scale is per-block (2D grid): shape (M//block_size, N//block_size)
        - Requires dimensions divisible by block_size
        """
        M, N = W_float32.shape

        # Validate dimensions are divisible by block_size
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(
                f"INT8 block-wise quantization requires dimensions divisible by block_size={self.block_size}. Got shape ({M}, {N}). Consider using --skip_inefficient_layers or a different block_size."
            )

        # Select quantization backend
        # Use BlockWiseINT8Layout (blockwise backend from quant_ops.py)
        qdata, layout_params = BlockWiseINT8Layout.quantize(W_float32, block_size=self.block_size, is_weight=True)
        scale = layout_params["scale"]  # Shape: (M//block_size, N//block_size)

        # Optional: Apply learned rounding optimization for INT8
        # INT8 Specific Optimization Logic
        if not self.no_learned_rounding and self.num_iter > 0:
            verbose("    - Applying learned rounding optimization for INT8...")
            qdata, scale = self._optimize_int8_learned_rounding(W_float32, qdata, scale)

        # Dequantize to get the reconstructed weight for bias correction
        dequantized_weight = BlockWiseINT8Layout.dequantize(
            qdata, scale, self.block_size, is_weight=True, orig_dtype=COMPUTE_DTYPE
        )

        # Clean up
        del W_float32
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight)

    def _convert_int8_tensorwise(
        self, W_float32: torch.Tensor, calibration_data: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        INT8 tensor-wise/row-wise quantization.

        Uses TensorWiseINT8Layout which handles both global and per-row scales.
        """
        from ..comfy.quant_ops import TensorWiseINT8Layout
        from ..utils.convrot import (
            build_hadamard,
            rotate_weight,
        )

        # Apply ConvRot if enabled and we're doing row-wise quantization
        convrot_applied = False
        layer_group_size = self.convrot_group_size
        if self.convrot and self.scaling_mode == "row":
            M, N = W_float32.shape
            if self.dynamic_convrot:
                from ..utils.convrot import find_max_compatible_group_size
                layer_group_size = find_max_compatible_group_size(N, min_group_size=self.convrot_group_size)

            # Only apply if in_features is divisible by the group size
            if layer_group_size is not None and N % layer_group_size == 0:
                try:
                    H = build_hadamard(layer_group_size, device=self.device, dtype=COMPUTE_DTYPE)
                    W_float32 = rotate_weight(W_float32, H, layer_group_size)
                    verbose(f"    - Applied ConvRot Hadamard rotation (group_size={layer_group_size}).")
                    convrot_applied = True
                except Exception as e:
                    verbose(f"    - WARNING: Failed to apply ConvRot: {e}")
            else:
                verbose(
                    f"    - WARNING: Skipping ConvRot: in_features ({N}) not divisible by group_size ({layer_group_size})."
                )

        # Phase 2: Calibration Data Management
        X_rot, Y_ref, H_mat = None, None, None
        if self.convrot and self.scaling_mode == "row" and convrot_applied:
            from ..utils.tensor_utils import (
                prepare_calibration_data,
            )

            X_rot, Y_ref, H_mat = prepare_calibration_data(
                W_float32, calibration_data, True, layer_group_size, self.device, COMPUTE_DTYPE,
                calib_scale=self.calib_scale
            )
            verbose("    - Executed Phase 2: Calibration Data Management (Captured X, rotated X, computed reference Y)")

        # Initial quantization
        # We need to manually handle tensor-wise vs row-wise if auto-quantizing
        if self.scaling_mode == "tensor":
            # Global scale
            w_max = W_float32.abs().max()
            dequant_scale = w_max.clamp_min(1e-12) / 127.0
            # Pass the pre-computed scale to quantize
            qdata, layout_params = TensorWiseINT8Layout.quantize(W_float32, scale=dequant_scale, is_weight=True)
            scale = dequant_scale
        else:
            # Row-wise (default for TensorWiseINT8Layout if is_weight=True)
            qdata, layout_params = TensorWiseINT8Layout.quantize(W_float32, is_weight=True)
            scale = layout_params["scale"]

        # Optional: Apply learned rounding optimization for INT8
        if not self.no_learned_rounding and self.num_iter > 0:
            verbose(f"    - Applying learned rounding optimization for INT8 ({self.scaling_mode}-wise)...")
            if self.scaling_mode == "tensor":
                qdata, scale = self._optimize_int8_tensorwise_learned_rounding(W_float32, qdata, scale)
            elif self.convrot and self.scaling_mode == "row" and X_rot is not None:
                if self.scale_optimization == "dualround":
                    verbose("    - Scale Optimization: DUALROUND (Pass 1)")
                    qdata, scale = self._optimize_int8_adaround(W_float32, qdata, scale, X_rot, Y_ref)

                    # Scale Re-Estimation
                    verbose("    - Scale Optimization: Re-estimating scales based on Pass 1 output...")
                    dequant_opt = TensorWiseINT8Layout.dequantize(qdata, scale, orig_dtype=COMPUTE_DTYPE)
                    row_max_opt = dequant_opt.abs().amax(dim=1, keepdim=True)
                    scale_opt = row_max_opt.clamp_min(1e-12) / 127.0
                    qdata, _ = TensorWiseINT8Layout.quantize(W_float32, scale=scale_opt, is_weight=True)
                    scale = scale_opt.squeeze(1) if scale.dim() == 1 else scale_opt

                    # Clean up Pass 1 intermediate tensors immediately to prevent VRAM accumulation
                    del dequant_opt, row_max_opt, scale_opt
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    verbose("    - Scale Optimization: DUALROUND (Pass 2)")
                    qdata, scale = self._optimize_int8_adaround(W_float32, qdata, scale, X_rot, Y_ref)
                else:
                    qdata, scale = self._optimize_int8_adaround(W_float32, qdata, scale, X_rot, Y_ref)
            else:
                qdata, scale = self._optimize_int8_learned_rounding(W_float32, qdata, scale, scaling_mode="row")

        # Dequantize for bias correction
        dequantized_weight = TensorWiseINT8Layout.dequantize(qdata, scale, orig_dtype=COMPUTE_DTYPE)

        # Phase 4: Residual Bias Calibration
        if self.has_bias and self.convrot and self.scaling_mode == "row" and X_rot is not None and Y_ref is not None:
            with torch.no_grad():
                Y_quant = X_rot @ dequantized_weight.T
                bias_adj = (Y_ref - Y_quant).mean(dim=0)
                self._current_extra_tensors["bias_correction"] = bias_adj.cpu()
                verbose(
                    f"    - Phase 4: Residual Bias Calibration (Computed mean delta of output activations, bias correction norm: {bias_adj.norm().item():.6f})"
                )

        # Clean up
        self._cleanup_tensors(W_float32)
        if X_rot is not None or Y_ref is not None or H_mat is not None:
            self._cleanup_tensors(X_rot, Y_ref, H_mat)

        return (qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight)

    # _convert_int8_rowwise merged into _convert_int8_tensorwise

    def _optimize_int8_tensorwise_learned_rounding(self, W_float32: torch.Tensor, qdata: torch.Tensor,
                                                   scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned rounding optimization for INT8 tensor-wise quantization.
        """
        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Reuse FP8 optimizer logic but adapted for INT8 range
        # We need to pass the scale in a way that _optimize_* can handle it.
        # For INT8: dq = Q * scale
        # For FP8: dq = Q / scale_fp8 => scale_fp8 = 1/scale_int8
        scale_fp8_style = 1.0 / scale.clamp_min(1e-12)

        # Temporary switch to FP8-like mode for optimization
        orig_dtype = self.target_dtype
        orig_max = self.f8_max_val
        self.target_dtype = TARGET_INT8_DTYPE
        self.f8_max_val = float(INT8_SYMMETRIC_MAX)

        if self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale_fp8_style, U_k, Vh_k)
        elif self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_fp8_style, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale_fp8_style, U_k, Vh_k)
        elif self.optimizer_choice == "prodigy":
            final_tensor_scaled = self._optimize_prodigy(W_float32, scale_fp8_style, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        # Restore original state
        self.target_dtype = orig_dtype
        self.f8_max_val = orig_max

        # Extract quantized data and final scale
        with torch.no_grad():
            # final_tensor_scaled is W * scale_fp8_style = W / scale_int8
            final_qdata = final_tensor_scaled.clamp(-127, 127).round().to(TARGET_INT8_DTYPE)

        self._cleanup_tensors(U_k, Vh_k)

        return final_qdata, scale

    def _int8_dequantize_blockwise(
        self, qdata: torch.Tensor, scale: torch.Tensor, M: int, N: int, block_size: int
    ) -> torch.Tensor:
        """
        Differentiable block-wise INT8 dequantization for optimization.
        Matches BlockWiseINT8Layout._weight_quantize_pytorch logic.

        Args:
            qdata: Quantized values (can be float during optimization), shape (M, N)
            scale: Per-block scales, shape (M//block_size, N//block_size)
            M, N: Original tensor dimensions
            block_size: Block size for quantization

        Returns:
            Dequantized tensor, shape (M, N)
        """
        # Reshape to blocks: (M//bs, bs, N//bs, bs)
        q_blocked = qdata.reshape(M // block_size, block_size, N // block_size, block_size)
        # Permute to: (M//bs, N//bs, bs, bs)
        q_blocked = q_blocked.permute(0, 2, 1, 3)
        # Broadcast scale: (M//bs, N//bs, 1, 1)
        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        # Apply scale
        dequantized = q_blocked * scale_broadcast
        # Permute back and reshape: (M, N)
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    def _int8_dequantize_rowwise(self, qdata: torch.Tensor, scale: torch.Tensor, M: int, N: int) -> torch.Tensor:
        """
        Differentiable row-wise INT8 dequantization for optimization.

        Args:
            qdata: Quantized values (can be float during optimization), shape (M, N)
            scale: Per-row scales, shape (M, 1) or (M,)
            M, N: Original tensor dimensions

        Returns:
            Dequantized tensor, shape (M, N)
        """
        if scale.dim() == 1:
            scale_broadcast = scale.unsqueeze(1)
        else:
            scale_broadcast = scale

        dequantized = qdata * scale_broadcast
        return dequantized

    def _optimize_int8_adaround(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, X_rot: torch.Tensor, Y_ref: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply SVD-guided AdaRound (learned soft rounding) optimization over the rotated parameter space
        minimizing local activation reconstruction MSE using the cached calibration data.
        """
        M, N = W_float32.shape

        # 1. Compute SVD components of rotated parameter matrix
        U_k, Vh_k, k = self._compute_svd_components(W_float32, verbose=True)

        # 2. Setup soft rounding parameters V
        # W_scaled = W_rot / scale_row
        scale_broadcast = scale.unsqueeze(1) if scale.dim() == 1 else scale
        W_scaled = W_float32 / scale_broadcast.clamp_min(1e-12)
        W_floor = W_scaled.floor()

        # Target fraction for soft rounding
        target = W_scaled - W_floor
        target = torch.clamp(target, min=1e-6, max=1.0 - 1e-6)

        # Temperature schedule for sigmoid (AdaRound paper: start soft, sharpen over time)
        T_start, T_end = 20.0, 2.0
        V_init = -torch.log((1.0 / target) - 1.0) * T_start
        V = V_init.clone().detach().requires_grad_(True)

        # 3. Setup optimizer
        curr_lr = self.lr
        if self.optimizer_choice == "adamw":
            optimizer = AdamW([V], lr=curr_lr)
        elif self.optimizer_choice == "radam":
            optimizer = RAdam([V], lr=curr_lr)
        elif self.optimizer_choice == "prodigy":
            from prodigyplus.prodigy_plus_schedulefree import (
                ProdigyPlusScheduleFree,
            )

            optimizer = ProdigyPlusScheduleFree([V], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed)
        else:
            optimizer = None  # Will use manual SGD on V

        # 4. Compute initial metrics for dynamic self-tuning regularization
        with torch.no_grad():
            init_W_q_rounded = qdata.to(COMPUTE_DTYPE)
            init_W_rounded_dequant = init_W_q_rounded * scale_broadcast
            init_mse_rounded = torch.nn.functional.mse_loss(X_rot @ init_W_rounded_dequant.T, Y_ref)
            init_svd_rounded = torch.linalg.norm(U_k.T @ (init_W_rounded_dequant - W_float32) @ Vh_k.T)

        # Regularization balance factor: ~5% of initial rounded MSE loss
        lambda_reg = 0.05 * max(init_mse_rounded.item(), 1e-5)

        # SVD regularization balance factor: ~1% of initial rounded MSE loss
        if init_svd_rounded.item() > 1e-8:
            alpha_svd = 0.01 * (init_mse_rounded.item() / init_svd_rounded.item())
        else:
            alpha_svd = 0.0

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_V = V.detach().clone()
        best_converged_ratio = 0.0
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Only compute shape-aware plateau parameters if the plateau schedule is active
        effective_patience, effective_factor, effective_cooldown = None, None, None
        if schedule_name == "plateau":
            effective_patience, effective_factor, effective_cooldown = self._compute_shape_aware_plateau_params(M, N)

        # Dynamically derive early stop parameters from existing schedule config
        decay_factor = self.lr_factor if self.lr_factor is not None else 0.95
        if decay_factor >= 1.0:
            decay_factor = 0.95

        window_size = max(5, int(2.5 / (1.0 - decay_factor)))
        loss_span_threshold = self.early_stop_loss / (1.0 - decay_factor)
        target_converged_ratio = 0.90

        loss_history = []
        pbar = tqdm(
            range(self.num_iter), desc=f"    Optimizing (AdaRound-{self.optimizer_choice}-{schedule_name})", leave=False,
            dynamic_ncols=True
        )
        for i in pbar:
            if optimizer is not None:
                optimizer.zero_grad()

            # Forward pass: Optimized soft rounding (smooth AdaRound)
            # Calculate current temperature (linear decay from T_start to T_end)
            temp = T_start + (T_end - T_start) * (i / self.num_iter)
            h_V = torch.sigmoid(V / temp)
            # Use soft weights for smooth gradient flow during optimization
            W_q = W_floor + h_V
            W_dequant = W_q * scale_broadcast

            # --- Discretization and Convergence early stopping check ---
            # Track the true physical percentage of parameters converged to strict integer boundaries (temp=1.0)
            converged_ratio = ((torch.sigmoid(V) < 0.05) | (torch.sigmoid(V) > 0.95)).float().mean().item()

            # Loss 1: Output activation MSE on soft dequantized weights
            Y_pred = X_rot @ W_dequant.T
            loss_mse = torch.nn.functional.mse_loss(Y_pred, Y_ref)

            # Loss 2: SVD-guided weight-space projection error (soft)
            weight_error = W_dequant - W_float32
            projected_error = U_k.T @ weight_error @ Vh_k.T
            loss_svd = torch.linalg.norm(projected_error)

            # Loss 3: Soft rounding binary regularizer
            loss_reg = (1.0 - (2.0 * h_V - 1.0).pow(2)).mean()

            # Total Loss - Normalized for numerical stability on real-world weights
            # We scale MSE and SVD losses so they start relative to 1.0
            loss_mse_scaled = loss_mse / max(init_mse_rounded.item(), 1e-12)

            if alpha_svd > 0 and init_svd_rounded.item() > 1e-8:
                loss_svd_scaled = loss_svd / init_svd_rounded.item()
            else:
                loss_svd_scaled = 0.0

            # Combine with fixed weights: 1.0 for MSE, 0.01 for SVD, 0.1 for Reg
            loss = loss_mse_scaled + 0.01 * loss_svd_scaled + 0.1 * loss_reg

            # Scale up loss for backpropagation to prevent float32 underflow on large layers
            scaled_loss = loss * 1e5

            if optimizer is not None:
                scaled_loss.backward()
                if V.grad is not None:
                    # Scale gradients back down before optimizer steps to protect scale-sensitive optimizers (e.g. Prodigy distance estimator)
                    V.grad.div_(1e5)
                optimizer.step()
            else:
                # Manual SGD
                if V.grad is not None:
                    V.grad.zero_()
                scaled_loss.backward()
                with torch.no_grad():
                    # Divide gradient back down to match manual learning rate scale
                    V -= curr_lr * (V.grad / 1e5)

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            # Track loss over a rolling window of iterations
            loss_history.append(current_loss_val)
            if len(loss_history) > window_size:
                loss_history.pop(0)

            # Saturated Flatline Trigger using derived parameters:
            # If >= target ratio of parameters are frozen at hard boundaries, and the loss has flatlined
            # below the mathematically derived infinite sum limit, stop early.
            if converged_ratio >= target_converged_ratio and len(loss_history) == window_size:
                loss_span = max(loss_history) - min(loss_history)
                if loss_span < loss_span_threshold:
                    verbose(f"\n      - Discretization early stop: {converged_ratio*100:.2f}% parameters converged. Loss span: {loss_span:.2e} (< {loss_span_threshold:.2e}). Stopping.")
                    break

            if improved:
                best_loss = current_loss_val
                best_V = V.detach().clone()
                best_converged_ratio = converged_ratio
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
                # no-reset mode: worse_loss_counter preserved for tier calculation
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Schedule-based learning rate adjustments
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0  # Keep inactive
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        if optimizer is not None:
                            for param_group in optimizer.param_groups:
                                param_group["lr"] = curr_lr
                        cooldown_counter = effective_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {effective_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss_val:.3e})")
            else:  # "adaptive"
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    if optimizer is not None:
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Schedule-appropriate postfix
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{effective_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions with descriptive messages
            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        # Discretize V to get final quantized integers
        with torch.no_grad():
            best_V.sigmoid_().ge_(0.5)
            W_floor.add_(best_V)
            del best_V
            final_qdata = self._finalize_int8_qdata(W_floor)

            verbose(f"    - Discretization audit: {best_converged_ratio * 100:.2f}% of parameters converged to strict boundaries.")

        self._cleanup_tensors(U_k, Vh_k, V)
        U_k = None
        Vh_k = None
        V = None
        W_scaled = None
        W_floor = None
        X_rot = None
        Y_ref = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return final_qdata, scale

    def _optimize_int8_learned_rounding(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, scaling_mode: str = "block"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned rounding optimization for INT8 quantization.
        Uses SVD-based optimization similar to FP8 but adapted for INT8.
        """
        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Route to appropriate optimizer
        if self.optimizer_choice == "original":
            final_qdata = self._optimize_int8_original(W_float32, qdata, scale, U_k, Vh_k, scaling_mode)
        elif self.optimizer_choice == "adamw":
            final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k, scaling_mode)
        elif self.optimizer_choice == "radam":
            final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k, scaling_mode)
        elif self.optimizer_choice == "prodigy":
            final_qdata = self._optimize_int8_prodigy(W_float32, qdata, scale, U_k, Vh_k, scaling_mode)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        self._cleanup_tensors(U_k, Vh_k)

        return final_qdata, scale

    def _finalize_int8_qdata(self, qdata_float: torch.Tensor) -> torch.Tensor:
        """Convert the working INT8 tensor in place to minimize peak memory."""
        with torch.no_grad():
            qdata_float.clamp_(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            qdata_float.round_()
            final_qdata = qdata_float.to(TARGET_INT8_DTYPE)
        del qdata_float
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        return final_qdata

    def _optimize_int8_adamw(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor,
        scaling_mode: str = "block"
    ) -> torch.Tensor:
        """INT8 optimization using AdamW optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)

        curr_lr = self.lr
        optimizer = AdamW([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (AdamW-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta

            if scaling_mode == "block":
                current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            elif scaling_mode == "row":
                current_dq = self._int8_dequantize_rowwise(q_refined, scale, M, N)
            else:
                raise ValueError(f"Unsupported scaling mode for INT8 learned rounding: {scaling_mode}")

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0  # Keep inactive
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= self.lr_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{self.lr_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {self.lr_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{self.lr_patience} (Loss: {current_loss_val:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Schedule-appropriate postfix
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{self.lr_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions
            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        with torch.no_grad():
            qdata_float.add_(best_delta)
        del best_delta, delta
        final_qdata = self._finalize_int8_qdata(qdata_float)
        return final_qdata

    def _optimize_int8_radam(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor,
        scaling_mode: str = "block"
    ) -> torch.Tensor:
        """INT8 optimization using RAdam optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)

        curr_lr = self.lr
        optimizer = RAdam([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing INT8 (RAdam-{schedule_name})", leave=False, dynamic_ncols=True)
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta

            if scaling_mode == "block":
                current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            elif scaling_mode == "row":
                current_dq = self._int8_dequantize_rowwise(q_refined, scale, M, N)
            else:
                raise ValueError(f"Unsupported scaling mode for INT8 learned rounding: {scaling_mode}")

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0  # Keep inactive
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= self.lr_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{self.lr_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                        debug(f"      [LR] Decay: {old_lr:.2e} -> {curr_lr:.2e} (Factor: {self.lr_factor:.4f})")
                    plateau_counter = 0
                else:
                    if plateau_counter > 0:
                        debug(f"      [LR] Waiting: {plateau_counter}/{self.lr_patience} (Loss: {current_loss_val:.3e})")
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Schedule-appropriate postfix
            if schedule_name == "plateau":
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "plateau": f"{plateau_counter}/{self.lr_patience}"
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}"
                    }
                )

            # Early stopping conditions
            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        with torch.no_grad():
            qdata_float.add_(best_delta)
        del best_delta, delta
        final_qdata = self._finalize_int8_qdata(qdata_float)
        return final_qdata

    def _optimize_int8_prodigy(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor,
        scaling_mode: str = "block"
    ) -> torch.Tensor:
        """INT8 optimization using ProdigyPlusScheduleFree optimizer."""
        from prodigyplus.prodigy_plus_schedulefree import (
            ProdigyPlusScheduleFree,
        )

        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)

        curr_lr = self.lr
        optimizer = ProdigyPlusScheduleFree([delta], lr=curr_lr, use_schedulefree=False, use_speed=self.use_speed)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(
            range(self.num_iter), desc=f"    Optimizing INT8 (Prodigy-{schedule_name})", leave=False, dynamic_ncols=True
        )
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta

            if scaling_mode == "block":
                current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
            elif scaling_mode == "row":
                current_dq = self._int8_dequantize_rowwise(q_refined, scale, M, N)
            else:
                raise ValueError(f"Unsupported scaling mode for INT8 learned rounding: {scaling_mode}")

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Prodigy Warm-up: Skip LR decay for first 50 iterations
            prodigy_warmup = self.optimizer_choice == "prodigy" and i < 50

            # Manual LR update based on schedule
            if schedule_name == "exponential":
                if not prodigy_warmup:
                    curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if prodigy_warmup:
                    plateau_counter = 0  # Keep inactive
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - cosine-based schedule
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix({"loss": f"{current_loss_val:.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}"})

            if best_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr:
                    info("\n      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping early.")
                elif best_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        with torch.no_grad():
            qdata_float.add_(best_delta)
        del best_delta, delta
        final_qdata = self._finalize_int8_qdata(qdata_float)
        return final_qdata

    def _optimize_int8_original(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor,
        scaling_mode: str = "block"
    ) -> torch.Tensor:
        """INT8 optimization using original gradient-based optimizer (no autograd)."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        q_refined = qdata_float.clone()

        # Compute global max to precondition gradient magnitude (matches tensor-wise step size)
        w_max = W_float32.abs().max().clamp_min(1e-12)
        grad_scale = 127.0 / w_max

        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        plateau_counter = 0  # For plateau schedule
        cooldown_counter = 0  # For plateau cooldown
        curr_lr = self.lr

        schedule_name = self.lr_schedule

        # Shape-aware plateau parameters
        aspect_ratio = max(M, N) / min(M, N)

        if schedule_name == "plateau" and self.lr_shape_influence > 0:
            # Scale factor based on aspect ratio, modulated by influence
            # Elongated tensors need MORE AGGRESSIVE decay (lower factor)
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence

            # Patience unchanged per user feedback
            effective_patience = self.lr_patience

            # More aggressive factor for elongated tensors: factor^ar_factor makes it smaller
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend

            # Cooldown unchanged
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        pbar = tqdm(
            range(self.num_iter), desc=f"    Optimizing INT8 (Original-{schedule_name})", leave=False, dynamic_ncols=True
        )
        for i in pbar:
            with torch.no_grad():
                if scaling_mode == "block":
                    current_dq = self._int8_dequantize_blockwise(q_refined, scale, M, N, block_size)
                elif scaling_mode == "row":
                    current_dq = self._int8_dequantize_rowwise(q_refined, scale, M, N)
                else:
                    raise ValueError(f"Unsupported scaling mode for INT8 learned rounding: {scaling_mode}")
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()
            # Check if improvement exceeds threshold (supports rel/abs mode like PyTorch ReduceLROnPlateau)
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                    # Relative: significant if loss < best * (1 - threshold)
                    improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:  # 'abs'
                    # Absolute: significant if improvement > threshold
                    improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            # Store counter before potential reset (for no-reset adaptive mode)
            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss = current_loss
                best_tensor = q_refined.clone()
                plateau_counter = 0
                worse_loss_counter = 0
                # no-reset mode: worse_loss_counter preserved for tier calculation
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update based on schedule
            if schedule_name == "exponential":
                # ExponentialLR: lr = lr * gamma per step
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
                # ReduceLROnPlateau with cooldown (shape-aware)
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= effective_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i, (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr

                # Reset counter after boost in no-reset mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix(
                {
                    "loss": f"{current_loss:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}"
                }
            )

            # Early stopping conditions (configurable thresholds)
            if current_loss <= self.early_stop_loss or curr_lr <= self.early_stop_lr or worse_loss_counter > self.early_stop_stall:
                if curr_lr <= self.early_stop_lr * 1.75 and worse_loss_counter > self.early_stop_stall * 0.95:
                    info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif current_loss <= self.early_stop_loss and curr_lr <= self.early_stop_lr * 1.75:
                    info("      - Learning Rate has bottomed out and loss is negligible. Stopping.")
                elif worse_loss_counter > self.early_stop_stall * 0.95 and current_loss > self.early_stop_loss * 2:
                    info("\n      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping.")
                elif curr_lr <= self.early_stop_lr:
                    info("\n      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping.")
                break

            with torch.no_grad():
                # Gradient in quantized Q-space.
                # Use global max to scale gradient magnitude matching tensor mode.
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                q_refined -= curr_lr * (grad_direction * grad_scale)

        pbar.close()

        final_tensor = best_tensor if best_tensor is not None else q_refined
        if best_tensor is not None:
            del q_refined
        del qdata_float
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        final_qdata = self._finalize_int8_qdata(final_tensor)
        return final_qdata

    def _convert_fp8(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original FP8 quantization path."""

        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == "block":
            if W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                verbose(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(out_features, in_features)
            else:
                verbose(
                    f"    - WARNING: Tensor shape {list(W_float32.shape)} not suitable for block size {self.block_size}. Falling back to 'tensor' scaling."
                )
                current_scaling_mode = "tensor"

        if current_scaling_mode == "tensor":
            verbose(
                f"    - Using tensor-wise FP8 scaling ({self.optimizer_choice if not self.no_learned_rounding else 'simple'})."
            )
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale

        assert scale is not None, "scale should not be None after scaling mode selection"

        # Skip SVD optimization if no_learned_rounding is set
        if self.no_learned_rounding:
            verbose("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_f8 = (W_float32 * scale).clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                if compact_scale is None:
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
                else:
                    if current_scaling_mode == "block":
                        dequant_scale = compact_scale.reciprocal()
                    else:
                        dequant_scale = compact_scale.reciprocal()
                    dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
                dequantized_weight_tensor = W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
            del W_float32, scale, compact_scale
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return W_f8, dequant_scale, dequantized_weight_tensor

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "prodigy":
            final_tensor_scaled = self._optimize_prodigy(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            if compact_scale is None:
                verbose("    - WARNING: compact_scale is None, falling back to torch.ones for dequant_scale.")
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                if current_scaling_mode == "block":
                    dequant_scale = compact_scale.reciprocal()
                else:
                    dequant_scale = compact_scale.reciprocal()
                dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized_weight_tensor = W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
        del W_float32, scale, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight_tensor)

    def _convert_fp8_rowwise(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Row-wise FP8 quantization - one scale per row.

        Scale shape: (out_features,)
        Good balance between accuracy and memory for most weight matrices.
        """
        M, N = W_float32.shape
        verbose("    - Using row-wise FP8 scaling (1 scale per row).")

        # Compute per-row max
        row_max = W_float32.abs().amax(dim=1, keepdim=True)  # (M, 1)
        quant_scale = self.f8_max_val / row_max.clamp_min_(1e-12)  # (M, 1)

        if self.no_learned_rounding:
            verbose("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_scaled = W_float32 * quant_scale
                W_f8 = W_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)
                dequantized = W_f8.to(COMPUTE_DTYPE) / quant_scale

            del W_float32
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return (W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized)

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Use the appropriate optimizer with row-wise scale
        scale = quant_scale  # (M, 1) for broadcast
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "prodigy":
            final_tensor_scaled = self._optimize_prodigy(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)
            dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / quant_scale

        self._cleanup_tensors(W_float32, scale, U_k, Vh_k, final_tensor_scaled, quant_scale)

        return W_f8, dequant_scale, dequantized

    def _convert_fp8_block2d(self, W_float32: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        True 2D block-wise FP8 quantization - one scale per block_size x block_size tile.

        Scale shape: (M // block_size, N // block_size)
        Similar to INT8 block-wise scaling, optimized for tiled GEMM inference.
        """
        M, N = W_float32.shape
        bs = self.block_size

        # Validate dimensions
        if M % bs != 0 or N % bs != 0:
            info(f"    - WARNING: Dimensions ({M}, {N}) not divisible by block_size={bs}. Falling back to row-wise.")
            return self._convert_fp8_rowwise(W_float32)

        info(f"    - Using 2D block-wise FP8 scaling with block size {bs}.")

        # Reshape to 2D blocks
        W_blocked = W_float32.reshape(M // bs, bs, N // bs, bs).permute(0, 2, 1, 3)  # (M//bs, N//bs, bs, bs)
        block_max = W_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
        quant_scale = self.f8_max_val / block_max.clamp_min_(1e-12)  # (M//bs, N//bs)

        if self.no_learned_rounding:
            info("\n    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                # Apply scale per-block
                scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(-1)  # (M//bs, N//bs, 1, 1)
                W_scaled_blocked = W_blocked * scale_broadcast
                W_f8_blocked = W_scaled_blocked.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                W_f8 = W_f8_blocked.permute(0, 2, 1, 3).reshape(M, N)

                # Dequant scale is reciprocal
                dequant_scale = 1.0 / quant_scale  # (M//bs, N//bs)

                # Dequantize for bias correction
                dequant_broadcast = dequant_scale.unsqueeze(-1).unsqueeze(-1)
                dequantized_blocked = W_f8_blocked.to(COMPUTE_DTYPE) * dequant_broadcast
                dequantized = dequantized_blocked.permute(0, 2, 1, 3).reshape(M, N)

            del W_float32, W_blocked
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return (W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized)

        # With learned rounding - expand scale to full tensor for optimization
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(-1)  # (M//bs, N//bs, 1, 1)
        scale_full_blocked = scale_broadcast.expand(-1, -1, bs, bs)
        scale_full = scale_full_blocked.permute(0, 2, 1, 3).reshape(M, N)

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Use the optimizer with the expanded scale
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "prodigy":
            final_tensor_scaled = self._optimize_prodigy(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale_full, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            dequant_scale = 1.0 / quant_scale  # (M//bs, N//bs)
            dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / scale_full

        self._cleanup_tensors(W_float32, W_blocked, scale_full, scale_broadcast, U_k, Vh_k, final_tensor_scaled, quant_scale)

        return W_f8, dequant_scale, dequantized
