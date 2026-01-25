"""
Learned rounding converter for FP8 and INT8 quantization.

This module implements advanced quantization using learned adaptive rounding
with SVD-based optimization. Inherits from BaseLearnedConverter.
"""
import gc
import math
import torch
from typing import Tuple
from tqdm import tqdm
from torch.optim import AdamW, RAdam

from ..constants import (
    TARGET_FP8_DTYPE,
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MAX,
    INT8_SYMMETRIC_MAX,
)
from ..comfy.quant_ops import BlockWiseINT8Layout
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import info, verbose, debug, minimal
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
        lr: float = 8.077300000003e-3,
        **kwargs,
    ):
        """
        Initialize FP8/INT8 converter.

        Args:
            scaling_mode: Scale granularity ("tensor", "row", "block")
            block_size: Block size for block-wise scaling (default 64)
            target_format: Target format ("fp8" or "int8")
            **kwargs: All other args passed to BaseLearnedConverter
        """
        super().__init__(lr=lr, **kwargs)

        self.block_size = block_size
        self.target_format = target_format

        # INT8 defaults to block-wise scaling, but allows tensor-wise
        if target_format == "int8" and scaling_mode not in ("tensor", "block"):
            scaling_mode = "block"
        # Normalize block3d alias to block
        if scaling_mode == "block3d":
            scaling_mode = "block"
        self.scaling_mode = scaling_mode

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
            f"  - Using optimizer: '{self.optimizer_choice}'"
            + (" (disabled - simple quant)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            verbose(f"  - LR schedule: {self.lr_schedule}")
        verbose(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode in ("block", "block2d", "block3d"):
            verbose(f"    - Block size: {self.block_size}")

    def _optimize_adamw(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
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
        effective_patience, effective_factor, effective_cooldown = (
            self._compute_shape_aware_plateau_params(
                W_float32.shape[0], W_float32.shape[1]
            )
        )

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (AdamW-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
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
                worse_loss_counter = 0
                plateau_counter = 0
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
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
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
                        "plateau": f"{plateau_counter}/{effective_patience}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}",
                    }
                )

            # Early stopping conditions
            if (
                best_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
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
        return W_rounded + best_delta

    def _optimize_radam(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
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
        effective_patience, effective_factor, effective_cooldown = (
            self._compute_shape_aware_plateau_params(
                W_float32.shape[0], W_float32.shape[1]
            )
        )

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (RAdam-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
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
                worse_loss_counter = 0
                plateau_counter = 0
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
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
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
                        "plateau": f"{plateau_counter}/{effective_patience}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss_val:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}",
                    }
                )

            # Early stopping conditions
            if (
                best_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
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
        return W_rounded + best_delta

    def _optimize_original(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
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

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (Original-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
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
                if self.lr_adaptive_mode == "simple-reset":
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
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
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
                        "plateau": f"{plateau_counter}/{effective_patience}",
                    }
                )
            else:
                pbar.set_postfix(
                    {
                        "loss": f"{current_loss:.3e}",
                        "best": f"{best_loss:.3e}",
                        "lr": f"{curr_lr:.2e}",
                        "worse_count": f"{worse_loss_counter}",
                    }
                )

            # Early stopping conditions (configurable thresholds)
            if (
                current_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if (
                    curr_lr <= self.early_stop_lr * 1.75
                    and worse_loss_counter > self.early_stop_stall * 0.95
                ):
                    info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif (
                    current_loss <= self.early_stop_loss
                    and curr_lr <= self.early_stop_lr * 1.75
                ):
                    info("\n      - Learning Rate has bottomed out and loss is negligible. Stopping.")
                elif (
                    worse_loss_counter > self.early_stop_stall * 0.95
                    and current_loss > self.early_stop_loss * 2
                ):
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
        self, W_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                    dequant_scale = torch.ones(
                        num_blocks_m,
                        num_blocks_n,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                elif self.scaling_mode == "row":
                    # Row-wise: one scale per row
                    dequant_scale = torch.ones(
                        out_features, device=self.device, dtype=SCALE_DTYPE
                    )
                elif (
                    self.scaling_mode in ("block", "block2d")
                    and out_features % self.block_size == 0
                    and in_features % self.block_size == 0
                ):
                    # 2D block-wise: (M//bs, N//bs) - 'block' is primary, 'block2d' deprecated alias
                    num_blocks_m = out_features // self.block_size
                    num_blocks_n = in_features // self.block_size
                    dequant_scale = torch.ones(
                        num_blocks_m,
                        num_blocks_n,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                elif (
                    self.scaling_mode == "block3d"
                    and in_features > 0
                    and in_features % self.block_size == 0
                ):
                    # Per-row-group 3D: (out_features, num_blocks, 1)
                    num_blocks = in_features // self.block_size
                    dequant_scale = torch.ones(
                        out_features,
                        num_blocks,
                        1,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                else:
                    # Tensor-wise: single scale
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)

            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32)

        # INT8 quantization path
        if self.target_format == "int8":
            if self.scaling_mode == "tensor":
                return self._convert_int8_tensorwise(W_float32)
            else:
                return self._convert_int8(W_float32)

        # FP8 quantization path - route based on scaling_mode
        if self.scaling_mode == "row":
            return self._convert_fp8_rowwise(W_float32)
        elif self.scaling_mode in ("block", "block2d"):
            # 2D block-wise - 'block' is primary, 'block2d' is deprecated alias
            return self._convert_fp8_block2d(W_float32)
        elif self.scaling_mode == "block3d":
            # 3D per-row-group mode (legacy)
            return self._convert_fp8(W_float32)
        else:
            # 'tensor' mode
            return self._convert_fp8(W_float32)

    def _convert_int8(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
                f"INT8 block-wise quantization requires dimensions divisible by block_size={self.block_size}. "
                f"Got shape ({M}, {N}). Consider using --skip_inefficient_layers or a different block_size."
            )

        # Select quantization backend
        # Use BlockWiseINT8Layout (blockwise backend from quant_ops.py)
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            W_float32, block_size=self.block_size, is_weight=True
        )
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

        return (
            qdata,
            scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized_weight,
        )

    def _convert_int8_tensorwise(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        INT8 tensor-wise quantization.

        Uses global scale: scale = 127 / max(abs(W))
        """
        from ..comfy.quant_ops import TensorWiseINT8Layout

        # Initial quantization
        qdata, layout_params = TensorWiseINT8Layout.quantize(
            W_float32, is_weight=True
        )
        scale = layout_params["scale"]  # Global scale (scalar)

        # Optional: Apply learned rounding optimization for INT8
        if not self.no_learned_rounding and self.num_iter > 0:
            verbose("    - Applying learned rounding optimization for INT8 (tensor-wise)...")
            qdata, scale = self._optimize_int8_tensorwise_learned_rounding(W_float32, qdata, scale)

        # Re-create layout params with potentially updated scale
        layout_params["scale"] = scale

        # Dequantize for bias correction
        dequantized_weight = TensorWiseINT8Layout.dequantize(
            qdata, scale, orig_dtype=COMPUTE_DTYPE
        )

        # Clean up
        del W_float32
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (
            qdata,
            scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized_weight,
        )

    def _optimize_int8_tensorwise_learned_rounding(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            final_tensor_scaled = self._optimize_original(
                W_float32, scale_fp8_style, U_k, Vh_k
            )
        elif self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_fp8_style, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale_fp8_style, U_k, Vh_k)
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
        q_blocked = qdata.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        # Permute to: (M//bs, N//bs, bs, bs)
        q_blocked = q_blocked.permute(0, 2, 1, 3)
        # Broadcast scale: (M//bs, N//bs, 1, 1)
        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        # Apply scale
        dequantized = q_blocked * scale_broadcast
        # Permute back and reshape: (M, N)
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    def _optimize_int8_learned_rounding(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned rounding optimization for INT8 quantization.
        Uses SVD-based optimization similar to FP8 but adapted for INT8.
        """
        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Route to appropriate optimizer
        if self.optimizer_choice == "original":
            final_qdata = self._optimize_int8_original(
                W_float32, qdata, scale, U_k, Vh_k
            )
        elif self.optimizer_choice == "adamw":
            final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        self._cleanup_tensors(U_k, Vh_k)

        return final_qdata, scale

    def _optimize_int8_adamw(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
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

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT8 (AdamW-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(
                q_refined, scale, M, N, block_size
            )

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
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
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

        final_qdata = (
            (qdata_float + best_delta)
            .clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_radam(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
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

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT8 (RAdam-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(
                q_refined, scale, M, N, block_size
            )

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
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - cosine-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
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

        final_qdata = (
            (qdata_float + best_delta)
            .clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_original(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 optimization using original gradient-based optimizer (no autograd)."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        q_refined = qdata_float.clone()

        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        plateau_counter = 0  # For plateau schedule
        cooldown_counter = 0  # For plateau cooldown
        curr_lr = self.lr
        # Dimension-aware small_mult for adaptive LR schedule
        if M == N:
            small_mult = math.gamma((M ** (1/3) / M) + 1)
        elif M > N:
            small_mult = math.pow(100, M / math.pow(N, 2))
        else:  # M < N
            small_mult = math.pow(10, N / math.pow(M, 2))

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
            range(self.num_iter),
            desc=f"    Optimizing INT8 (Original-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            with torch.no_grad():
                current_dq = self._int8_dequantize_blockwise(
                    q_refined, scale, M, N, block_size
                )
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
                if self.lr_adaptive_mode == "simple-reset":
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
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
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
                    "worse_count": f"{worse_loss_counter}",
                }
            )

            # Early stopping conditions (configurable thresholds)
            if (
                current_loss <= self.early_stop_loss
                or curr_lr <= self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if (
                    curr_lr <= self.early_stop_lr * 1.75
                    and worse_loss_counter > self.early_stop_stall * 0.95
                ):
                    info("\n      - Loss has stalled and learning rate has bottomed out. Stopping.")
                elif (
                    current_loss <= self.early_stop_loss
                    and curr_lr <= self.early_stop_lr * 1.75
                ):
                    info(
                        "      - Learning Rate has bottomed out and loss is negligible. Stopping."
                    )
                elif (
                    worse_loss_counter > self.early_stop_stall * 0.95
                    and current_loss > self.early_stop_loss * 2
                ):
                    info("\n      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss <= self.early_stop_loss:
                    info("\n      - Loss is negligible. Stopping.")
                elif curr_lr <= self.early_stop_lr:
                    info("\n      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    info("\n      - Loss has stalled. Stopping.")
                break

            with torch.no_grad():
                # Compute gradient direction in INT8 quantized space
                #
                # Math derivation:
                # - Dequantization: dq = Q * scale (per-block)
                # - Loss L is computed on dq
                # - By chain rule: ∂L/∂Q = ∂L/∂dq * ∂dq/∂Q = ∂L/∂dq * scale
                #
                # So we need to MULTIPLY the weight-space gradient by scale to get Q-space gradient
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k

                # Transform gradient through block-wise structure
                # Reshape grad to blocks, multiply by scale (chain rule), then reshape back
                grad_blocked = grad_direction.reshape(
                    M // block_size, block_size, N // block_size, block_size
                )
                grad_blocked = grad_blocked.permute(0, 2, 1, 3)
                scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
                grad_scaled = grad_blocked * scale_broadcast
                grad_scaled = grad_scaled.permute(0, 2, 1, 3).reshape(M, N)

                q_refined -= curr_lr * grad_scaled

        pbar.close()

        final_tensor = best_tensor if best_tensor is not None else q_refined
        final_qdata = (
            final_tensor.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, q_refined
        return final_qdata

    def _convert_fp8(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original FP8 quantization path."""

        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == "block":
            if (
                W_float32.ndim == 2
                and W_float32.shape[1] > 0
                and W_float32.shape[1] % self.block_size == 0
            ):
                verbose(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(
                    out_features, in_features
                )
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

        assert (
            scale is not None
        ), "scale should not be None after scaling mode selection"

        # Skip SVD optimization if no_learned_rounding is set
        if self.no_learned_rounding:
            verbose("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_f8 = (
                    (W_float32 * scale)
                    .clamp(-self.f8_max_val, self.f8_max_val)
                    .to(TARGET_FP8_DTYPE)
                )
                if compact_scale is None:
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
                else:
                    if current_scaling_mode == "block":
                        dequant_scale = compact_scale.reciprocal()
                    else:
                        dequant_scale = compact_scale.reciprocal()
                    dequant_scale = dequant_scale.to(
                        device=self.device, dtype=SCALE_DTYPE
                    )
                dequantized_weight_tensor = (
                    W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
                )
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
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            if compact_scale is None:
                verbose(
                    "    - WARNING: compact_scale is None, falling back to torch.ones for dequant_scale."
                )
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                if current_scaling_mode == "block":
                    dequant_scale = compact_scale.reciprocal()
                else:
                    dequant_scale = compact_scale.reciprocal()
                dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized_weight_tensor = (
                W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
            )
        del W_float32, scale, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (
            W_f8,
            dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized_weight_tensor,
        )

    def _convert_fp8_rowwise(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

            return (
                W_f8,
                dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
                dequantized,
            )

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Use the appropriate optimizer with row-wise scale
        scale = quant_scale  # (M, 1) for broadcast
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
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

    def _convert_fp8_block2d(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        True 2D block-wise FP8 quantization - one scale per block_size x block_size tile.

        Scale shape: (M // block_size, N // block_size)
        Similar to INT8 block-wise scaling, optimized for tiled GEMM inference.
        """
        M, N = W_float32.shape
        bs = self.block_size

        # Validate dimensions
        if M % bs != 0 or N % bs != 0:
            info(
                f"    - WARNING: Dimensions ({M}, {N}) not divisible by block_size={bs}. Falling back to row-wise."
            )
            return self._convert_fp8_rowwise(W_float32)

        info(f"    - Using 2D block-wise FP8 scaling with block size {bs}.")

        # Reshape to 2D blocks
        W_blocked = W_float32.reshape(M // bs, bs, N // bs, bs).permute(
            0, 2, 1, 3
        )  # (M//bs, N//bs, bs, bs)
        block_max = W_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
        quant_scale = self.f8_max_val / block_max.clamp_min_(1e-12)  # (M//bs, N//bs)

        if self.no_learned_rounding:
            info("\n    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                # Apply scale per-block
                scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(
                    -1
                )  # (M//bs, N//bs, 1, 1)
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

            return (
                W_f8,
                dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
                dequantized,
            )

        # With learned rounding - expand scale to full tensor for optimization
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(
            -1
        )  # (M//bs, N//bs, 1, 1)
        scale_full_blocked = scale_broadcast.expand(-1, -1, bs, bs)
        scale_full = scale_full_blocked.permute(0, 2, 1, 3).reshape(M, N)

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Use the optimizer with the expanded scale
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(
                W_float32, scale_full, U_k, Vh_k
            )
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
            dequant_scale = 1.0 / quant_scale  # (M//bs, N//bs)
            dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / scale_full

        self._cleanup_tensors(
            W_float32, W_blocked, scale_full, scale_broadcast,
            U_k, Vh_k, final_tensor_scaled, quant_scale
        )

        return W_f8, dequant_scale, dequantized
