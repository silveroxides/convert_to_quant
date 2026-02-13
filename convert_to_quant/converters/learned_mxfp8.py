"""
Learned Rounding MXFP8 (Microscaling FP8) Quantization Converter.

Implements MXFP8 block quantization with SVD-based learned rounding
optimization. Inherits from BaseLearnedConverter for shared infrastructure.

Uses comfy-kitchen CUDA/Triton kernels when available, with PyTorch fallback.

Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.
"""
import gc
import math
from typing import Tuple, Optional, Dict

import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import (
    MXFP8_BLOCK_SIZE,
    MXFP8_DTYPE,
    E8M0_BIAS,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
)
from ..utils.float_utils import (
    roundup,
    e8m0_to_f32,
    mxfp8_to_blocked,
    mxfp8_from_blocked,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import verbose, debug, minimal, info
from .base_converter import BaseLearnedConverter

# Check for comfy-kitchen availability
try:
    import comfy_kitchen as ck
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False

# Track if fallback warning has been shown
_FALLBACK_WARNING_SHOWN = False


class LearnedMXFP8Converter(BaseLearnedConverter):
    """
    Learned Rounding MXFP8 block quantization converter.

    Inherits shared infrastructure from BaseLearnedConverter.
    Adds MXFP8-specific: block_size=32 validation, pad_to_32x padding.
    """

    def __init__(
        self,
        block_size: int = 32,
        pad_to_32x: bool = True,
        scale_refinement_rounds: int = 1,
        scale_optimization: str = "fixed",
        lr: float = 8.077300000003e-3,
        extract_lora: bool = False,
        lora_rank: int = 32,
        lora_depth: int = 1,
        lora_target: Optional[str] = None,
        lora_ar_threshold: float = 0.0,
        **kwargs,
    ):
        """
        Initialize MXFP8 converter.

        Args:
            block_size: Block size for quantization (must be 32 for MXFP8)
            pad_to_32x: Pad dimensions to be divisible by 32
            scale_refinement_rounds: Number of scale refinement rounds (for iterative mode)
            scale_optimization: Scale optimization mode:
                - "fixed": Scales computed once from original weights (default)
                - "iterative": Scales recomputed periodically during optimization
                - "joint": STE-based joint optimization of weights and scales
            **kwargs: All other args passed to BaseLearnedConverter
        """
        if block_size != 32:
            raise ValueError("MXFP8 requires block_size=32")

        valid_scale_modes = ("fixed", "iterative", "joint")
        if scale_optimization not in valid_scale_modes:
            raise ValueError(f"scale_optimization must be one of {valid_scale_modes}, got '{scale_optimization}'")

        super().__init__(
            lr=lr,
            extract_lora=extract_lora,
            lora_rank=lora_rank,
            lora_depth=lora_depth,
            lora_target=lora_target,
            lora_ar_threshold=lora_ar_threshold,
            **kwargs,
        )

        self.block_size = block_size
        self.pad_to_32x = pad_to_32x
        self.scale_optimization = scale_optimization
        self.scale_refinement_rounds = max(1, scale_refinement_rounds) if scale_optimization == "iterative" else 1
        self.fp8_max = torch.finfo(MXFP8_DTYPE).max

        verbose(f"LearnedMXFP8Converter initialized on device: {self.device}")
        verbose(f"  - Format: MXFP8 (Microscaling FP8)")
        verbose(f"  - Block size: {self.block_size}")
        verbose(f"  - Scale optimization: {self.scale_optimization}")
        if self.scale_optimization == "iterative" and self.scale_refinement_rounds > 1:
            verbose(f"  - Scale refinement rounds: {self.scale_refinement_rounds}")
        verbose(
            f"  - Using optimizer: '{self.optimizer_choice}'"
            + (" (disabled - simple quant)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            verbose(f"  - LR schedule: {self.lr_schedule}")

    def convert(
        self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Convert tensor to MXFP8 format with learned rounding optimization.

        Args:
            W_orig: Input tensor (2D)

        Returns:
            Tuple of (qdata_fp8, block_scales_e8m0, dequantized_weight)
        """
        global _FALLBACK_WARNING_SHOWN

        # Transfer to GPU with pinned memory for large tensors
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)
        W_float32_for_lora = W_float32.clone() if self.extract_lora else None

        # Determine if we should optimize
        if torch.all(W_float32 == 0):
            verbose("  - Tensor is all zeros, skipping optimization.")
            return self._quantize_zeros(W_float32)

        # Handle padding
        orig_shape = W_float32.shape
        rows, cols = orig_shape

        if self.pad_to_32x:
            padded_rows = roundup(rows, 32)
            padded_cols = roundup(cols, 32)
            if padded_rows != rows or padded_cols != cols:
                W_float32 = torch.nn.functional.pad(
                    W_float32, (0, padded_cols - cols, 0, padded_rows - rows)
                )

        # Validate dimensions
        M, N = W_float32.shape
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(
                f"MXFP8 requires dimensions divisible by {self.block_size}. "
                f"Got shape ({M}, {N}). Enable --pad_to_32x or use --heur to skip."
            )

        # Compute initial block scales
        block_scales_e8m0, block_scales_f32, zero_mask = self._compute_block_scales(
            W_float32, M, N
        )

        if self.no_learned_rounding:
            # Simple quantization without optimization
            qdata = self._simple_quantize(W_float32, block_scales_f32, zero_mask)
        else:
            # Apply learned rounding optimization
            qdata, block_scales_e8m0, block_scales_f32 = self._optimize_mxfp8(
                W_float32, block_scales_f32, zero_mask, block_scales_e8m0
            )

        # Convert block scales to cuBLAS tiled layout
        blocked_scales = mxfp8_to_blocked(block_scales_e8m0, flatten=False)

        # Dequantize for bias correction / output
        dequantized = self._dequantize(qdata, block_scales_f32, W_float32.shape)

        # Cleanup
        del W_float32
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        # Error Correction LoRA extraction
        extra_tensors = {}
        if self._should_extract_lora(key, orig_shape, depth):
            lora_data = self._extract_error_lora(W_float32_for_lora, dequantized)
            if lora_data:
                extra_tensors.update(lora_data)

        return qdata, blocked_scales, dequantized, extra_tensors

    def _compute_block_scales(
        self, W: torch.Tensor, M: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute E8M0 block scales for MXFP8 quantization.

        Args:
            W: Weight tensor to compute scales from
            M, N: Tensor dimensions

        Returns:
            Tuple of (block_scales_e8m0, block_scales_f32, zero_mask)
        """
        num_blocks = N // self.block_size
        tensor_blocks = W.reshape(M, num_blocks, self.block_size)
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)

        # Compute scale needed to fit in FP8 range
        scale_needed = block_max.float() / self.fp8_max
        scale_needed = torch.clamp(scale_needed, min=2**(-127))

        # Convert to E8M0 exponent (round up to ensure values fit)
        log2_scale = torch.log2(scale_needed)
        exp_biased = torch.ceil(log2_scale).to(torch.int32) + E8M0_BIAS
        exp_biased = torch.clamp(exp_biased, 0, 254)

        block_scales_e8m0 = exp_biased.to(torch.uint8)
        block_scales_f32 = e8m0_to_f32(block_scales_e8m0)

        # Handle zero blocks
        zero_mask = (block_max == 0)
        block_scales_f32 = torch.where(
            zero_mask,
            torch.ones_like(block_scales_f32),
            block_scales_f32
        )

        return block_scales_e8m0, block_scales_f32, zero_mask

    _compute_block_scales_from_tensor = _compute_block_scales

    def _quantize_zeros(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Handle all-zeros tensor."""
        M, N = W_float32.shape
        num_blocks = N // self.block_size

        qdata = torch.zeros((M, N), dtype=MXFP8_DTYPE, device=self.device)

        # Block scales in cuBLAS tiled layout
        block_scales_shape = (roundup(M, 128), roundup(num_blocks, 4))
        block_scales = torch.zeros(block_scales_shape, dtype=torch.uint8, device=self.device)

        dequantized = torch.zeros_like(W_float32)

        return qdata, block_scales, dequantized, {}

    def _simple_quantize(
        self, W_float32: torch.Tensor, block_scales_f32: torch.Tensor, zero_mask: torch.Tensor
    ) -> torch.Tensor:
        """Simple quantization without learned rounding."""
        M, N = W_float32.shape
        num_blocks = N // self.block_size
        tensor_blocks = W_float32.reshape(M, num_blocks, self.block_size)

        data_scaled = tensor_blocks.float() / block_scales_f32.unsqueeze(-1)
        data_scaled = torch.where(
            zero_mask.unsqueeze(-1),
            torch.zeros_like(data_scaled),
            data_scaled
        )
        data_scaled = torch.clamp(data_scaled, -self.fp8_max, self.fp8_max)
        qdata = data_scaled.reshape(M, N).to(MXFP8_DTYPE)

        return qdata

    def _dequantize(
        self, qdata: torch.Tensor, block_scales_f32: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        """Dequantize FP8 data to float."""
        M, N = orig_shape
        num_blocks = N // self.block_size

        data_f32 = qdata.float().reshape(M, num_blocks, self.block_size)
        dequantized = data_f32 * block_scales_f32.unsqueeze(-1)

        return dequantized.view(M, N).to(COMPUTE_DTYPE)

    def _optimize_mxfp8(
        self, W_float32: torch.Tensor, block_scales_f32: torch.Tensor,
        zero_mask: torch.Tensor, block_scales_e8m0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply learned rounding optimization for MXFP8.

        Returns:
            Tuple of (qdata, block_scales_e8m0, block_scales_f32)
        """
        M, N = W_float32.shape

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Route to appropriate optimizer
        if self.optimizer_choice == "original":
            qdata, block_scales_e8m0, block_scales_f32 = self._optimize_original(
                W_float32, block_scales_f32, U_k, Vh_k, block_scales_e8m0, zero_mask
            )
        elif self.optimizer_choice == "adamw":
            qdata, block_scales_e8m0, block_scales_f32 = self._optimize_adamw(
                W_float32, block_scales_f32, U_k, Vh_k, block_scales_e8m0, zero_mask
            )
        elif self.optimizer_choice == "radam":
            qdata, block_scales_e8m0, block_scales_f32 = self._optimize_radam(
                W_float32, block_scales_f32, U_k, Vh_k, block_scales_e8m0, zero_mask
            )
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        # Cleanup SVD tensors
        self._cleanup_tensors(U_k, Vh_k)

        return qdata, block_scales_e8m0, block_scales_f32

    def _mxfp8_dequantize_blockwise(
        self, qdata_float: torch.Tensor, block_scales_f32: torch.Tensor, M: int, N: int,
        discretize: bool = True
    ) -> torch.Tensor:
        """
        Differentiable block-wise MXFP8 dequantization for optimization.

        Args:
            discretize: If True, applies FP8 discretization (breaks gradients).
                        Set to False for autograd-based optimizers (RAdam, AdamW).
        """
        num_blocks = N // self.block_size

        if discretize:
            # Apply FP8 discretization to simulate quantization error
            # Only use with torch.no_grad() contexts (original optimizer)
            qdata_discrete = qdata_float.to(MXFP8_DTYPE).float()
        else:
            # Keep gradient flow for autograd optimizers
            qdata_discrete = qdata_float

        data_blocks = qdata_discrete.reshape(M, num_blocks, self.block_size)
        dequantized = data_blocks * block_scales_f32.unsqueeze(-1)
        return dequantized.view(M, N)

    def _optimize_original(
        self,
        W_float32: torch.Tensor,
        block_scales_f32: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        block_scales_e8m0: torch.Tensor,
        zero_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MXFP8 optimization using original gradient descent with tier-based LR."""
        M, N = W_float32.shape
        num_blocks = N // self.block_size

        # Start with initial scaled values
        tensor_blocks = W_float32.reshape(M, num_blocks, self.block_size)
        W_q_initial = tensor_blocks / block_scales_f32.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -self.fp8_max, self.fp8_max)
        # Snap to FP8 grid to ensure non-zero initial loss
        W_q_initial = W_q_initial.to(MXFP8_DTYPE).float()
        W_q_refined = W_q_initial.view(M, N).clone()

        current_block_scales_f32 = block_scales_f32.clone()
        current_block_scales_e8m0 = block_scales_e8m0.clone()

        best_loss = float("inf")
        best_qdata = None
        best_block_scales_f32 = block_scales_f32.clone()
        best_block_scales_e8m0 = block_scales_e8m0.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        curr_lr = self.lr


        schedule_name = self.lr_schedule

        # Shape-aware plateau parameters (matching learned_rounding.py)
        aspect_ratio = max(M, N) / min(M, N)

        if schedule_name == "plateau" and self.lr_shape_influence > 0:
            # Scale factor based on aspect ratio, modulated by influence
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence

            effective_patience = self.lr_patience
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing MXFP8 (Original-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            with torch.no_grad():
                current_dq = self._mxfp8_dequantize_blockwise(
                    W_q_refined, current_block_scales_f32, M, N, discretize=False
                )
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()

            # Check improvement (matching learned_rounding.py logic)
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                     improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:
                     improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss = current_loss
                best_qdata = W_q_refined.clone()
                best_block_scales_f32 = current_block_scales_f32.clone()
                best_block_scales_e8m0 = current_block_scales_e8m0.clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
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
                    if plateau_counter > 0:
                         debug(f"      [LR] Waiting: {plateau_counter}/{effective_patience} (Loss: {current_loss:.3e})")
            else:  # 'adaptive'
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

            # Postfix
            if schedule_name == "plateau":
                pbar.set_postfix({
                    "loss": f"{current_loss:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{effective_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{current_loss:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse": f"{worse_loss_counter}",
                })

            # Early stopping conditions (explicit match to learned_rounding.py)
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

            # Gradient step with proper scaling
            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k

                # Apply 1/scale factor to gradient (scale = block_scales_f32)
                # Reshape grad to blocks, divide by scale, reshape back
                grad_blocked = grad_direction.reshape(M, num_blocks, self.block_size)
                grad_scaled = grad_blocked / current_block_scales_f32.unsqueeze(-1)
                grad_scaled = grad_scaled.reshape(M, N)

                W_q_refined -= curr_lr * grad_scaled
                # Note: No clamp inside the loop, consistent with learned_rounding.py

        pbar.close()

        # Convert best values to FP8
        best_qdata = best_qdata if best_qdata is not None else W_q_refined
        qdata = best_qdata.to(MXFP8_DTYPE)

        return qdata, best_block_scales_e8m0, best_block_scales_f32

    def _optimize_adamw(
        self,
        W_float32: torch.Tensor,
        block_scales_f32: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        block_scales_e8m0: torch.Tensor,
        zero_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MXFP8 optimization using AdamW optimizer."""
        M, N = W_float32.shape
        num_blocks = N // self.block_size

        tensor_blocks = W_float32.reshape(M, num_blocks, self.block_size)
        W_q_initial = tensor_blocks / block_scales_f32.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -self.fp8_max, self.fp8_max)

        # Initialize with rounded values to ensure non-zero starting loss
        W_q_initial = W_q_initial.to(MXFP8_DTYPE).float()

        qdata_f32 = W_q_initial.view(M, N)

        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.lr
        optimizer = AdamW([delta], lr=curr_lr)

        current_block_scales_f32 = block_scales_f32.clone()
        current_block_scales_e8m0 = block_scales_e8m0.clone()

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        best_block_scales_f32 = block_scales_f32.clone()
        best_block_scales_e8m0 = block_scales_e8m0.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Shape-aware plateau parameters
        effective_patience, effective_factor, effective_cooldown = (
            self._compute_shape_aware_plateau_params(M, N)
        )

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing MXFP8 (AdamW-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_f32 + delta
            # Use discretize=False to keep gradient flow for autograd optimizer
            current_dq = self._mxfp8_dequantize_blockwise(q_refined, current_block_scales_f32, M, N, discretize=False)

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
                best_block_scales_f32 = current_block_scales_f32.clone()
                best_block_scales_e8m0 = current_block_scales_e8m0.clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR schedule update
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
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
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Postfix
            if schedule_name == "plateau":
                pbar.set_postfix({
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{effective_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                })

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

        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -self.fp8_max, self.fp8_max)
        qdata = final_q.to(MXFP8_DTYPE)

        return qdata, best_block_scales_e8m0, best_block_scales_f32

    def _optimize_radam(
        self,
        W_float32: torch.Tensor,
        block_scales_f32: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        block_scales_e8m0: torch.Tensor,
        zero_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """MXFP8 optimization using RAdam optimizer."""
        M, N = W_float32.shape
        num_blocks = N // self.block_size

        tensor_blocks = W_float32.reshape(M, num_blocks, self.block_size)
        W_q_initial = tensor_blocks / block_scales_f32.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -self.fp8_max, self.fp8_max)

        # Initialize with rounded values to ensure non-zero starting loss
        W_q_initial = W_q_initial.to(MXFP8_DTYPE).float()

        qdata_f32 = W_q_initial.view(M, N)

        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.lr
        optimizer = RAdam([delta], lr=curr_lr)

        current_block_scales_f32 = block_scales_f32.clone()
        current_block_scales_e8m0 = block_scales_e8m0.clone()

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        best_block_scales_f32 = block_scales_f32.clone()
        best_block_scales_e8m0 = block_scales_e8m0.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        # Shape-aware plateau parameters
        effective_patience, effective_factor, effective_cooldown = (
            self._compute_shape_aware_plateau_params(M, N)
        )

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing MXFP8 (RAdam-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_f32 + delta
            # Use discretize=False to keep gradient flow for autograd optimizer
            current_dq = self._mxfp8_dequantize_blockwise(q_refined, current_block_scales_f32, M, N, discretize=False)

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
                best_block_scales_f32 = current_block_scales_f32.clone()
                best_block_scales_e8m0 = current_block_scales_e8m0.clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR schedule update
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                    debug(f"      [LR] Cooldown: {cooldown_counter} left")
                elif plateau_counter >= effective_patience:
                    debug(f"      [LR] Plateau {plateau_counter}/{effective_patience} reached. Decaying.")
                    if curr_lr > self.lr_min:
                        old_lr = curr_lr
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
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
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            # Postfix
            if schedule_name == "plateau":
                pbar.set_postfix({
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "plateau": f"{plateau_counter}/{effective_patience}",
                })
            else:
                pbar.set_postfix({
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                })

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

        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -self.fp8_max, self.fp8_max)
        qdata = final_q.to(MXFP8_DTYPE)

        return qdata, best_block_scales_e8m0, best_block_scales_f32

    def _check_early_stop(
        self, current_loss: float, curr_lr: float, worse_loss_counter: int
    ) -> bool:
        """Check early stopping conditions."""
        if self.early_stop_loss > 0 and current_loss <= self.early_stop_loss:
            info("\n      - Loss is negligible. Stopping early.")
            return True
        if self.early_stop_lr > 0 and curr_lr <= self.early_stop_lr:
            info("\n      - Learning rate bottomed out. Stopping early.")
            return True
        if self.early_stop_stall > 0 and worse_loss_counter >= self.early_stop_stall:
            info("\n      - Stalled. Stopping early.")
            return True
        return False
