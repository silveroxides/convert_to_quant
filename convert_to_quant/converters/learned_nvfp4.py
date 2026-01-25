"""
Learned Rounding NVFP4 (E2M1) Quantization Converter.

Implements NVIDIA FP4 E2M1 block quantization with SVD-based learned rounding
optimization. Inherits from BaseLearnedConverter for shared infrastructure.

Uses comfy-kitchen CUDA/Triton kernels when available, with PyTorch fallback.

Requires SM >= 10.0 (datacenter Blackwell) or SM >= 12.0 (consumer RTX 50 series).
"""
import gc
import math
from typing import Tuple, Optional

import torch
from torch.optim import AdamW, RAdam
from tqdm import tqdm

from ..constants import (
    FP4_E2M1_MAX,
    FP4_BLOCK_SIZE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
)
from ..utils.float_utils import (
    F8_E4M3_MAX,
    roundup,
    pack_uint4,
    unpack_uint4,
    to_blocked,
    from_blocked,
    _f32_to_floatx_unpacked,
    _floatx_unpacked_to_f32,
    _float8_round,
    F4_E2M1_EBITS,
    F4_E2M1_MBITS,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.logging import info, verbose, debug, minimal
from .base_converter import BaseLearnedConverter

# Check for comfy-kitchen availability
try:
    import comfy_kitchen as ck
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False

class LearnedNVFP4Converter(BaseLearnedConverter):
    """
    Learned Rounding NVFP4 (E2M1) block quantization converter.

    Inherits shared infrastructure from BaseLearnedConverter.
    Adds NVFP4-specific: block_size=16 validation, pad_to_16x padding.
    """

    def __init__(
        self,
        block_size: int = 16,
        pad_to_16x: bool = True,
        scale_refinement_rounds: int = 1,
        scale_optimization: str = "fixed",
        lr: float = 8.077300000003e-3,
        **kwargs,
    ):
        """
        Initialize NVFP4 converter.

        Args:
            block_size: Block size for quantization (must be 16 for NVFP4)
            pad_to_16x: Pad dimensions to be divisible by 16
            scale_refinement_rounds: Number of scale refinement rounds (for iterative mode)
            scale_optimization: Scale optimization mode:
                - "fixed": Scales computed once from original weights (default)
                - "iterative": Scales recomputed periodically during optimization
                - "joint": STE-based joint optimization of weights and scales
            **kwargs: All other args passed to BaseLearnedConverter
        """
        if block_size != 16:
            raise ValueError("NVFP4 requires block_size=16")

        valid_scale_modes = ("fixed", "iterative", "joint")
        if scale_optimization not in valid_scale_modes:
            raise ValueError(f"scale_optimization must be one of {valid_scale_modes}, got '{scale_optimization}'")

        super().__init__(lr=lr, **kwargs)

        self.block_size = block_size
        self.pad_to_16x = pad_to_16x
        self.scale_optimization = scale_optimization
        # For iterative mode, use provided rounds; for others, ignore
        self.scale_refinement_rounds = max(1, scale_refinement_rounds) if scale_optimization == "iterative" else 1

        verbose(f"LearnedNVFP4Converter initialized on device: {self.device}")
        verbose(f"  - Format: NVFP4 (FP4 E2M1)")
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
        self, W_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert tensor to NVFP4 format with learned rounding optimization.

        Args:
            W_orig: Input tensor (2D)

        Returns:
            Tuple of (packed_qdata, block_scales, per_tensor_scale, dequantized_weight)
        """
        # Transfer to GPU with pinned memory for large tensors
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        # Determine if we should optimize
        if torch.all(W_float32 == 0):
            verbose("  - Tensor is all zeros, skipping optimization.")
            return self._quantize_zeros(W_float32)

        # Handle padding
        orig_shape = W_float32.shape
        rows, cols = orig_shape

        if self.pad_to_16x:
            padded_rows = roundup(rows, 16)
            padded_cols = roundup(cols, 16)
            if padded_rows != rows or padded_cols != cols:
                W_float32 = torch.nn.functional.pad(
                    W_float32, (0, padded_cols - cols, 0, padded_rows - rows)
                )

        # Validate dimensions
        M, N = W_float32.shape
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(
                f"NVFP4 requires dimensions divisible by {self.block_size}. "
                f"Got shape ({M}, {N}). Enable --pad_to_16x or use --heur to skip."
            )

        # Compute per-tensor scale (fixed)
        amax = torch.amax(torch.abs(W_float32))
        per_tensor_scale = (amax / (F8_E4M3_MAX * FP4_E2M1_MAX)).to(dtype=torch.float32)

        # Compute initial block scales
        scaled_block_scales_fp8, total_scale, zero_scale_mask = self._compute_block_scales(
            W_float32, per_tensor_scale, M, N
        )
        total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

        if self.no_learned_rounding:
            # Simple quantization without optimization
            qdata = self._simple_quantize(W_float32, total_scale_safe, zero_scale_mask)
            final_total_scale = total_scale
        else:
            # Apply learned rounding optimization (may update scales iteratively)
            qdata, final_total_scale = self._optimize_nvfp4(
                W_float32, total_scale_safe, zero_scale_mask, per_tensor_scale
            )

        # Compute final block scales for storage (from final_total_scale)
        # Note: If scales were updated during optimization, we need to recompute the FP8 representation
        if self.scale_refinement_rounds > 1 and not self.no_learned_rounding:
            # Extract block scales from final_total_scale
            scaled_block_scales_fp8 = (final_total_scale / per_tensor_scale)
            scaled_block_scales_fp8 = torch.clamp(scaled_block_scales_fp8, max=F8_E4M3_MAX)

        # Pack to uint8
        data_packed = pack_uint4(qdata)

        # Convert block scales to cuBLAS tiled layout
        blocked_scales = to_blocked(
            scaled_block_scales_fp8.to(torch.float8_e4m3fn),
            flatten=False
        )

        # Dequantize for bias correction / output
        dequantized = self._dequantize(qdata, final_total_scale, W_float32.shape)

        # Cleanup
        del W_float32
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return data_packed, blocked_scales, per_tensor_scale, dequantized

    def _compute_block_scales(
        self, W: torch.Tensor, per_tensor_scale: torch.Tensor, M: int, N: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute block scales for NVFP4 quantization.

        Args:
            W: Weight tensor to compute scales from
            per_tensor_scale: Global per-tensor scale
            M, N: Tensor dimensions

        Returns:
            Tuple of (scaled_block_scales_fp8, total_scale, zero_scale_mask)
        """
        tensor_blocks = W.reshape(M, -1, self.block_size)
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)
        block_scale_fp32 = block_max / FP4_E2M1_MAX

        # Scale block scales relative to per-tensor scale
        scaled_block_scales = block_scale_fp32 / per_tensor_scale
        # Match comfy-kitchen: only max clamp, no min clamp
        scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
        scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)

        # Total scale for each block
        total_scale = per_tensor_scale * scaled_block_scales_fp32

        # Handle zero blocks (from padding): avoid 0/0 NaN
        zero_scale_mask = (total_scale == 0)

        return scaled_block_scales_fp8, total_scale, zero_scale_mask

    # Alias for use inside optimization loop
    _compute_block_scales_from_tensor = _compute_block_scales

    def _quantize_zeros(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Handle all-zeros tensor."""
        M, N = W_float32.shape

        # Packed output shape
        packed_shape = (M, N // 2)
        qdata = torch.zeros(packed_shape, dtype=torch.uint8, device=self.device)

        # Block scales
        num_blocks = N // self.block_size
        block_scales_shape = (roundup(M, 128), roundup(num_blocks, 4))
        block_scales = torch.zeros(block_scales_shape, dtype=torch.float8_e4m3fn, device=self.device)

        per_tensor_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
        dequantized = torch.zeros_like(W_float32)

        return qdata, block_scales, per_tensor_scale, dequantized

    def _simple_quantize(
        self, W_float32: torch.Tensor, total_scale: torch.Tensor, zero_scale_mask: torch.Tensor
    ) -> torch.Tensor:
        """Simple quantization without learned rounding (matches comfy-kitchen)."""
        M, N = W_float32.shape
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)

        data_scaled = tensor_blocks / total_scale.unsqueeze(-1)
        # Zero out blocks where scale was zero (padding)
        data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)
        data_scaled = torch.clamp(data_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        data_scaled = data_scaled.view(M, N)

        # Convert to FP4 E2M1
        qdata = _f32_to_floatx_unpacked(data_scaled.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
        return qdata

    def _dequantize(
        self, qdata: torch.Tensor, total_scale: torch.Tensor, orig_shape: tuple
    ) -> torch.Tensor:
        """Dequantize FP4 data to float."""
        M, N = orig_shape

        # Convert unpacked FP4 to float32
        data_f32 = _floatx_unpacked_to_f32(qdata, F4_E2M1_EBITS, F4_E2M1_MBITS)

        # Reshape to blocks and apply scale
        data_blocks = data_f32.reshape(M, -1, self.block_size)
        dequantized = data_blocks * total_scale.unsqueeze(-1)

        return dequantized.view(M, N).to(COMPUTE_DTYPE)

    def _optimize_nvfp4(
        self, W_float32: torch.Tensor, total_scale: torch.Tensor, zero_scale_mask: torch.Tensor,
        per_tensor_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learned rounding optimization for NVFP4.

        Returns:
            Tuple of (qdata, final_total_scale) - scales may have been updated during optimization.
        """
        M, N = W_float32.shape

        # Use inherited SVD computation
        U_k, Vh_k, k = self._compute_svd_components(W_float32)

        # Route to appropriate optimizer - all now support scale_optimization modes
        if self.optimizer_choice == "original":
            qdata, final_total_scale = self._optimize_original(W_float32, total_scale, U_k, Vh_k, per_tensor_scale)
        elif self.optimizer_choice == "adamw":
            qdata, final_total_scale = self._optimize_adamw(W_float32, total_scale, U_k, Vh_k, per_tensor_scale)
        elif self.optimizer_choice == "radam":
            qdata, final_total_scale = self._optimize_radam(W_float32, total_scale, U_k, Vh_k, per_tensor_scale)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        # Cleanup SVD tensors
        self._cleanup_tensors(U_k, Vh_k)

        return qdata, final_total_scale

    def _ste_fp8_scale(self, scale_float: torch.Tensor) -> torch.Tensor:
        """Apply Straight-Through Estimator for FP8 scale quantization.

        Forward: Returns FP8-quantized scale values
        Backward: Gradients flow through as if no quantization occurred
        """
        # Clamp to FP8 range
        scale_clamped = torch.clamp(scale_float, min=1e-12, max=F8_E4M3_MAX)
        # Quantize to FP8 and back to float32
        scale_fp8 = _float8_round(scale_clamped)
        # STE: detach the quantized version, add back the difference from continuous
        # This makes forward use quantized, backward use continuous gradients
        return scale_fp8.detach() + (scale_clamped - scale_clamped.detach())

    def _nvfp4_dequantize_blockwise(
        self, qdata_float: torch.Tensor, total_scale: torch.Tensor, M: int, N: int
    ) -> torch.Tensor:
        """
        Differentiable block-wise NVFP4 dequantization for optimization.

        Works on float representation of quantized values during optimization.
        """
        # Reshape to blocks
        data_blocks = qdata_float.reshape(M, -1, self.block_size)
        # Apply scale per block
        dequantized = data_blocks * total_scale.unsqueeze(-1)
        return dequantized.view(M, N)

    def _optimize_original(
        self,
        W_float32: torch.Tensor,
        total_scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        per_tensor_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NVFP4 optimization using original gradient descent with tier-based LR.

        Supports three scale_optimization modes:
        - "fixed": Scales computed once from original weights, never updated
        - "iterative": Scales recomputed periodically during optimization
        - "joint": STE-based joint optimization of weights AND block scales

        Works in the pre-scaled continuous float space:
        - W_q = W_float32 / scale (continuous, clamped to [-6, 6])
        - Dequant: W_dq = W_q * scale
        - Gradient descent on W_q (and optionally scales in joint mode)
        - Final: encode to FP4

        Returns:
            Tuple of (qdata, final_total_scale)
        """
        M, N = W_float32.shape

        # Start with initial scaled values (continuous, not discretized yet)
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)
        W_q_initial = tensor_blocks / total_scale.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        # Snap to FP4 grid to ensure non-zero initial loss
        q_tmp = _f32_to_floatx_unpacked(W_q_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        W_q_initial = _floatx_unpacked_to_f32(q_tmp, F4_E2M1_EBITS, F4_E2M1_MBITS)
        W_q_refined = W_q_initial.view(M, N).clone()

        # Current scale state (will be updated iteratively or jointly if enabled)
        current_total_scale = total_scale.clone()

        # For joint mode: initialize learnable block scales from initial total_scale
        if self.scale_optimization == "joint":
            # block_scales_float: continuous representation of block scales
            # total_scale = per_tensor_scale * block_scales, so:
            block_scales_float = (total_scale / per_tensor_scale).clone()
            best_block_scales = block_scales_float.clone()
        else:
            block_scales_float = None
            best_block_scales = None

        best_loss = float("inf")
        best_qdata = None
        best_total_scale = total_scale.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        curr_lr = self.lr
        scale_lr = curr_lr * 0.1  # Lower LR for scales in joint mode

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
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence
            effective_patience = self.lr_patience
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor ** ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        # Scale refinement interval (for iterative mode only)
        if self.scale_optimization == "iterative" and self.scale_refinement_rounds > 1:
            scale_update_interval = max(1, self.num_iter // self.scale_refinement_rounds)
        else:
            scale_update_interval = self.num_iter + 1  # Never update for fixed/joint

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (Original-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            # Iterative scale refinement: recompute scales periodically (iterative mode only)
            if self.scale_optimization == "iterative" and i > 0 and i % scale_update_interval == 0:
                # Dequantize current state
                current_dq = self._nvfp4_dequantize_blockwise(W_q_refined, current_total_scale, M, N)
                # Recompute scales from current dequantized weights
                new_block_scales, new_total_scale, _ = self._compute_block_scales_from_tensor(
                    current_dq, per_tensor_scale, M, N
                )
                current_total_scale = new_total_scale
                # Re-normalize W_q to new scale
                tensor_blocks_new = current_dq.reshape(M, -1, self.block_size)
                W_q_refined = (tensor_blocks_new / current_total_scale.unsqueeze(-1)).view(M, N)
                W_q_refined = torch.clamp(W_q_refined, -FP4_E2M1_MAX, FP4_E2M1_MAX)
                if self.scale_optimization == "iterative":
                    pbar.set_description(f"    Optimizing NVFP4 (scale update {i // scale_update_interval + 1})")

            # For joint mode: apply STE to block scales before dequantization
            if self.scale_optimization == "joint":
                # STE: forward uses FP8-quantized scales, backward uses continuous
                block_scales_ste = self._ste_fp8_scale(block_scales_float)
                current_total_scale = per_tensor_scale * block_scales_ste

            with torch.no_grad():
                # Dequantize: W_dq = W_q * scale (block-wise)
                current_dq = self._nvfp4_dequantize_blockwise(W_q_refined, current_total_scale, M, N)
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()

            # Threshold-based improvement check
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
                best_total_scale = current_total_scale.clone()
                if self.scale_optimization == "joint":
                    best_block_scales = block_scales_float.clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update based on schedule
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                scale_lr = curr_lr * 0.1
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= effective_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        scale_lr = curr_lr * 0.1
                        cooldown_counter = effective_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - tier-based schedule
                # Use counter before reset for boost calculation to prevent compounding
                counter_for_update = prev_worse_counter if improved else worse_loss_counter
                new_lr, lr_updated = self._adaptive_lr_update_cosine(
                    curr_lr, improved, counter_for_update, i,
                    (M, N), self.early_stop_lr
                )
                if lr_updated:
                    curr_lr = new_lr

                scale_lr = curr_lr * 0.1
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix({
                "loss": f"{current_loss:.3e}",
                "best": f"{best_loss:.3e}",
                "lr": f"{curr_lr:.2e}",
                "worse_count": f"{worse_loss_counter}",
            })

            # Early stopping
            if self._check_early_stop(current_loss, curr_lr, worse_loss_counter):
                break

            # Gradient step in pre-scaled space
            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * grad_direction
                # Clamp to FP4 range
                W_q_refined = torch.clamp(W_q_refined, -FP4_E2M1_MAX, FP4_E2M1_MAX)

                # Joint mode: also update block scales via gradient descent
                if self.scale_optimization == "joint":
                    # Compute gradient w.r.t. block scales
                    # For dequant: W_dq = W_q * (per_tensor_scale * block_scales)
                    # d(loss)/d(block_scales) ~ per_tensor_scale * sum(error * W_q) per block
                    W_q_blocks = W_q_refined.reshape(M, -1, self.block_size)
                    error_blocks = error.reshape(M, -1, self.block_size)
                    # Gradient: sum of (error * W_q) per block
                    scale_grad = torch.sum(error_blocks * W_q_blocks, dim=-1) * per_tensor_scale
                    # Normalize gradient to avoid scale explosion
                    scale_grad_norm = scale_grad / (scale_grad.abs().max().clamp_min(1e-8))
                    # Update scales (gradient descent)
                    block_scales_float = block_scales_float - scale_lr * scale_grad_norm
                    # Clamp to valid FP8 range
                    block_scales_float = torch.clamp(block_scales_float, min=1e-6, max=F8_E4M3_MAX)

        pbar.close()

        # Convert best continuous values to FP4 encoding
        best_qdata = best_qdata if best_qdata is not None else W_q_refined
        qdata = _f32_to_floatx_unpacked(best_qdata.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
        return qdata, best_total_scale

    def _optimize_adamw(
        self,
        W_float32: torch.Tensor,
        total_scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        per_tensor_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NVFP4 optimization using AdamW optimizer.

        Supports scale_optimization modes: fixed, iterative (not implemented), joint (STE).
        """
        M, N = W_float32.shape

        # Start with continuous pre-scaled values (like _optimize_original)
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)
        W_q_initial = tensor_blocks / total_scale.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        # Snap to FP4 grid to ensure non-zero initial loss
        q_tmp = _f32_to_floatx_unpacked(W_q_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        W_q_initial = _floatx_unpacked_to_f32(q_tmp, F4_E2M1_EBITS, F4_E2M1_MBITS)
        qdata_f32 = W_q_initial.view(M, N)

        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.lr

        # For joint mode: initialize learnable block scales
        current_total_scale = total_scale.clone()
        if self.scale_optimization == "joint":
            block_scales_float = (total_scale / per_tensor_scale).clone().requires_grad_(True)
            optimizer = AdamW([delta, block_scales_float], lr=curr_lr)
            best_block_scales = block_scales_float.detach().clone()
        else:
            block_scales_float = None
            optimizer = AdamW([delta], lr=curr_lr)
            best_block_scales = None

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        best_total_scale = total_scale.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (AdamW-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_f32 + delta

            # For joint mode: apply STE to get effective scales
            if self.scale_optimization == "joint":
                block_scales_ste = self._ste_fp8_scale(block_scales_float)
                current_total_scale = per_tensor_scale * block_scales_ste

            current_dq = self._nvfp4_dequantize_blockwise(q_refined, current_total_scale, M, N)

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            # Clamp block scales to valid range after optimizer step
            if self.scale_optimization == "joint":
                with torch.no_grad():
                    block_scales_float.clamp_(min=1e-6, max=F8_E4M3_MAX)

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                best_total_scale = current_total_scale.detach().clone()
                if self.scale_optimization == "joint":
                    best_block_scales = block_scales_float.detach().clone()
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
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
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
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

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

            if self._check_early_stop(best_loss, curr_lr, worse_loss_counter):
                break

        pbar.close()

        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        qdata = _f32_to_floatx_unpacked(final_q.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
        return qdata, best_total_scale

    def _optimize_radam(
        self,
        W_float32: torch.Tensor,
        total_scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
        per_tensor_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """NVFP4 optimization using RAdam optimizer.

        Supports scale_optimization modes: fixed, iterative (not implemented), joint (STE).
        """
        M, N = W_float32.shape

        # Start with continuous pre-scaled values (like _optimize_original)
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)
        W_q_initial = tensor_blocks / total_scale.unsqueeze(-1)
        W_q_initial = torch.clamp(W_q_initial, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        # Snap to FP4 grid to ensure non-zero initial loss
        q_tmp = _f32_to_floatx_unpacked(W_q_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        W_q_initial = _floatx_unpacked_to_f32(q_tmp, F4_E2M1_EBITS, F4_E2M1_MBITS)
        qdata_f32 = W_q_initial.view(M, N)

        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.lr

        # For joint mode: initialize learnable block scales
        current_total_scale = total_scale.clone()
        if self.scale_optimization == "joint":
            block_scales_float = (total_scale / per_tensor_scale).clone().requires_grad_(True)
            optimizer = RAdam([delta, block_scales_float], lr=curr_lr)
            best_block_scales = block_scales_float.detach().clone()
        else:
            block_scales_float = None
            optimizer = RAdam([delta], lr=curr_lr)
            best_block_scales = None

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        best_total_scale = total_scale.clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        mode_suffix = f"-{self.scale_optimization}" if self.scale_optimization != "fixed" else ""
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (RAdam-{schedule_name}{mode_suffix})",
            leave=False,
            dynamic_ncols=True,
        )

        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_f32 + delta

            # For joint mode: apply STE to get effective scales
            if self.scale_optimization == "joint":
                block_scales_ste = self._ste_fp8_scale(block_scales_float)
                current_total_scale = per_tensor_scale * block_scales_ste

            current_dq = self._nvfp4_dequantize_blockwise(q_refined, current_total_scale, M, N)

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            # Clamp block scales to valid range after optimizer step
            if self.scale_optimization == "joint":
                with torch.no_grad():
                    block_scales_float.clamp_(min=1e-6, max=F8_E4M3_MAX)

            current_loss_val = loss.item()
            prev_worse_counter = worse_loss_counter
            improved = self._check_improvement(current_loss_val, best_loss)

            if improved:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                best_total_scale = current_total_scale.detach().clone()
                if self.scale_optimization == "joint":
                    best_block_scales = block_scales_float.detach().clone()
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
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for pg in optimizer.param_groups:
                            pg["lr"] = curr_lr
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
                    for pg in optimizer.param_groups:
                        pg["lr"] = curr_lr

                # Reset counter after boost in no-reset adaptive mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

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

            if self._check_early_stop(best_loss, curr_lr, worse_loss_counter):
                break

        pbar.close()

        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        qdata = _f32_to_floatx_unpacked(final_q.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
        return qdata, best_total_scale



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
            info("\n      - Loss has stalled. Stopping early.")
            return True
        return False

