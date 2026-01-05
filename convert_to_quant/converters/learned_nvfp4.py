"""
Learned Rounding NVFP4 (E2M1) Quantization Converter.

Implements NVIDIA FP4 E2M1 block quantization with SVD-based learned rounding
optimization. Provides high-quality quantization using the same optimization
infrastructure as LearnedRoundingConverter for FP8/INT8.

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
    F8_E4M3_EPS,
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


class LearnedNVFP4Converter:
    """
    Learned Rounding NVFP4 (E2M1) block quantization converter.
    
    Uses SVD-based optimization to minimize quantization error in the
    principal component space, providing significantly better accuracy
    than naive rounding.
    
    Args:
        optimizer: Optimization algorithm ("original", "adamw", "radam")
        num_iter: Number of optimization iterations
        top_p: Proportion of principal components to use
        min_k: Minimum number of SVD components
        max_k: Maximum number of SVD components
        block_size: Block size for quantization (must be 16 for NVFP4)
        pad_to_16x: Pad dimensions to be divisible by 16
        full_matrix: Use full SVD instead of lowrank
        no_learned_rounding: Skip optimization, use simple quantization
        lr_schedule: LR schedule ("adaptive", "exponential", "plateau")
        lr_gamma: Decay factor for exponential schedule
        lr_patience: Steps before decay for plateau schedule
        lr_factor: LR reduction factor for plateau
        lr_min: Minimum learning rate
        lr_cooldown: Cooldown steps after LR reduction
        lr_threshold: Minimum improvement threshold
        lr_adaptive_mode: Counter reset mode ("simple-reset", "no-reset")
        lr_shape_influence: Shape-aware LR scaling (0.0-1.0)
        lr_threshold_mode: Threshold mode ("rel", "abs")
        early_stop_loss: Stop when loss drops below this
        early_stop_lr: Stop when LR drops below this
        early_stop_stall: Stop after this many steps without improvement
    """
    
    def __init__(
        self,
        optimizer: str = "original",
        num_iter: int = 500,
        top_p: float = 0.01,
        min_k: int = 1,
        max_k: int = 16,
        block_size: int = 16,
        pad_to_16x: bool = True,
        full_matrix: bool = False,
        no_learned_rounding: bool = False,
        lr_schedule: str = "adaptive",
        lr_gamma: float = 0.99,
        lr_patience: int = 50,
        lr_factor: float = 0.5,
        lr_min: float = 1e-8,
        lr_cooldown: int = 0,
        lr_threshold: float = 0.0,
        lr_adaptive_mode: str = "simple-reset",
        lr_shape_influence: float = 1.0,
        lr_threshold_mode: str = "rel",
        early_stop_loss: float = 1e-8,
        early_stop_lr: float = 1e-10,
        early_stop_stall: int = 1000,
        **kwargs,
    ):
        if block_size != 16:
            raise ValueError("NVFP4 requires block_size=16")
        
        self.block_size = block_size
        self.pad_to_16x = pad_to_16x
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # SVD parameters
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.full_matrix = full_matrix
        
        # Optimizer configuration
        self.optimizer_choice = optimizer
        self.num_iter = num_iter
        self.no_learned_rounding = no_learned_rounding
        self.optimizer_kwargs = kwargs
        
        # LR schedule configuration
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_cooldown = lr_cooldown
        self.lr_threshold = lr_threshold
        self.lr_adaptive_mode = lr_adaptive_mode
        self.lr_shape_influence = lr_shape_influence
        self.lr_threshold_mode = lr_threshold_mode
        
        # Early stopping thresholds
        self.early_stop_loss = early_stop_loss
        self.early_stop_lr = early_stop_lr
        self.early_stop_stall = early_stop_stall
        
        print(f"LearnedNVFP4Converter initialized on device: {self.device}")
        print(f"  - Format: NVFP4 (FP4 E2M1)")
        print(f"  - Block size: {self.block_size}")
        print(
            f"  - Using optimizer: '{self.optimizer_choice}'"
            + (" (disabled - simple quant)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            print(f"  - LR schedule: {self.lr_schedule}")
    
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
        
        # Handle all-zeros tensor
        if torch.all(W_float32 == 0):
            print("  - Tensor is all zeros, skipping optimization.")
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
        
        # Compute per-tensor scale
        amax = torch.amax(torch.abs(W_float32))
        per_tensor_scale = (amax / FP4_E2M1_MAX).to(dtype=torch.float32)
        
        # Compute per-block scales
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)
        block_scale_fp32 = block_max / FP4_E2M1_MAX
        
        # Scale block scales relative to per-tensor scale
        scaled_block_scales = block_scale_fp32 / per_tensor_scale
        scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, min=F8_E4M3_EPS, max=F8_E4M3_MAX)
        scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)
        
        # Total scale for each block
        total_scale = per_tensor_scale * scaled_block_scales_fp32  # (M, N//16)
        
        if self.no_learned_rounding:
            # Simple quantization without optimization
            qdata = self._simple_quantize(W_float32, total_scale)
        else:
            # Apply learned rounding optimization
            qdata = self._optimize_nvfp4(W_float32, total_scale)
        
        # Pack to uint8
        data_packed = pack_uint4(qdata)
        
        # Convert block scales to cuBLAS tiled layout
        blocked_scales = to_blocked(
            scaled_block_scales_fp8.to(torch.float8_e4m3fn),
            flatten=False
        )
        
        # Dequantize for bias correction / output
        dequantized = self._dequantize(qdata, total_scale, W_float32.shape)
        
        # Cleanup
        del W_float32, tensor_blocks
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return data_packed, blocked_scales, per_tensor_scale, dequantized
    
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
        self, W_float32: torch.Tensor, total_scale: torch.Tensor
    ) -> torch.Tensor:
        """Simple quantization without learned rounding."""
        M, N = W_float32.shape
        tensor_blocks = W_float32.reshape(M, -1, self.block_size)
        
        data_scaled = tensor_blocks / total_scale.unsqueeze(-1)
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
        self, W_float32: torch.Tensor, total_scale: torch.Tensor
    ) -> torch.Tensor:
        """Apply learned rounding optimization for NVFP4."""
        M, N = W_float32.shape
        
        # Compute SVD for optimization
        max_rank = min(M, N)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)
        
        print(f"    - Tensor shape: [{M}, {N}], Max rank: {max_rank}. Using k={k} components.")
        
        if self.full_matrix:
            print("    - Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                print("    - Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)
        
        U_k, Vh_k = U[:, :k], Vh[:k, :]
        
        # Route to appropriate optimizer
        if self.optimizer_choice == "original":
            qdata = self._optimize_original(W_float32, total_scale, U_k, Vh_k)
        elif self.optimizer_choice == "adamw":
            qdata = self._optimize_adamw(W_float32, total_scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            qdata = self._optimize_radam(W_float32, total_scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")
        
        # Cleanup SVD tensors
        del U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        return qdata
    
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
    ) -> torch.Tensor:
        """NVFP4 optimization using original gradient descent with tier-based LR."""
        M, N = W_float32.shape
        
        # Initial quantization
        qdata_initial = self._simple_quantize(W_float32, total_scale)
        # Work with float representation for optimization
        qdata_f32 = _floatx_unpacked_to_f32(qdata_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        W_q_refined = qdata_f32.clone()
        
        best_loss = float("inf")
        best_qdata = None
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        
        # Shape-aware multiplier
        if M == N:
            small_mult = 0.95
        else:
            small_mult = 1.0
        
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
        
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (Original-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        
        for i in pbar:
            with torch.no_grad():
                current_dq = self._nvfp4_dequantize_blockwise(W_q_refined, total_scale, M, N)
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
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1
            
            # LR update based on schedule
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= effective_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - tier-based schedule
                counter_for_tier = (
                    prev_worse_counter
                    if (improved and self.lr_adaptive_mode == "no-reset")
                    else worse_loss_counter
                )
                curr_lr = self._adaptive_lr_update(curr_lr, improved, counter_for_tier, worse_loss_counter, small_mult)
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0
            
            pbar.set_postfix({
                "loss": f"{current_loss:.3e}",
                "best": f"{best_loss:.3e}",
                "lr": f"{curr_lr:.2e}",
                "worse": f"{worse_loss_counter}",
            })
            
            # Early stopping
            if self._check_early_stop(current_loss, curr_lr, worse_loss_counter):
                break
            
            # Gradient step (in FP4 value space)
            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                # Scale gradient to block-wise
                grad_blocks = grad_direction.reshape(M, -1, self.block_size)
                scaled_grad = grad_blocks * total_scale.unsqueeze(-1)
                W_q_refined -= curr_lr * scaled_grad.view(M, N)
                # Clamp to FP4 range
                W_q_refined = torch.clamp(W_q_refined, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        
        pbar.close()
        
        # Convert best float values back to FP4 encoding
        best_qdata = best_qdata if best_qdata is not None else W_q_refined
        return _f32_to_floatx_unpacked(best_qdata.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
    
    def _optimize_adamw(
        self,
        W_float32: torch.Tensor,
        total_scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """NVFP4 optimization using AdamW optimizer."""
        M, N = W_float32.shape
        
        qdata_initial = self._simple_quantize(W_float32, total_scale)
        qdata_f32 = _floatx_unpacked_to_f32(qdata_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        
        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = AdamW([delta], lr=curr_lr)
        
        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (AdamW-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        
        for i in pbar:
            optimizer.zero_grad()
            
            q_refined = qdata_f32 + delta
            current_dq = self._nvfp4_dequantize_blockwise(q_refined, total_scale, M, N)
            
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
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
            
            pbar.set_postfix({
                "loss": f"{current_loss_val:.3e}",
                "best": f"{best_loss:.3e}",
                "lr": f"{curr_lr:.2e}",
            })
            
            if self._check_early_stop(best_loss, curr_lr, worse_loss_counter):
                break
        
        pbar.close()
        
        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        return _f32_to_floatx_unpacked(final_q.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
    
    def _optimize_radam(
        self,
        W_float32: torch.Tensor,
        total_scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """NVFP4 optimization using RAdam optimizer."""
        M, N = W_float32.shape
        
        qdata_initial = self._simple_quantize(W_float32, total_scale)
        qdata_f32 = _floatx_unpacked_to_f32(qdata_initial, F4_E2M1_EBITS, F4_E2M1_MBITS)
        
        delta = torch.zeros_like(qdata_f32, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = RAdam([delta], lr=curr_lr)
        
        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0
        
        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing NVFP4 (RAdam-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        
        for i in pbar:
            optimizer.zero_grad()
            
            q_refined = qdata_f32 + delta
            current_dq = self._nvfp4_dequantize_blockwise(q_refined, total_scale, M, N)
            
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)
            
            loss.backward()
            optimizer.step()
            
            current_loss_val = loss.item()
            
            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
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
            
            pbar.set_postfix({
                "loss": f"{current_loss_val:.3e}",
                "best": f"{best_loss:.3e}",
                "lr": f"{curr_lr:.2e}",
            })
            
            if self._check_early_stop(best_loss, curr_lr, worse_loss_counter):
                break
        
        pbar.close()
        
        final_q = qdata_f32 + best_delta
        final_q = torch.clamp(final_q.detach(), -FP4_E2M1_MAX, FP4_E2M1_MAX)
        return _f32_to_floatx_unpacked(final_q.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)
    
    def _adaptive_lr_update(
        self,
        curr_lr: float,
        improved: bool,
        counter_for_tier: int,
        worse_loss_counter: int,
        small_mult: float,
    ) -> float:
        """Tier-based adaptive LR update schedule."""
        if improved and counter_for_tier < 50:
            return min(curr_lr * (1.25 * small_mult), 100.0)
        elif improved and counter_for_tier < 75:
            return min(curr_lr * (1.375 * small_mult), 100.0)
        elif improved and counter_for_tier < 100:
            return min(curr_lr * (1.5 * small_mult), 100.0)
        elif improved and counter_for_tier < 125:
            return min(curr_lr * (1.75 * small_mult), 100.0)
        elif improved and counter_for_tier < 150:
            return min(curr_lr * (2.0 * small_mult), 100.0)
        elif improved and counter_for_tier < 200:
            return min(curr_lr * (2.25 * small_mult), 100.0)
        elif improved and counter_for_tier < 250:
            return min(curr_lr * (2.5 * small_mult), 100.0)
        elif improved and counter_for_tier < 300:
            return min(curr_lr * (2.75 * small_mult), 100.0)
        elif improved:
            return min(curr_lr * (3.0 * small_mult), 100.0)
        elif worse_loss_counter < 26:
            return max(curr_lr * (0.95 * small_mult), 9e-8)
        elif worse_loss_counter < 51:
            return max(curr_lr * (0.97 * small_mult), 8e-8)
        elif worse_loss_counter < 76:
            return max(curr_lr * (0.985 * small_mult), 7e-8)
        elif worse_loss_counter < 101:
            return max(curr_lr * (0.9875 * small_mult), 6e-8)
        elif worse_loss_counter < 151:
            return max(curr_lr * (0.98875 * small_mult), 5e-8)
        elif worse_loss_counter < 201:
            return max(curr_lr * (0.99 * small_mult), 4e-8)
        elif worse_loss_counter < 251:
            return max(curr_lr * (0.99125 * small_mult), 3e-8)
        elif worse_loss_counter < 301:
            return max(curr_lr * (0.9925 * small_mult), 2e-8)
        else:
            return max(curr_lr * (0.995 * small_mult), 5e-9)
    
    def _check_early_stop(
        self, current_loss: float, curr_lr: float, worse_loss_counter: int
    ) -> bool:
        """Check early stopping conditions."""
        if current_loss < self.early_stop_loss:
            print("      - Loss is negligible. Stopping early.")
            return True
        if curr_lr < self.early_stop_lr:
            print("      - Learning rate bottomed out. Stopping early.")
            return True
        if worse_loss_counter > self.early_stop_stall:
            print("      - Loss has stalled. Stopping early.")
            return True
        return False
