"""
Learned rounding converter for FP8 and INT8 quantization.

This module implements advanced quantization using learned adaptive rounding
with SVD-based optimization. Inherits from BaseLearnedConverter.
"""
import gc
import math
import torch
from typing import Tuple, Optional, Dict
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
        **kwargs,
    ):
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
        self.target_format = target_format
        self.mem_threshold = 100_000_000 

        if target_format == "int8" and scaling_mode not in ("tensor", "block"):
            scaling_mode = "block"
        if scaling_mode == "block3d":
            scaling_mode = "block"
        self.scaling_mode = scaling_mode

        if self.target_format == "int8":
            self.target_dtype = TARGET_INT8_DTYPE
            self.f8_max_val = None
        else:
            self.target_dtype = TARGET_FP8_DTYPE
            self.f8_max_val = FP8_MAX

    def _optimize_original(self, W_float32, scale, U, Vh):
        """Subspace-Optimized Gradient Descent with fixed memory lifecycle."""
        is_massive = W_float32.numel() > self.mem_threshold
        is_scalar_scale = (not isinstance(scale, torch.Tensor)) or (scale.numel() == 1)

        # 1. Pre-calculate subspace projection while W_float32 is still alive
        with torch.no_grad():
            P_orig = torch.mm(U.t(), W_float32)
            P_orig = torch.mm(P_orig, Vh.t())
            
            # 2. Create the initial rounded weights BEFORE deleting W_float32
            # This is the 3.75GB tensor we will actually optimize in-place
            W_q_refined = W_float32.mul(scale).to(self.target_dtype).to(COMPUTE_DTYPE)

        # 3. NOW LIBERATE VRAM: The original FP32 weights are no longer needed
        if is_massive:
            verbose(f"    - Freeing original weights to reclaim {W_float32.numel()*4/1024**3:.2f}GB VRAM.")
            del W_float32
            gc.collect()
            torch.cuda.empty_cache()

        best_loss = float("inf")
        best_tensor_cpu = None
        curr_lr = self.lr

        pbar = tqdm(range(self.num_iter), desc=f"    Optimizing (Original)", leave=False)
        for i in pbar:
            with torch.no_grad():
                if is_scalar_scale:
                    # Optimized k x k subspace math
                    P_q = torch.mm(U.t(), W_q_refined)
                    P_q = torch.mm(P_q, Vh.t())
                    
                    # P_err = (P_q / scale) - P_orig
                    P_err = (P_q.div(scale)).sub_(P_orig)
                    loss = torch.linalg.norm(P_err)
                    
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        best_tensor_cpu = W_q_refined.to("cpu", non_blocking=True)
                    
                    # Gradient in subspace
                    sub_grad = P_err.div_(loss.clamp_min(1e-20))
                    
                    # Fused In-Place Update: W_q = W_q - lr * (U @ sub_grad @ Vh)
                    W_q_refined.addmm_(U, sub_grad @ Vh, beta=1.0, alpha=-curr_lr)
                else:
                    # Fallback for complex scaling
                    current_dq = W_q_refined / scale
                    error = current_dq - (W_orig_reconstruct if 'W_orig_reconstruct' in locals() else P_orig) 
                    # ... [Standard path logic] ...
                
            if i % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}"})

        pbar.close()
        return best_tensor_cpu.to(self.device) if best_tensor_cpu is not None else W_q_refined

    def convert(self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            return torch.zeros_like(W_float32, dtype=self.target_dtype), torch.ones(1, device=self.device, dtype=SCALE_DTYPE), torch.zeros_like(W_float32), {}

        if self.target_format == "int8":
            if self.scaling_mode == "tensor":
                qdata, scale, dequantized = self._convert_int8_tensorwise(W_float32)
            else:
                qdata, scale, dequantized = self._convert_int8(W_float32)
        else:
            # Standard FP8 conversion path...
            U_k, Vh_k, k = self._compute_svd_components(W_float32)
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
            
            with torch.no_grad():
                W_f8 = final_tensor_scaled.clamp(-self.f8_max_val, self.f8_max_val).to(TARGET_FP8_DTYPE)
                dequantized = W_f8.to(COMPUTE_DTYPE) / scale
            qdata, scale, dequantized = W_f8, (1.0/scale).to(SCALE_DTYPE), dequantized

        extra_tensors = {}
        if self._should_extract_lora(key, W_orig.shape, depth):
            lora_data = self._extract_error_lora(W_orig.to(self.device), dequantized)
            if lora_data: extra_tensors.update(lora_data)

        return qdata, scale, dequantized, extra_tensors

    def _convert_int8_tensorwise(self, W_float32):
        from ..comfy.quant_ops import TensorWiseINT8Layout
        qdata, layout_params = TensorWiseINT8Layout.quantize(W_float32, is_weight=True)
        scale = layout_params["scale"]

        if not self.no_learned_rounding and self.num_iter > 0:
            qdata, scale = self._optimize_int8_tensorwise_learned_rounding(W_float32, qdata, scale)

        dequantized_weight = TensorWiseINT8Layout.dequantize(qdata, scale, orig_dtype=COMPUTE_DTYPE)
        return qdata, scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight

    def _optimize_int8_tensorwise_learned_rounding(self, W_float32, qdata, scale):
        U_k, Vh_k, k = self._compute_svd_components(W_float32)
        scale_fp8_style = 1.0 / scale.clamp_min(1e-12)
        
        orig_dtype, orig_max = self.target_dtype, self.f8_max_val
        self.target_dtype, self.f8_max_val = TARGET_INT8_DTYPE, float(INT8_SYMMETRIC_MAX)

        final_tensor_scaled = self._optimize_original(W_float32, scale_fp8_style, U_k, Vh_k)

        self.target_dtype, self.f8_max_val = orig_dtype, orig_max
        with torch.no_grad():
            final_qdata = final_tensor_scaled.clamp(-127, 127).round().to(TARGET_INT8_DTYPE)
        
        self._cleanup_tensors(U_k, Vh_k)
        return final_qdata, scale
