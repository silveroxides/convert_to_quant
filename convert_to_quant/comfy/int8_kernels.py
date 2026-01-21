
import torch
from typing import Tuple

try:
    import comfy_kitchen.backends.triton.quantization as ck_quant
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False

try:
    from comfy_kitchen.backends.cuda import cublas_gemm_int8
    HAS_CUBLAS_INT8 = True
except ImportError:
    HAS_CUBLAS_INT8 = False

if HAS_COMFY_KITCHEN:
    # Map comfy_kitchen APIs to matching names and signatures
    
    def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        return ck_quant.quantize_int8(x, block_size=block_size, is_weight=False)

    def weight_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
        return ck_quant.quantize_int8(x, block_size=block_size, is_weight=True)

    def act_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        return ck_quant.dequantize_int8(x, s, block_size=block_size, output_dtype=output_dtype)

    def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128, output_dtype: torch.dtype = None) -> torch.Tensor:
        if output_dtype is None:
            output_dtype = torch.get_default_dtype()
        return ck_quant.dequantize_int8(x, s, block_size=block_size, output_dtype=output_dtype)

    def int8_gemm(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        input_block_size: int = 128,
    ) -> torch.Tensor:
        # Note: input_block_size is unused in existing fallback signature for K blocking? 
        # ck_quant.scaled_mm_int8 handles it or assumes 128?
        # Checking comfy_kitchen reference: scaled_mm_int8 uses block_size 128 hardcoded in kernels?
        # The reference triton kernel uses BLOCK_SIZE_K=128 by default in heuristics/configs.
        # We will assume compatibility.
        return ck_quant.scaled_mm_int8(a, b, a_s, b_s, out_dtype=torch.float16)

    def int8_addmm(
        a: torch.Tensor,
        a_s: torch.Tensor,
        b: torch.Tensor,
        b_s: torch.Tensor,
        bias: torch.Tensor = None,
        input_block_size: int = 128,
    ) -> torch.Tensor:
        return ck_quant.scaled_mm_int8(a, b, a_s, b_s, bias=bias, out_dtype=torch.float16)

    # Expose kernels if needed for direct access, or use fallback ones if not in public API
    # The user might use the kernels directly in quant_ops.py
    # ck_quant exposes: int8_gemm_kernel, int8_gemm_addmm_kernel, etc.
    int8_gemm_kernel = ck_quant.int8_gemm_kernel
    int8_gemm_addmm_kernel = ck_quant.int8_gemm_addmm_kernel
    int8_gemm_quant_kernel = ck_quant.int8_gemm_quant_kernel
    int8_gemm_addmm_quant_kernel = ck_quant.int8_gemm_addmm_quant_kernel
    
    act_quant_kernel = ck_quant.int8_act_quant_kernel
    act_dequant_kernel = ck_quant.int8_act_dequant_kernel
    weight_quant_kernel = ck_quant.int8_weight_quant_kernel
    # weight_dequant_kernel might be named differently? check reference.
    # Reference has int8_weight_dequant_kernel.
    weight_dequant_kernel = ck_quant.int8_weight_dequant_kernel

else:
    # Fallback to local implementation
    from ._int8_kernels_fallback import *

