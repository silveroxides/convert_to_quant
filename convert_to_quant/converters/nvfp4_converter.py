"""
NVFP4 (E2M1) Quantization Converter.

Implements NVIDIA FP4 E2M1 block quantization with learned rounding optimization.
Requires SM >= 10.0 (datacenter Blackwell) or SM >= 12.0 (consumer RTX 50 series).

Uses comfy-kitchen CUDA/Triton kernels when available, with PyTorch fallback.

Based on comfy-kitchen (Comfy Org, Apache-2.0) and PyTorch AO (Meta, BSD-3-Clause).
"""
import math
from typing import Tuple, Optional

import torch

from ..constants import (
    FP4_E2M1_MAX,
    FP4_BLOCK_SIZE,
    COMPUTE_DTYPE,
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

# Check for comfy-kitchen availability
try:
    import comfy_kitchen as ck
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False


class NVFP4Converter:
    """
    NVIDIA FP4 E2M1 block quantization converter.

    Uses 16-element blocks with per-block FP8 scales and per-tensor global scale.
    Delegates to comfy-kitchen kernels when available for exact compatibility.

    Args:
        block_size: Block size for quantization (default: 16, required by NVFP4)
        pad_to_16x: Pad dimensions to be divisible by 16
        optimize: Enable learned rounding optimization
        num_iter: Number of optimization iterations
        lr: Learning rate for optimization
    """

    def __init__(
        self,
        block_size: int = 16,
        pad_to_16x: bool = True,
        optimize: bool = True,
        num_iter: int = 100,
        lr: float = 1e-3,
    ):
        if block_size != 16:
            raise ValueError("NVFP4 requires block_size=16")

        self.block_size = block_size
        self.pad_to_16x = pad_to_16x
        self.optimize = optimize
        self.num_iter = num_iter
        self.lr = lr

    def quantize(
        self,
        tensor: torch.Tensor,
        per_tensor_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NVFP4 format.

        Args:
            tensor: Input tensor (2D)
            per_tensor_scale: Optional global scale (computed if None)

        Returns:
            Tuple of (quantized_data, block_scales, per_tensor_scale)
        """
        device = tensor.device

        # Compute per-tensor scale if not provided
        # Formula: scale = amax / (F8_E4M3_MAX * F4_E2M1_MAX)
        if per_tensor_scale is None:
            amax = torch.amax(torch.abs(tensor))
            per_tensor_scale = amax / (F8_E4M3_MAX * FP4_E2M1_MAX)

        per_tensor_scale = per_tensor_scale.to(device=device, dtype=torch.float32)

        # Use comfy-kitchen kernel if available
        if HAS_COMFY_KITCHEN:
            # CUDA kernel requires FP16/BF16 input (not float32)
            qdata, block_scales = ck.quantize_nvfp4(
                tensor.to(torch.bfloat16), per_tensor_scale, pad_16x=self.pad_to_16x
            )
            return qdata, block_scales, per_tensor_scale

        # Fallback: PyTorch implementation (matches comfy-kitchen exactly)
        return self._quantize_pytorch(tensor, per_tensor_scale)

    def _quantize_pytorch(
        self,
        tensor: torch.Tensor,
        per_tensor_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pure PyTorch quantization fallback (matches comfy-kitchen exactly)."""
        orig_shape = tensor.shape
        device = tensor.device

        # Handle padding
        rows, cols = orig_shape
        if self.pad_to_16x:
            padded_rows = roundup(rows, 16)
            padded_cols = roundup(cols, 16)
            if padded_rows != rows or padded_cols != cols:
                tensor = torch.nn.functional.pad(
                    tensor, (0, padded_cols - cols, 0, padded_rows - rows)
                )
                orig_shape = tensor.shape

        # Reshape to blocks
        tensor_blocks = tensor.reshape(orig_shape[0], -1, self.block_size)

        # Compute per-block scales
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)
        block_scale = block_max.to(torch.float32) / FP4_E2M1_MAX

        # Scale block scales by per-tensor scale
        scaled_block_scales = block_scale / per_tensor_scale
        # Match comfy-kitchen: only max clamp, no min clamp
        scaled_block_scales_fp8 = torch.clamp(scaled_block_scales, max=F8_E4M3_MAX)
        scaled_block_scales_fp32 = _float8_round(scaled_block_scales_fp8)

        # Compute total scale for data
        total_scale = per_tensor_scale * scaled_block_scales_fp32

        # Handle zero blocks (from padding): avoid 0/0 NaN - matches comfy-kitchen
        zero_scale_mask = (total_scale == 0)
        total_scale_safe = torch.where(zero_scale_mask, torch.ones_like(total_scale), total_scale)

        # Scale and quantize data
        data_scaled = tensor_blocks.float() / total_scale_safe.unsqueeze(-1)
        data_scaled = torch.where(zero_scale_mask.unsqueeze(-1), torch.zeros_like(data_scaled), data_scaled)

        data_scaled = torch.clamp(data_scaled, -FP4_E2M1_MAX, FP4_E2M1_MAX)
        data_scaled = data_scaled.view(orig_shape)

        # Convert to FP4 E2M1 format
        data_lp = _f32_to_floatx_unpacked(data_scaled.float(), F4_E2M1_EBITS, F4_E2M1_MBITS)

        # Pack two FP4 values per uint8
        data_packed = pack_uint4(data_lp)

        # Convert block scales to cuBLAS tiled layout
        blocked_scales = to_blocked(
            scaled_block_scales_fp8.to(torch.float8_e4m3fn),
            flatten=False
        )

        return data_packed, blocked_scales, per_tensor_scale

    def dequantize(
        self,
        qdata: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize NVFP4 tensor back to float.

        Args:
            qdata: Quantized FP4 tensor (packed as uint8)
            per_tensor_scale: Global scale factor
            block_scales: Block scales in cuBLAS tiled layout (float8_e4m3fn)
            output_dtype: Target output dtype

        Returns:
            Dequantized tensor
        """
        # Use comfy-kitchen kernel if available
        if HAS_COMFY_KITCHEN:
            return ck.dequantize_nvfp4(qdata, per_tensor_scale, block_scales, output_dtype)

        # Fallback: PyTorch implementation
        return self._dequantize_pytorch(qdata, per_tensor_scale, block_scales, output_dtype)

    def _dequantize_pytorch(
        self,
        qdata: torch.Tensor,
        per_tensor_scale: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Pure PyTorch dequantization fallback."""
        # Unpack FP4 data
        data_unpacked = unpack_uint4(qdata)

        # Convert unpacked FP4 to float32
        data_f32 = _floatx_unpacked_to_f32(data_unpacked, F4_E2M1_EBITS, F4_E2M1_MBITS)

        orig_shape = data_f32.shape

        # Reshape to blocks
        data_reshaped = data_f32.reshape(orig_shape[0], -1, self.block_size)

        # Unswizzle block_scales from cuBLAS tiled layout
        num_blocks_per_row = orig_shape[1] // self.block_size
        block_scales_unswizzled = from_blocked(
            block_scales,
            num_rows=orig_shape[0],
            num_cols=num_blocks_per_row
        )

        # Compute total scale
        total_scale = per_tensor_scale * block_scales_unswizzled.to(torch.float32)

        # Apply scaling
        data_dequantized = data_reshaped * total_scale.unsqueeze(-1)

        return data_dequantized.view(orig_shape).to(output_dtype)


def quantize_nvfp4(
    tensor: torch.Tensor,
    per_tensor_scale: Optional[torch.Tensor] = None,
    pad_to_16x: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convenience function to quantize a tensor to NVFP4 format.

    Args:
        tensor: Input tensor (2D)
        per_tensor_scale: Optional global scale (auto-computed if None)
        pad_to_16x: Pad dimensions to be divisible by 16

    Returns:
        Tuple of (quantized_data, block_scales, per_tensor_scale)
    """
    converter = NVFP4Converter(pad_to_16x=pad_to_16x, optimize=False)
    return converter.quantize(tensor, per_tensor_scale)


def dequantize_nvfp4(
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    per_tensor_scale: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convenience function to dequantize NVFP4 tensor.

    Args:
        qdata: Quantized FP4 tensor (packed as uint8)
        block_scales: Block scales in cuBLAS tiled layout
        per_tensor_scale: Global scale factor
        output_dtype: Target output dtype

    Returns:
        Dequantized tensor
    """
    converter = NVFP4Converter(optimize=False)
    return converter.dequantize(qdata, per_tensor_scale, block_scales, output_dtype)
