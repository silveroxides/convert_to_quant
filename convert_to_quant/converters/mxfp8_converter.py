"""
MXFP8 (Microscaling FP8) Quantization Converter.

Implements MXFP8 block quantization with E8M0 (power-of-2) scaling.
Requires SM >= 10.0 (Blackwell) for hardware-accelerated matmul.

Uses comfy-kitchen CUDA/Triton kernels when available, with PyTorch fallback.

Based on comfy-kitchen (Comfy Org, Apache-2.0).
"""
import math
from typing import Tuple, Optional

import torch

from ..constants import (
    MXFP8_BLOCK_SIZE,
    MXFP8_DTYPE,
    E8M0_BIAS,
    COMPUTE_DTYPE,
)
from ..utils.float_utils import (
    roundup,
    e8m0_to_f32,
    mxfp8_to_blocked,
    mxfp8_from_blocked,
)
from ..utils.logging import info

# Check for comfy-kitchen availability
try:
    import comfy_kitchen as ck
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False

# Track if fallback warning has been shown
_FALLBACK_WARNING_SHOWN = False


class MXFP8Converter:
    """
    MXFP8 block quantization converter.

    Uses 32-element blocks with E8M0 (power-of-2) scales.
    Delegates to comfy-kitchen kernels when available for exact compatibility.

    Args:
        block_size: Block size for quantization (default: 32, required by MXFP8)
        pad_to_32x: Pad dimensions to be divisible by 32
    """

    def __init__(
        self,
        block_size: int = 32,
        pad_to_32x: bool = True,
    ):
        if block_size != 32:
            raise ValueError("MXFP8 requires block_size=32")

        self.block_size = block_size
        self.pad_to_32x = pad_to_32x

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to MXFP8 format.

        Args:
            tensor: Input tensor (2D)

        Returns:
            Tuple of (quantized_data_fp8, block_scales_e8m0)
        """
        if not HAS_COMFY_KITCHEN:
            raise RuntimeError(
                "MXFP8 quantization requires comfy_kitchen with MXFP8 support. "
                "Install from the fork with NVIDIA engineer additions."
            )

        needs_padding = False
        orig_shape = tensor.shape
        if self.pad_to_32x:
            padded_shape = (roundup(orig_shape[0], 32), roundup(orig_shape[1], 32))
            needs_padding = padded_shape != orig_shape

        # CUDA kernel only supports FP16/BF16 input
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            tensor = tensor.to(torch.bfloat16)

        qdata, block_scales = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)
        return qdata, block_scales

    def _quantize_pytorch(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch quantization fallback (matches comfy-kitchen exactly)."""
        orig_shape = tensor.shape
        device = tensor.device

        # Handle padding
        rows, cols = orig_shape
        if self.pad_to_32x:
            padded_rows = roundup(rows, 32)
            padded_cols = roundup(cols, 32)
            if padded_rows != rows or padded_cols != cols:
                tensor = torch.nn.functional.pad(
                    tensor, (0, padded_cols - cols, 0, padded_rows - rows)
                )

        M, N = tensor.shape
        num_blocks = N // self.block_size

        # Reshape to blocks along last dimension
        tensor_blocks = tensor.reshape(M, num_blocks, self.block_size)

        # Compute per-block max absolute values
        block_max = torch.amax(torch.abs(tensor_blocks), dim=-1)

        # Compute scale needed to fit in FP8 range
        fp8_max = torch.finfo(MXFP8_DTYPE).max
        scale_needed = block_max.float() / fp8_max
        scale_needed = torch.clamp(scale_needed, min=2**(-127))  # Min E8M0 value

        # Convert to E8M0 exponent (round up to ensure values fit)
        log2_scale = torch.log2(scale_needed)
        exp_biased = torch.ceil(log2_scale).to(torch.int32) + E8M0_BIAS
        exp_biased = torch.clamp(exp_biased, 0, 254)  # Valid E8M0 range [0, 254]

        block_scales_e8m0 = exp_biased.to(torch.uint8)
        block_scales_f32 = e8m0_to_f32(block_scales_e8m0)

        # Handle zero blocks
        zero_mask = (block_max == 0)
        block_scales_f32 = torch.where(
            zero_mask,
            torch.ones_like(block_scales_f32),
            block_scales_f32
        )

        # Scale data and quantize to FP8 E4M3
        data_scaled = tensor_blocks.float() / block_scales_f32.unsqueeze(-1)
        data_scaled = torch.where(
            zero_mask.unsqueeze(-1),
            torch.zeros_like(data_scaled),
            data_scaled
        )

        # Clamp to FP8 range and convert
        data_scaled = torch.clamp(data_scaled, -fp8_max, fp8_max)
        qdata = data_scaled.reshape(M, N).to(MXFP8_DTYPE)

        # Handle zero blocks in scales
        block_scales_e8m0 = torch.where(
            zero_mask,
            torch.zeros_like(block_scales_e8m0),
            block_scales_e8m0
        )

        # Convert block scales to cuBLAS tiled layout
        blocked_scales = mxfp8_to_blocked(block_scales_e8m0, flatten=False)

        return qdata, blocked_scales

    def dequantize(
        self,
        qdata: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize MXFP8 tensor back to float.

        Args:
            qdata: Quantized FP8 tensor (float8_e4m3fn)
            block_scales: Block scales in cuBLAS tiled layout (E8M0/uint8)
            output_dtype: Target output dtype

        Returns:
            Dequantized tensor
        """
        if not HAS_COMFY_KITCHEN:
            raise RuntimeError(
                "MXFP8 dequantization requires comfy_kitchen with MXFP8 support. "
                "Install from the fork with NVIDIA engineer additions."
            )

        return ck.dequantize_mxfp8(qdata, block_scales, output_dtype)

    def _dequantize_pytorch(
        self,
        qdata: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Pure PyTorch dequantization fallback."""
        orig_shape = qdata.shape
        M, N = orig_shape

        # Convert FP8 data to float32
        data_f32 = qdata.float()

        # Reshape to blocks
        data_blocks = data_f32.reshape(M, -1, self.block_size)

        # Unswizzle block_scales from cuBLAS tiled layout
        num_blocks_per_row = N // self.block_size
        block_scales_unswizzled = mxfp8_from_blocked(
            block_scales,
            num_rows=M,
            num_cols=num_blocks_per_row
        )

        # Convert E8M0 scales to float32
        scales_f32 = e8m0_to_f32(block_scales_unswizzled)

        # Apply scaling
        data_dequantized = data_blocks * scales_f32.unsqueeze(-1)

        return data_dequantized.view(M, N).to(output_dtype)


def quantize_mxfp8(
    tensor: torch.Tensor,
    pad_to_32x: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to quantize a tensor to MXFP8 format.

    Args:
        tensor: Input tensor (2D)
        pad_to_32x: Pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_data_fp8, block_scales_e8m0)
    """
    converter = MXFP8Converter(pad_to_32x=pad_to_32x)
    return converter.quantize(tensor)


def dequantize_mxfp8(
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convenience function to dequantize MXFP8 tensor.

    Args:
        qdata: Quantized FP8 tensor (float8_e4m3fn)
        block_scales: Block scales in cuBLAS tiled layout
        output_dtype: Target output dtype

    Returns:
        Dequantized tensor
    """
    converter = MXFP8Converter()
    return converter.dequantize(qdata, block_scales, output_dtype)
