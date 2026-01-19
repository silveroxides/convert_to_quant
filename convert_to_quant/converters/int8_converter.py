"""
INT8 Block-32 Quantization Converter.

Implements INT8 block quantization with swizzled FP32 scales (SWIZZLE_32_4_4).
Requires SM >= 8.0 (Ampere) for optimal performance.

Uses comfy-kitchen CUDA kernels for quantization/dequantization.

Based on comfy-kitchen (Comfy Org, Apache-2.0).
"""
from typing import Tuple

import torch

from ..constants import (
    COMPUTE_DTYPE,
)
from ..utils.float_utils import roundup
from ..utils.logging import info

# Check for comfy-kitchen availability
try:
    import comfy_kitchen as ck
    HAS_COMFY_KITCHEN = True
except ImportError:
    HAS_COMFY_KITCHEN = False

# Track if fallback warning has been shown
_FALLBACK_WARNING_SHOWN = False


class INT8Block32Converter:
    """
    INT8 block-32 quantization converter with swizzled FP32 scales.

    Uses 32-element blocks with FP32 scales in cuBLAS SWIZZLE_32_4_4 format.
    Delegates to comfy-kitchen CUDA kernels for exact compatibility with
    scaled_mm_v2 matmul integration.

    Args:
        pad_to_32x: Pad dimensions to be divisible by 32
    """

    def __init__(self, pad_to_32x: bool = True):
        self.block_size = 32  # Fixed for CUDA backend
        self.pad_to_32x = pad_to_32x

    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to INT8 format with block-32 swizzled scales.

        Args:
            tensor: Input tensor (2D)

        Returns:
            Tuple of (quantized_int8_data, block_scales_fp32_swizzled)
        """
        if not HAS_COMFY_KITCHEN:
            raise RuntimeError(
                "INT8 block-32 quantization requires comfy_kitchen with CUDA backend. "
                "Install comfy-kitchen with CUDA support."
            )

        needs_padding = False
        orig_shape = tensor.shape
        if self.pad_to_32x:
            padded_shape = (roundup(orig_shape[0], 32), roundup(orig_shape[1], 32))
            needs_padding = padded_shape != orig_shape

        # CUDA kernel only supports FP16/BF16 input
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            tensor = tensor.to(torch.bfloat16)

        qdata, block_scales = ck.quantize_int8_block32(tensor, pad_32x=needs_padding)
        return qdata, block_scales

    def convert(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert tensor to INT8 block-32 format (interface for fp8_conversion.py).

        Args:
            tensor: Input tensor (2D)

        Returns:
            Tuple of (q_tensor, scale, dequant_w) matching LearnedRoundingConverter interface
        """
        # Move to CUDA for quantization
        orig_dtype = tensor.dtype
        device = tensor.device
        tensor_cuda = tensor.cuda()

        qdata, block_scales = self.quantize(tensor_cuda)

        # Dequantize for bias correction
        dequant_w = self.dequantize(qdata, block_scales, output_dtype=orig_dtype)

        return qdata, block_scales, dequant_w

    def _quantize_pytorch(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure PyTorch quantization fallback (for reference/testing)."""
        global _FALLBACK_WARNING_SHOWN
        if not _FALLBACK_WARNING_SHOWN:
            info("INT8 block-32: Using PyTorch fallback (no swizzled layout)")
            _FALLBACK_WARNING_SHOWN = True

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
        num_blocks_m = M // self.block_size
        num_blocks_n = N // self.block_size

        # Reshape to 2D blocks
        tensor_blocks = tensor.reshape(
            num_blocks_m, self.block_size, num_blocks_n, self.block_size
        )
        tensor_blocks = tensor_blocks.permute(0, 2, 1, 3)  # (bm, bn, bs, bs)

        # Compute per-block max absolute values
        block_max = torch.amax(torch.abs(tensor_blocks), dim=(-2, -1))

        # Compute scale for symmetric INT8 quantization
        int8_max = 127.0
        scale = block_max.float() / int8_max
        scale = torch.clamp(scale, min=1e-10)  # Avoid division by zero

        # Quantize data
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)
        data_scaled = tensor_blocks.float() / scale_expanded
        data_scaled = torch.clamp(data_scaled, -127.0, 127.0)
        data_scaled = torch.round(data_scaled)

        # Reshape back to original layout
        qdata = data_scaled.permute(0, 2, 1, 3).reshape(M, N).to(torch.int8)

        # Note: This returns non-swizzled scales (simple 2D grid)
        # The CUDA backend returns swizzled scales for scaled_mm_v2
        return qdata, scale

    def dequantize(
        self,
        qdata: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        Dequantize INT8 tensor back to float.

        Args:
            qdata: Quantized INT8 tensor
            block_scales: Block scales in cuBLAS swizzled layout (FP32)
            output_dtype: Target output dtype

        Returns:
            Dequantized tensor
        """
        if not HAS_COMFY_KITCHEN:
            raise RuntimeError(
                "INT8 block-32 dequantization requires comfy_kitchen with CUDA backend. "
                "Install comfy-kitchen with CUDA support."
            )

        return ck.dequantize_int8_block32(qdata, block_scales, output_type=output_dtype)

    def _dequantize_pytorch(
        self,
        qdata: torch.Tensor,
        block_scales: torch.Tensor,
        output_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """Pure PyTorch dequantization fallback (assumes non-swizzled scales)."""
        M, N = qdata.shape
        num_blocks_m = M // self.block_size
        num_blocks_n = N // self.block_size

        # Convert INT8 data to float32
        data_f32 = qdata.float()

        # Reshape to blocks
        data_blocks = data_f32.reshape(
            num_blocks_m, self.block_size, num_blocks_n, self.block_size
        )
        data_blocks = data_blocks.permute(0, 2, 1, 3)

        # Apply scaling
        scale_expanded = block_scales.unsqueeze(-1).unsqueeze(-1)
        data_dequantized = data_blocks * scale_expanded

        # Reshape back
        result = data_dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return result.to(output_dtype)


def quantize_int8_block32(
    tensor: torch.Tensor,
    pad_to_32x: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to quantize a tensor to INT8 block-32 format.

    Args:
        tensor: Input tensor (2D)
        pad_to_32x: Pad dimensions to be divisible by 32

    Returns:
        Tuple of (quantized_int8_data, block_scales_fp32_swizzled)
    """
    converter = INT8Block32Converter(pad_to_32x=pad_to_32x)
    return converter.quantize(tensor)


def dequantize_int8_block32(
    qdata: torch.Tensor,
    block_scales: torch.Tensor,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Convenience function to dequantize INT8 block-32 tensor.

    Args:
        qdata: Quantized INT8 tensor
        block_scales: Block scales in cuBLAS swizzled layout
        output_dtype: Target output dtype

    Returns:
        Dequantized tensor
    """
    converter = INT8Block32Converter()
    return converter.dequantize(qdata, block_scales, output_dtype)
