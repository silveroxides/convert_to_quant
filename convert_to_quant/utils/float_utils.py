"""
Float utilities for sub-byte quantization formats.

Provides FP4 E2M1 encoding/decoding, uint4 packing, and cuBLAS tiled layout
transformations for NVFP4 quantization support.

Portions derived from:
- comfy-kitchen (Comfy Org, Apache-2.0)
- PyTorch AO (Meta Platforms, BSD-3-Clause)
"""
import torch

def _n_ones(n: int) -> int:
    """Generate a bitmask with n ones."""
    return (1 << n) - 1


# Float format constants
EBITS_F32, MBITS_F32 = 8, 23
F32_EXP_BIAS = _n_ones(EBITS_F32 - 1)

# FP4 E2M1 constants
F4_E2M1_MAX = 6.0
F4_E2M1_EPS = 0.5
F4_E2M1_EBITS = 2
F4_E2M1_MBITS = 1

# FP8 constants
F8_E4M3_MAX = 448.0
F8_E4M3_EPS = 0.125
F8_E5M2_MAX = 57344.0
F8_E5M2_EPS = 0.0625

def roundup(x: int, multiple: int) -> int:
    """Round up x to the nearest multiple."""
    return ((x + multiple - 1) // multiple) * multiple

def ceil_div(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b

def down_size(size: tuple) -> tuple:
    """Halve the last dimension (for packing)."""
    assert size[-1] % 2 == 0, f"{size} last dim not divisible by two"
    return (*size[:-1], size[-1] // 2)

def up_size(size: tuple) -> tuple:
    """Double the last dimension (for unpacking)."""
    return (*size[:-1], size[-1] * 2)

def pack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Pack two 4-bit values into one uint8."""
    shape = uint8_data.shape
    assert shape[-1] % 2 == 0
    uint8_data = uint8_data.contiguous().view(-1)
    return (uint8_data[::2] << 4 | uint8_data[1::2]).view(down_size(shape))

def unpack_uint4(uint8_data: torch.Tensor) -> torch.Tensor:
    """Unpack uint8 into two 4-bit values."""
    assert uint8_data.is_contiguous()
    shape = uint8_data.shape
    first_elements = (uint8_data >> 4).to(torch.uint8)
    second_elements = (uint8_data & 0b1111).to(torch.uint8)
    return torch.stack([first_elements, second_elements], dim=-1).view(up_size(shape))

def _float8_round(x: torch.Tensor) -> torch.Tensor:
    """Round to FP8 precision."""
    return x.to(torch.float8_e4m3fn).to(torch.float32)

def _f32_to_floatx_unpacked(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """
    Convert FP32 to sub-byte float format (e.g., FP4 E2M1).

    Output: uint8 tensor with bit encoding in least significant bits.
    """
    assert x.dtype == torch.float
    assert 1 + ebits + mbits <= 8

    exp_bias = _n_ones(ebits - 1)
    max_int = _n_ones(ebits + mbits)
    sign_mask = 1 << (ebits + mbits)
    magic_adder = _n_ones(MBITS_F32 - mbits - 1)
    max_normal = 2 ** (_n_ones(ebits) - exp_bias) * (_n_ones(mbits + 1) / (2**mbits))
    min_normal = 2 ** (1 - exp_bias)

    denorm_exp = (F32_EXP_BIAS - exp_bias) + (MBITS_F32 - mbits) + 1
    denorm_mask_int = denorm_exp << MBITS_F32
    denorm_mask_float = torch.tensor(denorm_mask_int, dtype=torch.int32).view(torch.float32)

    # Extract sign and work with positive values
    x = x.view(torch.int32)
    sign = x & 0x80000000
    x = x ^ sign
    x = x.view(torch.float)

    # Create masks for saturation, denormal, and normal branches
    saturate_mask = x >= max_normal
    denormal_mask = torch.logical_and(~saturate_mask, x < min_normal)
    normal_mask = ~(saturate_mask | denormal_mask)

    # Denormal path
    denormal_x = x + denorm_mask_float
    denormal_x = denormal_x.view(torch.int32)
    denormal_x -= denorm_mask_int
    denormal_x = denormal_x.to(torch.uint8)

    # Normal path
    normal_x = x.view(torch.int32)
    mant_odd = (normal_x >> (MBITS_F32 - mbits)) & 1
    val_to_add = ((exp_bias - F32_EXP_BIAS) << MBITS_F32) + magic_adder
    normal_x += val_to_add
    normal_x += mant_odd
    normal_x = normal_x >> (MBITS_F32 - mbits)
    normal_x = normal_x.to(torch.uint8)

    # Combine branches
    x = torch.full_like(x, max_int, dtype=torch.uint8)
    x = torch.where(denormal_mask, denormal_x, x)
    x = torch.where(normal_mask, normal_x, x)

    # Add sign back
    sign_lp = sign >> (MBITS_F32 + EBITS_F32 - mbits - ebits)
    sign_lp = sign_lp.to(torch.uint8) & sign_mask
    x = x | sign_lp

    return x.to(torch.uint8)

def _floatx_unpacked_to_f32(x: torch.Tensor, ebits: int, mbits: int) -> torch.Tensor:
    """
    Convert sub-byte float format (e.g., FP4 E2M1) back to FP32.

    Input: uint8 tensor with bit encoding in least significant bits.
    """
    assert x.dtype == torch.uint8
    assert 1 + ebits + mbits <= 8

    sign_mask = 1 << (ebits + mbits)
    exp_bias = _n_ones(ebits - 1)
    mantissa_mask = _n_ones(mbits)

    sign_lp = x & sign_mask
    x_pos = x ^ sign_lp

    zero_mask = x_pos == 0
    denormal_mask = (x_pos > 0) & ((x_pos >> mbits) == 0)

    # Normal path
    exp_biased_lp = x_pos >> mbits
    exp_biased_f32 = exp_biased_lp - exp_bias + F32_EXP_BIAS
    exp_biased_f32 = exp_biased_f32.to(torch.int32) << MBITS_F32
    mantissa_lp_int32 = (x_pos & mantissa_mask).to(torch.int32)
    mantissa_f32 = mantissa_lp_int32 << (MBITS_F32 - mbits)
    result = exp_biased_f32 | mantissa_f32

    result[zero_mask] = 0

    denormal_exp_biased = 1 - exp_bias + F32_EXP_BIAS

    # Fast path for mbits=1 (FP4 E2M1)
    if mbits == 1:
        result[denormal_mask] = (denormal_exp_biased - mbits) << MBITS_F32
    else:
        for i in range(mbits):
            for mantissa_cmp in range(1 << i, 1 << (i + 1)):
                left_shift = mbits - i
                mantissa_f32_val = (mantissa_cmp - (1 << i)) << (left_shift + MBITS_F32 - mbits)
                exp_biased_f32_val = (denormal_exp_biased - left_shift) << MBITS_F32
                mantissa_lp_int32[mantissa_lp_int32 == mantissa_cmp] = exp_biased_f32_val + mantissa_f32_val
        result = torch.where(denormal_mask, mantissa_lp_int32, result)

    # Add sign back
    sign_f32 = sign_lp.to(torch.int32) << (MBITS_F32 - mbits + EBITS_F32 - ebits)
    result = result | sign_f32

    return result.view(torch.float)

def to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange matrix to cuBLAS 2D block scaling factors layout.

    See: https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout

    Args:
        input_matrix: Input tensor of shape (H, W)
        flatten: If True, return flattened tensor

    Returns:
        Rearranged tensor for cuBLAS block layout
    """
    rows, cols = input_matrix.shape
    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    padded = input_matrix
    if (rows, cols) != (padded_rows, padded_cols):
        padded = torch.zeros(
            (padded_rows, padded_cols),
            device=input_matrix.device,
            dtype=input_matrix.dtype,
        )
        padded[:rows, :cols] = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    if flatten:
        return rearranged.flatten()
    return rearranged.reshape(padded_rows, padded_cols)

def from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """
    Reverse cuBLAS tiled layout back to normal (H, W) layout.

    Args:
        blocked_matrix: Swizzled tensor from cuBLAS layout
        num_rows: Desired output rows (unpadded)
        num_cols: Desired output cols (unpadded)

    Returns:
        Unswizzled tensor of shape (num_rows, num_cols)
    """
    n_row_blocks = ceil_div(num_rows, 128)
    n_col_blocks = ceil_div(num_cols, 4)

    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    step1 = blocked_matrix.reshape(-1, 32, 16)
    step2 = step1.reshape(-1, 32, 4, 4).transpose(1, 2)
    step3 = step2.reshape(n_row_blocks, n_col_blocks, 4, 32, 4)
    step4 = step3.reshape(n_row_blocks, n_col_blocks, 128, 4)
    step5 = step4.permute(0, 2, 1, 3)
    unblocked = step5.reshape(padded_rows, padded_cols)

    return unblocked[:num_rows, :num_cols]


def fp4_x2_to_f32(packed_fp4: torch.Tensor) -> torch.Tensor:
    """Unpack and dequantize FP4 E2M1 values to float32."""
    unpacked = unpack_uint4(packed_fp4)
    return _floatx_unpacked_to_f32(unpacked, F4_E2M1_EBITS, F4_E2M1_MBITS)


# =============================================================================
# E8M0 Conversion Functions (for MXFP8)
# E8M0 is a power-of-2 exponent format: value = 2^(exp - 127)
# =============================================================================

E8M0_BIAS = 127


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Convert E8M0 (uint8 exponent) to float32.

    E8M0 represents pure power-of-2 values: 2^(exp - 127).

    Args:
        x: uint8 tensor with E8M0 exponent values

    Returns:
        float32 tensor with decoded values
    """
    assert x.dtype == torch.uint8, "Input must be uint8"
    biased_exp = x.to(torch.int32)
    result = biased_exp << MBITS_F32
    # Handle zero exponent (represents zero)
    result = torch.where(biased_exp == 0, torch.zeros_like(result), result)
    return result.view(torch.float32)


def f32_to_e8m0(x: torch.Tensor) -> torch.Tensor:
    """Convert float32 to E8M0 (power-of-2 exponent).

    Rounds to nearest power of 2.

    Args:
        x: float32 tensor (must be positive)

    Returns:
        uint8 tensor with E8M0 exponent values
    """
    assert x.dtype == torch.float32, "Input must be float32"
    x_int = x.view(torch.int32)
    biased_exp = (x_int >> MBITS_F32) & 0xFF

    # Get mantissa for rounding decision (round to nearest power of 2)
    mantissa = x_int & _n_ones(MBITS_F32)
    round_up = mantissa >= (1 << (MBITS_F32 - 1))
    biased_exp = biased_exp + round_up.to(torch.int32)

    biased_exp = torch.clamp(biased_exp, 0, 255)
    return biased_exp.to(torch.uint8)


# =============================================================================
# MXFP8 Blocked Layout Functions
# MXFP8 uses same cuBLAS tiled layout as NVFP4, but different block size
# =============================================================================

def mxfp8_to_blocked(input_matrix: torch.Tensor, flatten: bool = True) -> torch.Tensor:
    """
    Rearrange E8M0 block scales to cuBLAS tiled layout for MXFP8.

    Uses the same layout transformation as NVFP4 (to_blocked).

    Args:
        input_matrix: Input tensor of shape (num_rows, num_blocks)
        flatten: If True, return flattened tensor

    Returns:
        Rearranged tensor for cuBLAS block layout
    """
    return to_blocked(input_matrix, flatten=flatten)


def mxfp8_from_blocked(blocked_matrix: torch.Tensor, num_rows: int, num_cols: int) -> torch.Tensor:
    """
    Reverse cuBLAS tiled layout back to normal (H, num_blocks) layout for MXFP8.

    Uses the same reverse transformation as NVFP4 (from_blocked).

    Args:
        blocked_matrix: Swizzled tensor from cuBLAS layout
        num_rows: Desired output rows (unpadded)
        num_cols: Desired output num_blocks (unpadded)

    Returns:
        Unswizzled tensor of shape (num_rows, num_cols)
    """
    return from_blocked(blocked_matrix, num_rows, num_cols)

