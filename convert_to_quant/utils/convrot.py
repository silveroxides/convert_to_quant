"""Group-wise Hadamard rotation for INT8 quantization quality improvement.

Originally from: https://github.com/newgrit1004/ComfyUI-ZImage-Triton
License: MIT

Spreads activation outliers across channels using orthogonal Hadamard matrices.
Based on QuaRot (2024) and ConvRot (2025) approaches, adapted for DiT models
with group-wise rotation to avoid row-wise outlier amplification.
"""

import torch
from scipy.linalg import hadamard as scipy_hadamard

# Cache Hadamard matrices by (size, device, dtype) to avoid recomputation
_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def build_hadamard(
    size: int,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized REGULAR orthogonal Hadamard matrix (ConvRot).

    Size must be a power of 4 (e.g., 4, 16, 64, 256, 1024...).
    Uses the Kronecker construction from Theorem 3.3 to avoid the all-1s
    column of standard Sylvester Hadamard matrices, which amplifies
    row-wise outliers in diffusion models.
    """
    import math
    cache_key = (size, str(device), dtype)
    if cache_key in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[cache_key]

    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {size}")

    # Standard Sylvester Hadamard fallback for non-power-of-4 sizes (e.g. 512)
    is_power_of_4 = (math.log(size, 4) % 1 == 0)
    if not is_power_of_4:
        H_np = scipy_hadamard(size)
        H_normalized = torch.from_numpy(H_np).to(device=device, dtype=dtype) / (size**0.5)
        _HADAMARD_CACHE[cache_key] = H_normalized
        return H_normalized

    # Base H4 from Theorem 3.3 (Eq 9 in the paper)
    # Notice how every row and column sums to exactly 2
    H4 = torch.tensor([[ 1,  1,  1, -1],
        [ 1,  1, -1,  1],[ 1, -1,  1,  1],[-1,  1,  1,  1]
    ], dtype=dtype, device=device)

    H = H4
    current_size = 4

    # Kronecker construction for larger sizes: H_{4^{k+1}} = H_{4^k} \otimes H_4
    while current_size < size:
        H = torch.kron(H, H4)
        current_size *= 4

    # Normalize to make it orthogonal
    H_normalized = H / (size**0.5)
    _HADAMARD_CACHE[cache_key] = H_normalized

    return H_normalized

def rotate_weight(
    weight: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate weight matrix offline: W_rot = W @ H_block^T.

    For Linear(in, out) with weight shape (out, in):
    Each row of W is split into groups of group_size and rotated by H^T.

    Args:
        weight: Shape (out_features, in_features).
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated weight, same shape as input.
    """
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    n_groups = in_f // group_size

    # (out, in) → (out, n_groups, group_size)
    W_grouped = weight.view(out_f, n_groups, group_size)
    # Apply H^T to each group: (..., group_size) @ (group_size, group_size)
    H_t = H.T.to(dtype=weight.dtype, device=weight.device)
    W_rot = torch.matmul(W_grouped, H_t)
    return W_rot.reshape(out_f, in_f)


def rotate_activation(
    x: torch.Tensor,
    H: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Rotate activation online: x_rot = x @ H_block.

    Group-wise Hadamard spreads outliers across channels within each group.

    Args:
        x: Shape (..., features). Last dim must be divisible by group_size.
        H: Normalized Hadamard matrix, shape (group_size, group_size).
        group_size: Group size for block-diagonal rotation.

    Returns:
        Rotated activation, same shape as input.
    """
    orig_shape = x.shape
    features = orig_shape[-1]
    if features % group_size != 0:
        raise ValueError(
            f"features {features} not divisible by group_size {group_size}"
        )
    n_groups = features // group_size

    # (..., features) → (..., n_groups, group_size)
    x_grouped = x.view(*orig_shape[:-1], n_groups, group_size)
    H_dev = H.to(dtype=x.dtype, device=x.device)
    x_rot = torch.matmul(x_grouped, H_dev)
    return x_rot.view(orig_shape)


def resolve_int8_convrot_group_size(
    in_features: int,
    requested_group_size: int,
) -> int | None:
    """Resolve the static ConvRot group for a primary row-wise INT8 layer."""
    if in_features % requested_group_size == 0:
        return requested_group_size
    if requested_group_size == 256 and in_features % 64 == 0:
        return 64
    return None


def find_max_compatible_group_size(in_features: int, min_group_size: int = 256) -> int | None:
    """Find the largest power of 4 group size >= min_group_size that divides in_features."""
    if in_features < min_group_size:
        return None

    import math
    max_k = int(math.log(in_features, 4))
    group_size = 4 ** max_k

    while group_size >= min_group_size:
        if in_features % group_size == 0:
            return group_size
        group_size //= 4
    return None


def quantize_int4_rowwise(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Row-wise symmetric INT4 quantization: scale = absmax / 7, range [-7, 7].

    Matches comfy-kitchen's ``quantize_signed_int4_rowwise`` formula exactly
    so packed output round-trips against comfy-kitchen's kernels.

    Args:
        x: Shape (rows, cols), any float dtype.

    Returns:
        Tuple of (qdata, scale):
        - qdata: torch.int8, same shape as x, values in [-7, 7] (unpacked, one byte per value).
        - scale: torch.float32, shape (rows,).
    """
    absmax = x.abs().amax(dim=-1).clamp(min=1e-10)
    scale = absmax / 7.0
    qdata = (x / scale.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)
    return qdata, scale.to(torch.float32)


def pack_int4_row_major(qdata: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values (stored as int8 in [-7, 7]) 2-per-byte, row-major.

    Even column index -> low nibble, odd column index -> high nibble.

    Args:
        qdata: torch.int8, shape (rows, cols), cols must be even.

    Returns:
        torch.int8, shape (rows, cols // 2).
    """
    if qdata.shape[-1] % 2 != 0:
        raise ValueError(f"pack_int4_row_major requires an even number of columns, got {qdata.shape[-1]}")
    low = qdata[..., 0::2].to(torch.uint8) & 0x0F
    high = qdata[..., 1::2].to(torch.uint8) & 0x0F
    packed = (low | (high << 4)).to(torch.int8)
    return packed


def unpack_int4_row_major(qdata_packed: torch.Tensor) -> torch.Tensor:
    """Inverse of pack_int4_row_major.

    Args:
        qdata_packed: torch.int8, shape (rows, packed_cols).

    Returns:
        torch.int8, shape (rows, packed_cols * 2), values in [-7, 7].
    """
    packed_u8 = qdata_packed.to(torch.uint8)
    low = packed_u8 & 0x0F
    high = (packed_u8 >> 4) & 0x0F

    def _to_signed_nibble(v: torch.Tensor) -> torch.Tensor:
        # 4-bit two's complement: values 8-15 represent -8..-1
        return torch.where(v >= 8, v.to(torch.int16) - 16, v.to(torch.int16)).to(torch.int8)

    low_signed = _to_signed_nibble(low)
    high_signed = _to_signed_nibble(high)

    rows, packed_cols = qdata_packed.shape
    out = torch.empty(rows, packed_cols * 2, dtype=torch.int8, device=qdata_packed.device)
    out[..., 0::2] = low_signed
    out[..., 1::2] = high_signed
    return out


def dequantize_int4_rowwise(qdata_packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Unpack and dequantize row-wise INT4 data.

    Args:
        qdata_packed: torch.int8, shape (rows, packed_cols).
        scale: torch.float32, shape (rows,).

    Returns:
        torch.float32, shape (rows, packed_cols * 2).
    """
    qdata = unpack_int4_row_major(qdata_packed)
    return qdata.to(torch.float32) * scale.to(torch.float32).unsqueeze(-1)
