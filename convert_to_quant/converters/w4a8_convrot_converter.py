"""
W4A8-ConvRot (asym_w4a8_int8) Quantization Converter.

Implements the ComfyUI/comfy-kitchen "asym_w4a8_int8" format:
4-bit codebook weights + 8-bit dynamically-quantized activations with
offline group-wise Hadamard rotation (ConvRot, group 256).

Serialization contract (see search-results/w4a8-convrot-contract-20261108.md):
- format string: "asym_w4a8_int8" (ComfyUI comfy/ops.py _GROUPED_INT8_FORMATS)
- .comfy_quant JSON: {"format": "asym_w4a8_int8", "group_size": 16,
                       "convrot": true, "convrot_groupsize": 256}
- Per-layer tensors:
    weight            INT8 [N, K/2]  (even col = LOW nibble)
    weight_s_rel      F8_E4M3 [N, K/16]
    weight_s_channel  F32 [N]
    weight_codebook   F32 [16]  (frozen Lloyd-Max LUT)
    comfy_quant       U8 JSON bytes

Weight math: W_rot = W @ H^T per 256-column block, then per-16-group
symmetric codebook quantization with 2-pass ALS scale refinement and
final re-assignment through the decoded INT8 grid.

Activation side (online x @ H rotation + dynamic row-wise INT8) is handled
entirely at runtime by ComfyUI; no activation calibration is needed here.

Learned rounding is out of scope for this converter (RTN + ALS refinement only).
"""

from typing import Dict, Optional, Tuple

import torch

from ..utils.convrot import build_hadamard, rotate_weight
from ..utils.logging import verbose

# Frozen Lloyd-Max LUT for Gaussian-ish rotated weights
# (comfy_kitchen/tensor/w4a8_packed.py::_FIXED_LUT, verbatim).
W4A8_FIXED_LUT = (
    -0.980602, -0.794529, -0.638165, -0.500986, -0.377321, -0.263187, -0.155210, -0.050720,
    0.052541, 0.156985, 0.265284, 0.379533, 0.502636, 0.638953, 0.794876, 0.980671,
)

W4A8_FORMAT = "asym_w4a8_int8"
W4A8_GROUP_SIZE = 16  # codebook group size (fixed by the format)
W4A8_INT8_GRID_MAX = 127.0
_ALS_ITERS = 2


def pack_w4a8_codes(codes: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit codes [N, K] (uint8) into int8 [N, K/2].

    Contract packing order (comfy_kitchen _pack_codes):
    EVEN column = LOW nibble, ODD column = HIGH nibble:
        byte = (u[0::2] & 0xF) | ((u[1::2] & 0xF) << 4)
    """
    if codes.shape[-1] % 2 != 0:
        raise ValueError(f"Last dim must be even, got {codes.shape}")
    u = codes.to(torch.uint8).contiguous()
    packed = (u[..., 0::2] & 0xF) | ((u[..., 1::2] & 0xF) << 4)
    return packed.to(torch.int8).contiguous()


def unpack_w4a8_codes(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int8 [N, K/2] into 4-bit codes [N, K] (uint8).

    Even column = LOW nibble, odd column = HIGH nibble (inverse of
    pack_w4a8_codes, matches the comfy-kitchen decode contract).
    """
    u = packed.to(torch.uint8).contiguous()
    out = torch.empty(*u.shape[:-1], u.shape[-1] * 2, dtype=torch.uint8, device=u.device)
    out[..., 0::2] = u & 0xF
    out[..., 1::2] = (u >> 4) & 0xF
    return out


class W4A8ConvRotConverter:
    """W4A8-ConvRot (asym_w4a8_int8) quantization converter.

    Follows the comfy-kitchen eager reference implementation
    (_quantize_rotated_w4a8_int8_weight):

    1. Offline ConvRot rotation per convrot_groupsize-column block
       (W_rot = W @ H^T, regular Hadamard from build_hadamard).
    2. Per-16-group normalization (group_scale = amax clamped to 1e-8).
    3. Nearest-level codebook assignment against the frozen Lloyd-Max LUT.
    4. ALS scale refinement (2 alternating codebook/scale passes).
    5. Two-level scale factorization:
       s_channel = amax(|W_rot reconstructed post-ALS|) / 127 per output
       channel (F32 [N]); basing it on the reconstructed weight guarantees
       no clamping of the max decoded level on the INT8 grid.
       s_rel     = group_scale / s_channel cast to float8_e4m3fn ([N, K/16])
    6. Grid re-assignment: codes re-assigned so the decoded weight
       round(clamp(lut[code] * s_rel, -127, 127)) lands on the INT8 grid
       nearest to W_rot / s_channel.
    7. Nibble packing (even col = low nibble) into INT8 [N, K/2].

    The runtime dequantization is:
        int8_weight[n, k] = round(clamp(codebook[code] * s_rel[n, k // 16], -127, 127))
        effective weight  = int8_weight * s_channel[n]  (then fed through the
        shared INT8 linear with dynamically row-wise-quantized activations).
    """

    def __init__(
        self,
        group_size: int = W4A8_GROUP_SIZE,
        convrot_group_size: int = 256,
        als_iters: int = _ALS_ITERS,
        convrot: bool = False,  # accepted for CLI compat; rotation is inherent to the format
        dynamic_convrot: bool = False,  # accepted for CLI compat; ignored
        device: Optional[str] = None,
        **kwargs,
    ):
        if group_size != W4A8_GROUP_SIZE:
            raise ValueError(f"W4A8 requires group_size={W4A8_GROUP_SIZE}, got {group_size}")

        self.group_size = group_size
        self.block_size = group_size  # fp8_conversion reads converter.block_size
        # Validate convrot group size early (must be a power of 4 for the
        # regular Hadamard construction).
        build_hadamard(convrot_group_size)
        self.convrot_group_size = convrot_group_size
        self.als_iters = als_iters
        # Rotation is always applied by this format (mirrors the reference
        # checkpoint where convrot: true / convrot_groupsize: 256).
        self.convrot = True
        # Participates in the standard convrot-metadata detection in
        # fp8_conversion (which requires scaling_mode == "row").
        self.scaling_mode = "row"
        self.device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        self.codebook = torch.tensor(W4A8_FIXED_LUT, dtype=torch.float32)

        verbose(f"W4A8ConvRotConverter initialized on device: {self.device}")
        verbose(f"  - Format: {W4A8_FORMAT} (codebook group {self.group_size})")
        verbose(f"  - ConvRot group size: {self.convrot_group_size} (always on)")
        verbose(f"  - ALS refinement iters: {self.als_iters}")

    # ------------------------------------------------------------------
    # Quantization
    # ------------------------------------------------------------------

    @torch.no_grad()
    def convert(
        self, W_orig: torch.Tensor, key: Optional[str] = None, depth: int = -1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """Quantize a 2D weight tensor to W4A8-ConvRot format.

        Args:
            W_orig: Input tensor (2D, shape [N, K]). K must be divisible by
                both the codebook group size (16) and convrot_group_size.

        Returns:
            Tuple of:
                packed_weight: INT8 [N, K/2] nibble-packed codes
                weight_s_rel: float8_e4m3fn [N, K/16] per-group relative scales
                weight_s_channel: float32 [N] per-output-channel scales
                weight_codebook: float32 [16] frozen Lloyd-Max LUT
                dequantized_weight: float32 [N, K] effective (un-rotated) weight
                extra_tensors: dict (currently always empty; bias correction
                    is handled by the caller from dequantized_weight)
        """
        W = W_orig.to(device=self.device, dtype=torch.float32).contiguous()
        N, K = W.shape
        if K % self.group_size != 0:
            raise ValueError(f"K={K} not divisible by group_size={self.group_size}")
        if K % self.convrot_group_size != 0:
            raise ValueError(f"K={K} not divisible by convrot_group_size={self.convrot_group_size}")

        # 1. Offline ConvRot rotation: W_rot = W @ H_block^T
        H = build_hadamard(self.convrot_group_size, device=W.device, dtype=torch.float32)
        W_rot = rotate_weight(W, H, self.convrot_group_size)
        del W

        n_groups = K // self.group_size
        grouped = W_rot.view(N, n_groups, self.group_size)
        lut = self.codebook.to(device=grouped.device)
        lut_mids = (lut[:-1] + lut[1:]) / 2

        # 2. Group normalize + 3. initial nearest-level assignment
        group_scale = grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        codes = torch.searchsorted(lut_mids, grouped / group_scale)

        # 4. ALS scale refinement (alternating codebook/scale, contract _ALS_ITERS=2)
        for _ in range(self.als_iters):
            levels = lut[codes]
            denom = (levels * levels).sum(dim=-1, keepdim=True).clamp(min=1e-8)
            group_scale = ((grouped * levels).sum(dim=-1, keepdim=True) / denom).clamp(min=1e-8)
            codes = torch.searchsorted(lut_mids, grouped / group_scale)

        # 5. Two-level scale factorization. s_channel is based on the RECONSTRUCTED
        #    post-ALS weight (lut[codes] * group_scale, matching comfy-kitchen's
        #    shifted_weight), which guarantees the max decoded level does not clamp
        #    on the INT8 grid (the raw-rotated amax could exceed it).
        shifted_weight = lut[codes] * group_scale
        s_channel = (shifted_weight.abs().amax(dim=(1, 2)) / W4A8_INT8_GRID_MAX).clamp(min=1e-8)  # [N] F32
        s_rel_f32 = group_scale.squeeze(-1) / s_channel.unsqueeze(1)  # [N, K/gs]
        s_rel = s_rel_f32.to(torch.float8_e4m3fn).contiguous()
        s_rel_dec = s_rel.to(torch.float32)  # decoded (stored-precision) relative scale

        # 6. Grid re-assignment through the decoded INT8 grid:
        #    grid[c] = round(clamp(lut[c] * s_rel, -127, 127)); pick the code
        #    whose decoded level is nearest to W_rot / s_channel.
        grid = torch.round(
            torch.clamp(lut.view(1, 1, -1) * s_rel_dec.unsqueeze(-1), -W4A8_INT8_GRID_MAX, W4A8_INT8_GRID_MAX)
        )  # [N, n_groups, 16]
        grid_mids = (grid[:, :, :-1] + grid[:, :, 1:]) / 2
        target = grouped / s_channel.view(N, 1, 1)
        codes = torch.searchsorted(grid_mids.contiguous(), target.contiguous())

        # 7. Pack (even col = low nibble)
        packed = pack_w4a8_codes(codes.view(N, K))

        # Effective (un-rotated) dequantized weight for bias correction:
        # decode on the INT8 grid, then apply the inverse rotation
        # ((W @ H^T) @ H = W since H is orthogonal).
        dequantized = self._decode_rotated(packed, s_rel, s_channel).view(N, K // self.convrot_group_size, self.convrot_group_size)
        dequantized = torch.matmul(dequantized, H.to(dequantized.dtype)).view(N, K)

        codebook_out = self.codebook.to(device="cpu")
        s_channel_out = s_channel.to(device="cpu", dtype=torch.float32).contiguous()
        s_rel_out = s_rel.to(device="cpu")

        del W_rot, grouped, grid, grid_mids, target, codes, shifted_weight

        return packed, s_rel_out, s_channel_out, codebook_out, dequantized, {}

    # ------------------------------------------------------------------
    # Dequantization
    # ------------------------------------------------------------------

    def _decode_rotated(
        self, packed: torch.Tensor, s_rel: torch.Tensor, s_channel: torch.Tensor
    ) -> torch.Tensor:
        """Decode packed codes to the rotated weight (contract decode math).

        int8_grid[n, k] = round(clamp(codebook[code] * s_rel[n, k // 16], -127, 127))
        W_rot[n, k]     = int8_grid[n, k] * s_channel[n]
        """
        codes = unpack_w4a8_codes(packed)
        N, K = codes.shape
        lut = self.codebook.to(device=codes.device)
        levels = lut[codes.long()].view(N, K // self.group_size, self.group_size)
        s_rel_dec = s_rel.to(device=codes.device, dtype=torch.float32)
        int8_grid = torch.round(
            torch.clamp(levels * s_rel_dec.unsqueeze(-1), -W4A8_INT8_GRID_MAX, W4A8_INT8_GRID_MAX)
        )
        return (int8_grid * s_channel.to(device=codes.device, dtype=torch.float32).view(N, 1, 1)).view(N, K)

    def dequantize(
        self,
        packed: torch.Tensor,
        s_rel: torch.Tensor,
        s_channel: torch.Tensor,
        output_dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Dequantize W4A8 tensors back to the un-rotated weight space.

        Args:
            packed: INT8 [N, K/2] packed codes
            s_rel: float8_e4m3fn [N, K/16] relative scales
            s_channel: float32 [N] channel scales
            output_dtype: Target dtype

        Returns:
            Effective weight [N, K] (rotation undone), matching what the
            runtime computes as W @ H^T @ (decode) @ H.
        """
        packed = packed.to(device=self.device)
        s_rel = s_rel.to(device=self.device)
        s_channel = s_channel.to(device=self.device)
        K = packed.shape[1] * 2
        H = build_hadamard(self.convrot_group_size, device=packed.device, dtype=torch.float32)
        w_rot = self._decode_rotated(packed, s_rel, s_channel)
        N = w_rot.shape[0]
        w = torch.matmul(
            w_rot.view(N, K // self.convrot_group_size, self.convrot_group_size), H.to(w_rot.dtype)
        ).view(N, K)
        return w.to(output_dtype)


def quantize_w4a8_convrot(
    tensor: torch.Tensor, convrot_group_size: int = 256
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convenience function: quantize a tensor to W4A8-ConvRot.

    Returns:
        Tuple of (packed_weight, s_rel, s_channel, codebook, dequantized)
    """
    converter = W4A8ConvRotConverter(convrot_group_size=convrot_group_size)
    packed, s_rel, s_channel, codebook, dequantized, _ = converter.convert(tensor)
    return packed, s_rel, s_channel, codebook, dequantized


def dequantize_w4a8_convrot(
    packed: torch.Tensor,
    s_rel: torch.Tensor,
    s_channel: torch.Tensor,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convenience function: dequantize W4A8 tensors (un-rotates)."""
    converter = W4A8ConvRotConverter()
    return converter.dequantize(packed, s_rel, s_channel, output_dtype=output_dtype)
