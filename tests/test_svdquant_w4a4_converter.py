# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for SVDQuant W4A4 converter.

Covers:
- Nibble pack/unpack round-trip and convention (even col -> low nibble)
- Quantizer clamp: signed emission is exactly [-7, 7], never -8
- Smooth modes: ones produces no-op, weight_only produces channel-scaled result
- SVD correction: dequantized reconstruction error is lower with rank>0
- Output shapes: all five tensors match kitchen's state_dict_tensors layout
- wscales dtype: must stay in compute_dtype (bf16/fp16), not float32
- Rank clamping: rank > min(N, K) is handled without error
- rank=0: disables LoRA correction and produces zero-column proj tensors
- External smooth: validates shape mismatch raises
- K % 64 != 0: raises ValueError
"""
from __future__ import annotations

import pytest
import torch

from convert_to_quant.converters.svdquant_w4a4_converter import (
    SVDQuantW4A4Converter,
    _pack_int4_weight,
    _unpack_int4_weight,
    _INT4_GROUP_SIZE,
    _INT4_QMAX,
)


# ---------------------------------------------------------------------------
# Nibble packing (must match svdquant_utils.cuh::pack_int4_pair)
# ---------------------------------------------------------------------------

class TestNibblePacking:
    def test_even_col_to_low_nibble(self):
        # q[n, 2k]   = byte & 0x0F  (low  nibble)
        # q[n, 2k+1] = byte >> 4    (high nibble)
        q = torch.tensor([[3, 5]], dtype=torch.int8)   # col0=3, col1=5
        packed = _pack_int4_weight(q)
        byte = packed[0, 0].item() & 0xFF
        lo = byte & 0x0F
        hi = (byte >> 4) & 0x0F
        assert lo == 3, "even column must be stored in the low nibble"
        assert hi == 5, "odd column must be stored in the high nibble"

    def test_round_trip(self):
        torch.manual_seed(42)
        N, K = 8, 128
        q_orig = torch.randint(-7, 8, (N, K), dtype=torch.int8)
        packed = _pack_int4_weight(q_orig)
        unpacked = _unpack_int4_weight(packed, K)
        assert torch.equal(unpacked, q_orig)

    def test_known_bytes(self):
        # col0=2, col1=7 -> lo=2, hi=7 -> byte = 0x72
        q = torch.tensor([[2, 7, -1, -7]], dtype=torch.int8)
        packed = _pack_int4_weight(q)
        b0 = packed[0, 0].item() & 0xFF
        b1 = packed[0, 1].item() & 0xFF
        assert b0 == ((7 << 4) | 2), f"byte0 should be 0x72, got {hex(b0)}"
        # col2=-1 -> 0b1111=0xF, col3=-7 -> 0b1001=0x9 -> byte = 0x9F
        assert b1 == ((0x9 << 4) | 0xF), f"byte1 should be 0x9F, got {hex(b1)}"

    def test_neg8_never_packed(self):
        # -8 should never appear since the quantizer clamps to [-7, 7],
        # but we verify unpack handles the bit pattern correctly for completeness.
        q = torch.tensor([[-8, 0]], dtype=torch.int8)
        packed = _pack_int4_weight(q)
        unpacked = _unpack_int4_weight(packed, 2)
        # -8 in 4 bits is 0b1000; after sign-extension: 8 - 16 = -8
        assert unpacked[0, 0].item() == -8


# ---------------------------------------------------------------------------
# Converter output shapes and dtypes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    @pytest.mark.parametrize("N,K,R", [
        (64, 128, 16),
        (128, 256, 32),
        (512, 512, 64),
    ])
    def test_shapes_and_dtypes(self, N, K, R):
        W = torch.randn(N, K)
        conv = SVDQuantW4A4Converter(rank=R, compute_dtype=torch.bfloat16)
        qweight, wscales, proj_down, proj_up, smooth_factor, dequant = conv.convert(W)

        assert qweight.shape == (N, K // 2), f"qweight shape {qweight.shape}"
        assert qweight.dtype == torch.int8
        assert wscales.shape == (K // _INT4_GROUP_SIZE, N), f"wscales shape {wscales.shape}"
        assert wscales.dtype == torch.bfloat16, "wscales must stay in compute_dtype"
        assert proj_down.shape == (K, min(R, min(N, K))), f"proj_down shape {proj_down.shape}"
        assert proj_up.shape == (N, min(R, min(N, K))), f"proj_up shape {proj_up.shape}"
        assert smooth_factor.shape == (K,), f"smooth_factor shape {smooth_factor.shape}"
        assert smooth_factor.dtype == torch.bfloat16
        assert dequant.shape == (N, K), f"dequant shape {dequant.shape}"
        assert dequant.dtype == torch.bfloat16

    def test_wscales_dtype_fp16(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=8, compute_dtype=torch.float16)
        _, wscales, _, _, _, _ = conv.convert(W)
        assert wscales.dtype == torch.float16

    def test_rank_zero_zero_columns(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=0)
        qweight, wscales, proj_down, proj_up, smooth_factor, _ = conv.convert(W)
        assert proj_down.shape[1] == 0
        assert proj_up.shape[1] == 0
        assert qweight.shape == (64, 64)

    def test_rank_clamped_to_min_nk(self):
        # 4x256: min(N,K) = 4, rank=32 should be clamped
        W = torch.randn(4, 256)
        conv = SVDQuantW4A4Converter(rank=32)
        _, _, proj_down, proj_up, _, _ = conv.convert(W)
        assert proj_down.shape[1] == 4
        assert proj_up.shape[1] == 4


# ---------------------------------------------------------------------------
# Quantizer contract
# ---------------------------------------------------------------------------

class TestQuantizerContract:
    def test_signed_clamp_never_emits_neg8(self):
        """Signed quantizer must emit values in [-7, 7], never -8."""
        torch.manual_seed(0)
        W = torch.randn(64, 256) * 50.0   # large values to force saturation
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="ones")
        qweight, wscales, _, _, _, _ = conv.convert(W)
        unpacked = _unpack_int4_weight(qweight, 256)
        assert unpacked.min().item() >= -_INT4_QMAX, "quantizer emitted -8"
        assert unpacked.max().item() <= _INT4_QMAX

    def test_signed_clamp_forced_outlier(self):
        W = torch.zeros(64, 256)
        W[0, 0] = -1000.0
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="ones")
        qweight, _, _, _, _, _ = conv.convert(W)
        unpacked = _unpack_int4_weight(qweight, 256)
        assert (unpacked != -8).all()
        assert unpacked.min().item() == -_INT4_QMAX

    def test_wscales_nonnegative(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="ones")
        _, wscales, _, _, _, _ = conv.convert(W)
        assert (wscales >= 0).all()


# ---------------------------------------------------------------------------
# Smooth modes
# ---------------------------------------------------------------------------

class TestSmoothModes:
    def test_ones_smooth_is_identity(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="ones")
        _, _, _, _, smooth_factor, _ = conv.convert(W)
        assert torch.allclose(
            smooth_factor.float(), torch.ones(128), atol=1e-4
        ), "smooth_mode='ones' must produce all-ones smooth factor"

    def test_weight_only_smooth_positive(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="weight_only")
        _, _, _, _, smooth_factor, _ = conv.convert(W)
        assert (smooth_factor > 0).all()

    def test_weight_only_smooth_alpha_zero_is_ones(self):
        # alpha=0 -> col_absmax^0 = 1 for any nonzero col
        W = torch.randn(64, 128).abs() + 0.1
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="weight_only", smooth_alpha=0.0)
        _, _, _, _, smooth_factor, _ = conv.convert(W)
        assert torch.allclose(
            smooth_factor.float(), torch.ones(128), atol=1e-4
        )

    def test_external_smooth_used(self):
        W = torch.randn(64, 128)
        ext = torch.full((128,), 2.0)
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="external")
        _, _, _, _, smooth_factor, _ = conv.convert(W, smooth=ext)
        assert torch.allclose(smooth_factor.float(), ext.float(), atol=1e-4)

    def test_external_smooth_shape_mismatch_raises(self):
        W = torch.randn(64, 128)
        bad_smooth = torch.ones(64)   # wrong K
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="external")
        with pytest.raises(ValueError, match="shape"):
            conv.convert(W, smooth=bad_smooth)

    def test_external_smooth_missing_raises(self):
        W = torch.randn(64, 128)
        conv = SVDQuantW4A4Converter(rank=0, smooth_mode="external")
        with pytest.raises(ValueError, match="requires a smooth tensor"):
            conv.convert(W)


# ---------------------------------------------------------------------------
# Reconstruction quality (SVD correction reduces error)
# ---------------------------------------------------------------------------

class TestReconstructionQuality:
    def test_rank0_vs_rank16_reconstruction_error(self):
        """rank=16 should produce lower reconstruction error than rank=0."""
        torch.manual_seed(7)
        W = torch.randn(128, 256).float()
        conv_r0 = SVDQuantW4A4Converter(rank=0, smooth_mode="ones")
        conv_r16 = SVDQuantW4A4Converter(rank=16, smooth_mode="ones")

        _, _, _, _, _, dq_r0 = conv_r0.convert(W.clone())
        _, _, _, _, _, dq_r16 = conv_r16.convert(W.clone())

        err_r0 = (W - dq_r0.float()).norm().item()
        err_r16 = (W - dq_r16.float()).norm().item()
        assert err_r16 < err_r0, (
            f"rank=16 reconstruction error ({err_r16:.4f}) should be less than "
            f"rank=0 ({err_r0:.4f})"
        )


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

class TestConstraints:
    def test_k_not_divisible_raises(self):
        W = torch.randn(64, 100)   # 100 % 64 != 0
        conv = SVDQuantW4A4Converter()
        with pytest.raises(ValueError, match="not divisible"):
            conv.convert(W)

    def test_invalid_smooth_mode_raises(self):
        with pytest.raises(ValueError, match="smooth_mode"):
            SVDQuantW4A4Converter(smooth_mode="calibrated")

    def test_invalid_group_size_raises(self):
        with pytest.raises(ValueError, match="group_size"):
            SVDQuantW4A4Converter(group_size=128)

    def test_float32_compute_dtype_raises(self):
        with pytest.raises(ValueError, match="compute_dtype"):
            SVDQuantW4A4Converter(compute_dtype=torch.float32)

    def test_square_small_layer(self):
        # Tiny layer that happens to be K-divisible
        W = torch.randn(8, 64)
        conv = SVDQuantW4A4Converter(rank=4)
        qweight, wscales, proj_down, proj_up, smooth_factor, dequant = conv.convert(W)
        assert qweight.shape == (8, 32)
        assert wscales.shape == (1, 8)
