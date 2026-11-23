"""
Tests for universal row-chunked streaming quantization across converters.
Verifies numerical parity between single-shot and chunked quantization.
"""

import torch

from convert_to_quant.converters.int4_converter import INT4Converter
from convert_to_quant.converters.learned_mxfp8 import LearnedMXFP8Converter
from convert_to_quant.converters.learned_nvfp4 import LearnedNVFP4Converter
from convert_to_quant.converters.w6a8_converter import W6A8Converter


def test_w6a8_row_chunked_parity():
    torch.manual_seed(1234)
    rows, cols = 128, 128
    W = torch.randn(rows, cols)

    # Full budget (single chunk)
    conv_full = W6A8Converter(no_learned_rounding=True, chunk_budget=1 << 28)
    p_full, s_rel_full, s_ch_full, dq_full, _ = conv_full.convert(W)

    # Small budget (forcing 32-row chunks)
    conv_chunked = W6A8Converter(no_learned_rounding=True, chunk_budget=32 * cols)
    p_chunked, s_rel_chunked, s_ch_chunked, dq_chunked, _ = conv_chunked.convert(W)

    assert torch.equal(p_full, p_chunked)
    assert torch.equal(s_rel_full.view(torch.uint8), s_rel_chunked.view(torch.uint8))
    assert torch.equal(s_ch_full, s_ch_chunked)
    assert torch.allclose(dq_full, dq_chunked, atol=1e-5)


def test_int4_row_chunked_parity():
    torch.manual_seed(1234)
    rows, cols = 128, 128
    W = torch.randn(rows, cols)

    conv_full = INT4Converter(no_learned_rounding=True, chunk_budget=1 << 28)
    p_full, s_full, dq_full, _ = conv_full.convert(W)

    conv_chunked = INT4Converter(no_learned_rounding=True, chunk_budget=32 * cols)
    p_chunked, s_chunked, dq_chunked, _ = conv_chunked.convert(W)

    assert torch.equal(p_full, p_chunked)
    assert torch.equal(s_full, s_chunked)
    assert torch.allclose(dq_full, dq_chunked, atol=1e-5)


def test_nvfp4_row_chunked_parity():
    torch.manual_seed(1234)
    rows, cols = 128, 128
    W = torch.randn(rows, cols)

    conv_full = LearnedNVFP4Converter(no_learned_rounding=True, chunk_budget=1 << 28)
    p_full, bs_full, pts_full, dq_full, _ = conv_full.convert(W)

    conv_chunked = LearnedNVFP4Converter(no_learned_rounding=True, chunk_budget=32 * cols)
    p_chunked, bs_chunked, pts_chunked, dq_chunked, _ = conv_chunked.convert(W)

    assert torch.equal(p_full, p_chunked)
    assert torch.equal(bs_full, bs_chunked)
    assert torch.equal(pts_full, pts_chunked)
    assert torch.equal(dq_full, dq_chunked)


def test_mxfp8_row_chunked_parity():
    torch.manual_seed(1234)
    rows, cols = 128, 128
    W = torch.randn(rows, cols)

    conv_full = LearnedMXFP8Converter(no_learned_rounding=True, chunk_budget=1 << 28)
    p_full, bs_full, dq_full, _ = conv_full.convert(W)

    conv_chunked = LearnedMXFP8Converter(no_learned_rounding=True, chunk_budget=32 * cols)
    p_chunked, bs_chunked, dq_chunked, _ = conv_chunked.convert(W)

    assert torch.equal(p_full.view(torch.uint8), p_chunked.view(torch.uint8))
    assert torch.equal(bs_full, bs_chunked)
    assert torch.equal(dq_full.view(torch.uint8), dq_chunked.view(torch.uint8))
