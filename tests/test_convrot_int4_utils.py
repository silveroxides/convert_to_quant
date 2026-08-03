import torch
import pytest

from convert_to_quant.utils.convrot import (
    quantize_int4_rowwise,
    pack_int4_row_major,
    unpack_int4_row_major,
    dequantize_int4_rowwise,
)


def test_quantize_int4_rowwise_range_and_scale():
    x = torch.tensor([[1.0, -2.0, 4.0, -8.0], [0.5, 0.5, 0.5, 0.5]])
    qdata, scale = quantize_int4_rowwise(x)
    assert qdata.dtype == torch.int8
    assert qdata.shape == x.shape
    assert scale.shape == (2,)
    assert qdata.min() >= -7 and qdata.max() <= 7
    # Row 0: absmax=8.0 -> scale=8/7
    assert torch.allclose(scale[0], torch.tensor(8.0 / 7.0), atol=1e-6)
    # Row 1: absmax=0.5 -> scale=0.5/7
    assert torch.allclose(scale[1], torch.tensor(0.5 / 7.0), atol=1e-6)


def test_quantize_int4_rowwise_zero_row_no_nan():
    x = torch.zeros(2, 8)
    qdata, scale = quantize_int4_rowwise(x)
    assert not torch.isnan(scale).any()
    assert torch.all(qdata == 0)


def test_pack_unpack_int4_roundtrip():
    torch.manual_seed(0)
    qdata = torch.randint(-7, 8, (4, 16), dtype=torch.int8)
    packed = pack_int4_row_major(qdata)
    assert packed.dtype == torch.int8
    assert packed.shape == (4, 8)
    unpacked = unpack_int4_row_major(packed)
    assert torch.equal(unpacked, qdata)


def test_pack_int4_odd_columns_raises():
    qdata = torch.randint(-7, 8, (4, 15), dtype=torch.int8)
    with pytest.raises(ValueError):
        pack_int4_row_major(qdata)


def test_dequantize_int4_rowwise_matches_quantize():
    torch.manual_seed(1)
    x = torch.randn(6, 32) * 3.0
    qdata, scale = quantize_int4_rowwise(x)
    packed = pack_int4_row_major(qdata)
    dequant = dequantize_int4_rowwise(packed, scale)
    assert dequant.dtype == torch.float32
    assert dequant.shape == x.shape
    # Quantization error should be small relative to the dynamic range per row
    row_max = x.abs().amax(dim=-1)
    err = (dequant - x).abs().amax(dim=-1)
    assert torch.all(err <= row_max / 7.0 + 1e-4)
