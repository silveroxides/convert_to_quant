import pytest
import torch

from convert_to_quant.utils.convrot import (
    dequantize_int4_rowwise,
    pack_int4_row_major,
    quantize_int4_rowwise,
    unpack_int4_row_major,
)


def test_quantize_int4_rowwise_range_and_scale():
    x = torch.tensor([[1.0, -2.0, 4.0, -8.0], [0.5, 0.5, 0.5, 0.5]])
    qdata, scale = quantize_int4_rowwise(x)
    assert qdata.dtype == torch.int8
    assert qdata.shape == x.shape
    assert qdata.min().item() >= -7
    assert qdata.max().item() <= 7
    # Scale should be max(abs(row)) / 7.0
    expected_scale_0 = 8.0 / 7.0
    expected_scale_1 = 0.5 / 7.0
    assert torch.isclose(scale[0], torch.tensor(expected_scale_0), atol=1e-5)
    assert torch.isclose(scale[1], torch.tensor(expected_scale_1), atol=1e-5)


def test_pack_unpack_int4_roundtrip():
    # Generate arbitrary valid signed int4 values [-7, 7]
    torch.manual_seed(42)
    original_q = torch.randint(-7, 8, (10, 64), dtype=torch.int8)
    packed = pack_int4_row_major(original_q)
    assert packed.shape == (10, 32)
    assert packed.dtype == torch.int8

    unpacked = unpack_int4_row_major(packed)
    assert unpacked.shape == original_q.shape
    assert unpacked.dtype == torch.int8
    assert torch.equal(original_q, unpacked)


def test_pack_int4_row_major_matches_nibble_order():
    # Low nibble = even col, High nibble = odd col
    # col 0 = 3 (0x03), col 1 = -2 (4-bit two's complement is 0x0E -> 14)
    # Byte: 0x03 | (0x0E << 4) = 0xE3 = 227 unsigned -> -29 signed int8
    row = torch.tensor([[3, -2]], dtype=torch.int8)
    packed = pack_int4_row_major(row)
    assert packed.item() == -29

    unpacked = unpack_int4_row_major(packed)
    assert torch.equal(unpacked, row)


def test_dequantize_int4_rowwise():
    x = torch.randn(6, 32) * 3.0
    qdata, scale = quantize_int4_rowwise(x)
    packed = pack_int4_row_major(qdata)
    dequant = dequantize_int4_rowwise(packed, scale)
    assert dequant.dtype == torch.float32
    assert dequant.shape == x.shape

    row_max = x.abs().amax(dim=-1)
    err = (dequant - x).abs().amax(dim=-1)
    assert torch.all(err <= row_max / 7.0 + 1e-4)


def test_odd_columns_rejected():
    odd_q = torch.zeros((2, 5), dtype=torch.int8)
    with pytest.raises(ValueError, match="even number of columns"):
        pack_int4_row_major(odd_q)


def test_parity_with_comfy_kitchen_eager_convrot():
    import sys
    from pathlib import Path

    kitchen_path = Path(__file__).resolve().parent.parent / "local_reference" / "comfy-kitchen"
    if not (kitchen_path / "comfy_kitchen").exists():
        pytest.skip("local_reference/comfy-kitchen not found")

    if str(kitchen_path) not in sys.path:
        sys.path.insert(0, str(kitchen_path))

    try:
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            dequantize_convrot_w4a4_weight as ck_dequantize,
            quantize_convrot_w4a4_weight as ck_quantize,
        )
    except ImportError:
        pytest.skip("comfy_kitchen eager backend not importable")

    torch.manual_seed(100)
    W = torch.randn(16, 64, dtype=torch.float32)

    # Comfy kitchen reference
    ck_packed, ck_scale = ck_quantize(W, convrot_groupsize=64, quant_group_size=64, stochastic_rounding=0)
    ck_dq = ck_dequantize(ck_packed, ck_scale, convrot_groupsize=64, quant_group_size=64, output_dtype=torch.float32)

    # Our pure PyTorch INT4Converter
    from convert_to_quant.converters.int4_converter import INT4Converter
    converter = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        device="cpu",
        no_learned_rounding=True,
    )
    our_packed, our_scale, our_dq, _ = converter.convert(W)

    assert torch.equal(our_packed, ck_packed)
    assert torch.allclose(our_scale, ck_scale, atol=1e-5)
    assert torch.allclose(our_dq, ck_dq, atol=1e-5)
