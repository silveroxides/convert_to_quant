import torch
import pytest

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.utils.convrot import unpack_int4_row_major


def test_int4_converter_construction_forces_convrot_and_row_scaling():
    converter = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=4, no_learned_rounding=True, device="cpu"
    )
    assert converter.target_format == "int4"
    assert converter.convrot is True
    assert converter.scaling_mode == "row"


def test_int4_simple_rtn_roundtrip_shapes():
    torch.manual_seed(0)
    converter = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=4, no_learned_rounding=True, device="cpu"
    )
    W = torch.randn(8, 16)
    packed, scale, dequant, extra = converter.convert(W, key="test.weight", has_bias=False)

    assert packed.dtype == torch.int8
    assert packed.shape == (8, 8)  # in_features // 2
    assert scale.shape == (8,)
    assert dequant.shape == (8, 16)
    assert isinstance(extra, dict)

    unpacked = unpack_int4_row_major(packed)
    assert unpacked.min() >= -7 and unpacked.max() <= 7


def test_int4_adaround_reduces_reconstruction_error_vs_rtn():
    torch.manual_seed(0)
    W = torch.randn(16, 64) * 3.0
    calib = torch.randn(64, 64)

    rtn = LearnedRoundingConverter(target_format="int4", convrot_group_size=64, no_learned_rounding=True, device="cpu")
    _, _, dequant_rtn, _ = rtn.convert(W.clone(), key="w", calibration_data=calib.clone(), has_bias=False)

    adaround = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=64, no_learned_rounding=False, num_iter=25, device="cpu",
        optimizer="adamw"
    )
    _, _, dequant_ada, _ = adaround.convert(W.clone(), key="w", calibration_data=calib.clone(), has_bias=False)

    err_rtn = (calib @ dequant_rtn.T - calib @ W.T).pow(2).mean()
    err_ada = (calib @ dequant_ada.T - calib @ W.T).pow(2).mean()
    assert err_ada <= err_rtn


def test_int4_requires_group_size_divisibility():
    converter = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=256, no_learned_rounding=True, device="cpu"
    )
    W = torch.randn(4, 100)  # 100 not divisible by 256
    with pytest.raises(ValueError):
        converter.convert(W, key="test.weight", has_bias=False)


def test_int4_all_zero_tensor():
    converter = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=4, no_learned_rounding=True, device="cpu"
    )
    W = torch.zeros(4, 16)
    packed, scale, dequant, extra = converter.convert(W, key="test.weight", has_bias=False)
    assert packed.shape == (4, 8)
    assert torch.all(packed == 0)
    assert scale.shape == (4,)
    assert torch.all(dequant == 0)
