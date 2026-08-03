import torch

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor
from convert_to_quant.utils.convrot import unpack_int4_row_major
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_int4_convrot_end_to_end_matches_metadata_contract():
    torch.manual_seed(42)
    out_features, in_features = 32, 256
    W = torch.randn(out_features, in_features)
    calib = torch.randn(64, in_features)

    converter = LearnedRoundingConverter(
        target_format="int4", convrot_group_size=256, no_learned_rounding=True, device="cpu"
    )
    packed, scale, dequant, extra = converter.convert(W, key="model.layer.weight", calibration_data=calib, has_bias=True)

    # Storage contract
    assert packed.dtype == torch.int8
    assert packed.shape == (out_features, in_features // 2)
    assert scale.shape == (out_features,)
    assert "bias_correction" in extra

    # Metadata contract
    meta = create_comfy_quant_tensor("int4_convrot_w4a4", convrot=True, convrot_groupsize=256)
    cfg = tensor_to_dict(meta)
    assert cfg == {"format": "int4_convrot_w4a4", "convrot": True, "convrot_groupsize": 256}

    # Reconstruction quality: relative error should be bounded (INT4 is lossy but not garbage)
    rel_err = (dequant - W).norm() / W.norm()
    assert rel_err < 0.5

    # Round-trip: unpacking + rescaling should match the returned dequant only after un-rotation,
    # so instead verify the packed integer values are within INT4 range.
    unpacked = unpack_int4_row_major(packed)
    assert unpacked.min() >= -7 and unpacked.max() <= 7


def test_int4_convrot_matches_comfy_kitchen_eager_when_available():
    """Cross-check our pure-PyTorch rotation+quantization against comfy-kitchen's
    eager reference implementation, if comfy-kitchen is installed. Skipped otherwise."""
    try:
        from comfy_kitchen.backends.eager.convrot_w4a4 import (
            quantize_convrot_w4a4_weight as ck_quantize,
        )
    except ImportError:
        import pytest
        pytest.skip("comfy-kitchen not installed")

    torch.manual_seed(7)
    from convert_to_quant.utils.convrot import build_hadamard, pack_int4_row_major, quantize_int4_rowwise, rotate_weight

    W = torch.randn(16, 64)
    group_size = 64

    H = build_hadamard(group_size, device="cpu", dtype=torch.float32)
    W_rot = rotate_weight(W, H, group_size)
    qdata, scale = quantize_int4_rowwise(W_rot)
    packed = pack_int4_row_major(qdata)

    ck_packed, ck_scale = ck_quantize(W, convrot_groupsize=group_size, quant_group_size=64, stochastic_rounding=0)

    assert torch.equal(packed, ck_packed)
    assert torch.allclose(scale, ck_scale, atol=1e-5)
