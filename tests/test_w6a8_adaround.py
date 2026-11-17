import pytest
import torch

from convert_to_quant.converters.w6a8_converter import W6A8Converter
from convert_to_quant.utils.convrot import unpack_int6_w6a8


def test_w6a8_adaround_reduces_reconstruction_error_vs_rtn():
    torch.manual_seed(42)
    out_features, in_features = 16, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32) * 0.5
    calib = torch.randn(64, in_features, dtype=torch.float32)

    rtn_conv = W6A8Converter(
        convrot=True,
        convrot_group_size=64,
        group_size=16,
        dynamic_convrot=False,
        no_learned_rounding=True,
        device="cpu",
    )
    _, _, _, dequant_rtn, _ = rtn_conv.convert(
        W.clone(), key="layer", calibration_data=calib.clone(), has_bias=False
    )

    ada_conv = W6A8Converter(
        convrot=True,
        convrot_group_size=64,
        group_size=16,
        dynamic_convrot=False,
        no_learned_rounding=False,
        optimizer="adamw",
        num_iter=35,
        lr=1.0,
        device="cpu",
    )
    packed_ada, s_rel_ada, s_ch_ada, dequant_ada, _ = ada_conv.convert(
        W.clone(), key="layer", calibration_data=calib.clone(), has_bias=False
    )

    err_rtn = torch.nn.functional.mse_loss(calib @ dequant_rtn.T, calib @ W.T).item()
    err_ada = torch.nn.functional.mse_loss(calib @ dequant_ada.T, calib @ W.T).item()

    assert err_ada <= err_rtn
    codes = unpack_int6_w6a8(packed_ada, in_features)
    assert codes.min().item() >= 1
    assert codes.max().item() <= 63
    assert s_rel_ada.shape == (out_features, in_features // 16)
    assert s_ch_ada.shape == (out_features,)


@pytest.mark.parametrize("opt_name", ["adamw", "radam", "original", "prodigy"])
def test_w6a8_adaround_optimizers_run_cleanly(opt_name):
    torch.manual_seed(123)
    out_features, in_features = 8, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
    calib = torch.randn(32, in_features, dtype=torch.float32)

    conv = W6A8Converter(
        convrot=True,
        convrot_group_size=64,
        group_size=16,
        dynamic_convrot=False,
        no_learned_rounding=False,
        optimizer=opt_name,
        num_iter=10,
        lr=0.5,
        device="cpu",
    )
    packed, s_rel, s_channel, dequant, _ = conv.convert(
        W, key="test_opt", calibration_data=calib, has_bias=False
    )

    assert packed.shape == (out_features, (3 * in_features) // 4)
    assert s_rel.shape == (out_features, in_features // 16)
    assert s_channel.shape == (out_features,)
    assert dequant.shape == (out_features, in_features)
    codes = unpack_int6_w6a8(packed, in_features)
    assert codes.min().item() >= 1
    assert codes.max().item() <= 63


def test_w6a8_adaround_dualround_execution():
    torch.manual_seed(99)
    out_features, in_features = 8, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32) * 0.1
    calib = torch.randn(32, in_features, dtype=torch.float32)

    conv = W6A8Converter(
        convrot=True,
        convrot_group_size=64,
        group_size=16,
        dynamic_convrot=False,
        no_learned_rounding=False,
        scale_optimization="dualround",
        optimizer="adamw",
        num_iter=15,
        device="cpu",
    )
    packed, s_rel, s_channel, _, _ = conv.convert(
        W, key="test_dual", calibration_data=calib, has_bias=False
    )

    assert packed.shape == (out_features, (3 * in_features) // 4)
    assert s_rel.shape == (out_features, in_features // 16)
    assert s_channel.shape == (out_features,)
    codes = unpack_int6_w6a8(packed, in_features)
    assert codes.min().item() >= 1
    assert codes.max().item() <= 63
