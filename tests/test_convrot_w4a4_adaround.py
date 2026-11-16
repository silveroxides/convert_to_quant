import pytest
import torch

from convert_to_quant.converters.int4_converter import INT4Converter
from convert_to_quant.utils.convrot import unpack_int4_row_major


def test_int4_adaround_reduces_reconstruction_error_vs_rtn():
    torch.manual_seed(42)
    out_features, in_features = 16, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32) * 3.0
    calib = torch.randn(64, in_features, dtype=torch.float32)

    # 1. Simple RTN
    rtn_conv = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        no_learned_rounding=True,
        device="cpu",
    )
    packed_rtn, scale_rtn, dequant_rtn, _ = rtn_conv.convert(
        W.clone(), key="layer", calibration_data=calib.clone(), has_bias=False
    )

    # 2. Learned AdaRound (AdamW)
    ada_conv = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        no_learned_rounding=False,
        optimizer="adamw",
        num_iter=35,
        lr=1.0,
        device="cpu",
    )
    packed_ada, scale_ada, dequant_ada, _ = ada_conv.convert(
        W.clone(), key="layer", calibration_data=calib.clone(), has_bias=False
    )

    err_rtn = torch.nn.functional.mse_loss(calib @ dequant_rtn.T, calib @ W.T).item()
    err_ada = torch.nn.functional.mse_loss(calib @ dequant_ada.T, calib @ W.T).item()

    # AdaRound must reduce output activation MSE vs RTN
    assert err_ada < err_rtn

    # Clamping contract
    unpacked_ada = unpack_int4_row_major(packed_ada)
    assert unpacked_ada.min().item() >= -7
    assert unpacked_ada.max().item() <= 7


@pytest.mark.parametrize("opt_name", ["adamw", "radam", "original", "prodigy"])
def test_int4_adaround_optimizers_run_cleanly(opt_name):
    torch.manual_seed(123)
    out_features, in_features = 8, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32)
    calib = torch.randn(32, in_features, dtype=torch.float32)

    conv = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        no_learned_rounding=False,
        optimizer=opt_name,
        num_iter=10,
        lr=0.5,
        device="cpu",
    )
    packed, scale, dequant, _ = conv.convert(
        W, key="test_opt", calibration_data=calib, has_bias=False
    )

    assert packed.shape == (out_features, in_features // 2)
    assert scale.shape == (out_features,)
    assert dequant.shape == (out_features, in_features)
    unpacked = unpack_int4_row_major(packed)
    assert unpacked.min().item() >= -7
    assert unpacked.max().item() <= 7


def test_int4_adaround_dualround_execution():
    torch.manual_seed(99)
    out_features, in_features = 8, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32)
    calib = torch.randn(32, in_features, dtype=torch.float32)

    conv = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        no_learned_rounding=False,
        scale_optimization="dualround",
        optimizer="adamw",
        num_iter=15,
        device="cpu",
    )
    packed, scale, dequant, _ = conv.convert(
        W, key="test_dual", calibration_data=calib, has_bias=False
    )

    assert packed.shape == (out_features, in_features // 2)
    assert scale.shape == (out_features,)
    unpacked = unpack_int4_row_major(packed)
    assert unpacked.min().item() >= -7
    assert unpacked.max().item() <= 7
