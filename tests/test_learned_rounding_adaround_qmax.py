import torch

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter


def _make_converter(**overrides):
    kwargs = dict(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=4,
        no_learned_rounding=False,
        num_iter=5,
        device="cpu",
        optimizer="adamw",
    )
    kwargs.update(overrides)
    return LearnedRoundingConverter(**kwargs)


def test_adaround_respects_custom_qmax():
    torch.manual_seed(0)
    converter = _make_converter()
    M, N = 8, 16
    W = torch.randn(M, N) * 5.0
    X_rot = torch.randn(32, N)
    Y_ref = X_rot @ W.T

    # Baseline RTN quantization at qmax=7 to seed the optimizer
    scale = W.abs().amax(dim=-1).clamp(min=1e-10) / 7.0
    qdata = (W / scale.unsqueeze(-1)).round().clamp(-7, 7).to(torch.int8)

    final_qdata, final_scale = converter._optimize_int8_adaround(W, qdata, scale, X_rot, Y_ref, qmax=7)

    assert final_qdata.min() >= -7
    assert final_qdata.max() <= 7
    assert final_scale.shape == (M,)


def test_adaround_default_qmax_is_int8():
    torch.manual_seed(0)
    converter = _make_converter()
    M, N = 8, 16
    W = torch.randn(M, N) * 50.0
    X_rot = torch.randn(32, N)
    Y_ref = X_rot @ W.T

    scale = W.abs().amax(dim=-1).clamp(min=1e-10) / 127.0
    qdata = (W / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

    final_qdata, _ = converter._optimize_int8_adaround(W, qdata, scale, X_rot, Y_ref)

    assert final_qdata.min() >= -127
    assert final_qdata.max() <= 127
