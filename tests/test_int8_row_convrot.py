import torch
import pytest
import math
from convert_to_quant.converters.learned_rounding import (
    LearnedRoundingConverter,
    _quantize_convrot_int8_rowwise,
)
from convert_to_quant.comfy.quant_ops import TensorWiseINT8Layout
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight, rotate_activation

@pytest.fixture
def setup_data():
    # Setup reproducible random inputs and weights
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_features = 128
    in_features = 256
    num_samples = 64

    W_orig = torch.randn(out_features, in_features, device=device, dtype=torch.float32)
    X = torch.randn(num_samples, in_features, device=device, dtype=torch.float32)

    # Layer with a pre-existing bias
    bias = torch.randn(out_features, device=device, dtype=torch.float32)

    return W_orig, X, bias, out_features, in_features, num_samples, device

def test_orthogonal_invariance(setup_data):
    W_orig, X, bias, out_features, in_features, num_samples, device = setup_data

    # Group size must divide in_features
    group_size = 64
    H = build_hadamard(group_size, device=device, dtype=torch.float32)

    # Rotate weight and activation
    W_rot = rotate_weight(W_orig, H, group_size)
    X_rot = rotate_activation(X, H, group_size)

    # Check rotation mathematical invariance: X_rot @ W_rot.T == X @ W_orig.T
    Y_orig = X @ W_orig.T
    Y_rot = X_rot @ W_rot.T

    error = torch.norm(Y_orig - Y_rot)
    print(f"Rotation invariance error: {error.item():.6e}")
    # Under machine precision and Kronecker-Kronecker operations, slightly relaxed tolerance is appropriate
    assert error.item() < 1e-3

def test_adaround_convrot_pipeline(setup_data):
    W_orig, X, bias, out_features, in_features, num_samples, device = setup_data

    # Initialize the converter for INT8, row-wise scaling with ConvRot active
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=64,
        num_iter=200,      # Small iterations for test speed
        optimizer="adamw", # Robust optimizer
        lr=1.0,
        device=device
    )

    # Verify initialized parameters
    assert converter.convrot is True
    assert converter.scaling_mode == "row"
    assert converter.target_format == "int8"

    # Perform conversion with calibration data X
    qdata, scale, dequant_w, extra_tensors = converter.convert(
        W_orig, key="test_layer.weight", depth=0, calibration_data=X
    )

    # 1. Check weight shape and types
    assert qdata.shape == W_orig.shape
    assert qdata.dtype == torch.int8
    # In row-wise mode, scales are 2D with shape (out_features, 1) for natural broadcasting
    assert scale.shape == (out_features, 1)
    assert scale.dtype == torch.float32
    assert dequant_w.shape == W_orig.shape

    # 2. Check metadata
    assert "bias_correction" in extra_tensors
    bias_correction = extra_tensors["bias_correction"]
    assert bias_correction.shape == (out_features,)

    # 3. Validation of Residual Bias Calibration (Phase 4)
    # The output mean error with uncorrected bias vs corrected bias
    group_size = 64
    H = build_hadamard(group_size, device=device, dtype=torch.float32)
    W_rot = rotate_weight(W_orig, H, group_size)
    X_rot = rotate_activation(X, H, group_size)

    Y_ref = X_rot @ W_rot.T + bias

    # Quantized output without bias correction
    Y_quant_uncorrected = X_rot @ dequant_w.T + bias

    # Quantized output with bias correction
    corrected_bias = bias + bias_correction.to(device=device)
    Y_quant_corrected = X_rot @ dequant_w.T + corrected_bias

    # Compute mean errors
    mean_shift_uncorrected = (Y_ref - Y_quant_uncorrected).mean(dim=0).norm().item()
    mean_shift_corrected = (Y_ref - Y_quant_corrected).mean(dim=0).norm().item()

    print(f"Mean shift uncorrected: {mean_shift_uncorrected:.6e}")
    print(f"Mean shift corrected: {mean_shift_corrected:.6e}")

    # The shift should be virtually eliminated (very close to zero)
    assert mean_shift_corrected < 1e-5
    assert mean_shift_corrected < mean_shift_uncorrected

def test_triton_fused_kernel_compatibility(setup_data):
    W_orig, X, bias, out_features, in_features, num_samples, device = setup_data

    # Verify row-wise scale dequantization equation matches perfectly
    # TensorWiseINT8Layout.dequantize(qdata, scale) == qdata * scale
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=64,
        num_iter=10,
        device=device
    )

    qdata, scale, dequant_w, extra_tensors = converter.convert(W_orig, calibration_data=X)

    # Row-wise dequantized reference - scale is (out_features, 1) so we multiply directly
    dequant_ref = qdata.float() * scale

    assert torch.allclose(dequant_w, dequant_ref, atol=1e-6)


def test_dynamic_convrot_resolution():
    from convert_to_quant.utils.convrot import find_max_compatible_group_size

    assert find_max_compatible_group_size(4096, 256) == 4096
    assert find_max_compatible_group_size(1024, 256) == 1024
    assert find_max_compatible_group_size(1152, 256) is None  # Not divisible by any power of 4 >= 256
    assert find_max_compatible_group_size(3072, 256) == 1024  # 3072 is divisible by 1024
    assert find_max_compatible_group_size(512, 256) == 256    # 512 is divisible by 256
    assert find_max_compatible_group_size(384, 256) is None   # 384 is divisible by 64 (which is < 256)


def test_dynamic_convrot_pipeline():
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # W with in_features = 512 (divisible by 256, but not 1024)
    W_orig = torch.randn(128, 512, device=device, dtype=torch.float32)
    X = torch.randn(64, 512, device=device, dtype=torch.float32)

    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        dynamic_convrot=True,
        convrot_group_size=256,  # min group size
        num_iter=10,
        device=device
    )

    assert converter.dynamic_convrot is True
    assert converter.convrot is True

    qdata, scale, dequant_w, extra_tensors = converter.convert(W_orig, calibration_data=X)

    assert qdata.shape == W_orig.shape
    assert qdata.dtype == torch.int8
    assert scale.shape == (128, 1)


@pytest.mark.parametrize("group_size", [64, 256])
@pytest.mark.parametrize("zero_rows", [False, True])
def test_quantize_convrot_int8_rowwise_equations(group_size, zero_rows):
    torch.manual_seed(group_size)
    tensor = torch.randn(4, group_size, dtype=torch.float32)
    if zero_rows:
        tensor[0].zero_()

    qdata, scale = _quantize_convrot_int8_rowwise(tensor)

    expected_scale = (tensor.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    scale_for_math = expected_scale.to(dtype=tensor.dtype)
    scale_for_math = torch.where(
        scale_for_math == 0,
        torch.full_like(scale_for_math, torch.finfo(tensor.dtype).tiny),
        scale_for_math,
    )
    expected_qdata = (tensor / scale_for_math).round().clamp(-128.0, 127.0).to(torch.int8)

    assert torch.equal(qdata, expected_qdata)
    assert torch.equal(scale, expected_scale)
    assert scale.dtype == torch.float32
    assert scale.shape == (tensor.shape[0], 1)


@pytest.mark.parametrize(
    ("width", "expected_group"),
    [(512, 256), (2688, 64), (96, None)],
)
def test_primary_simple_convrot_applied_groups_and_incompatible_width(width, expected_group, monkeypatch):
    torch.manual_seed(width)
    weight = torch.randn(2, width, dtype=torch.float32)
    applied_groups = []
    original_rotate_weight = rotate_weight

    def record_rotate_weight(tensor, hadamard, group_size):
        applied_groups.append(group_size)
        return original_rotate_weight(tensor, hadamard, group_size)

    monkeypatch.setattr(
        "convert_to_quant.utils.convrot.rotate_weight",
        record_rotate_weight,
    )
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        primary_int8_convrot_compat=True,
        no_learned_rounding=True,
        device="cpu",
    )

    qdata, scale, dequantized, _ = converter.convert(weight, has_bias=False)

    assert applied_groups == ([] if expected_group is None else [expected_group])
    assert converter.last_primary_convrot_group_size == expected_group
    assert qdata.dtype == torch.int8
    assert scale.dtype == torch.float32
    assert scale.shape == (weight.shape[0], 1)
    assert torch.equal(dequantized, qdata.float() * scale)


def test_primary_simple_convrot_zero_rows():
    weight = torch.zeros(3, 2688, dtype=torch.float32)
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        primary_int8_convrot_compat=True,
        no_learned_rounding=True,
        device="cpu",
    )

    qdata, scale, dequantized, _ = converter.convert(weight, has_bias=False)

    assert converter.last_primary_convrot_group_size == 64
    assert torch.count_nonzero(qdata) == 0
    assert torch.equal(scale, torch.full((3, 1), 1e-30, dtype=torch.float32))
    assert torch.count_nonzero(dequantized) == 0


def test_primary_learned_convrot_group64_uses_optimizer(monkeypatch):
    torch.manual_seed(1)
    weight = torch.randn(2, 2688, dtype=torch.float32)
    calibration = torch.randn(2, 2688, dtype=torch.float32)
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        primary_int8_convrot_compat=True,
        no_learned_rounding=False,
        num_iter=1,
        device="cpu",
    )
    calls = []

    def record_optimizer(tensor, qdata, scale, x_rot, y_ref):
        calls.append((tensor.shape, x_rot.shape, y_ref.shape))
        return qdata, scale

    monkeypatch.setattr(converter, "_optimize_int8_adaround", record_optimizer)

    _, scale, _, extra_tensors = converter.convert(
        weight,
        calibration_data=calibration,
        has_bias=True,
    )

    assert converter.last_primary_convrot_group_size == 64
    assert calls == [((2, 2688), (2, 2688), (2, 2))]
    assert scale.shape == (2, 1)
    assert "bias_correction" in extra_tensors

