import torch
import pytest
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight, rotate_activation
from convert_to_quant.utils.tensor_utils import prepare_calibration_data


def test_build_hadamard():
    H = build_hadamard(4, device="cpu", dtype=torch.float32)
    assert H.shape == (4, 4)
    # Check orthogonality: H @ H.T should be Identity
    identity = torch.eye(4)
    assert torch.allclose(H @ H.T, identity, atol=1e-5)


def test_rotate_weight():
    H = build_hadamard(4, device="cpu", dtype=torch.float32)
    weight = torch.randn(8, 12)
    rotated = rotate_weight(weight, H, 4)
    assert rotated.shape == (8, 12)


def test_rotate_activation():
    H = build_hadamard(4, device="cpu", dtype=torch.float32)
    act = torch.randn(2, 8, 12)
    rotated = rotate_activation(act, H, 4)
    assert rotated.shape == (2, 8, 12)


def test_prepare_calibration_data():
    W = torch.randn(8, 16)
    X = torch.randn(10, 16)

    # Test without convrot
    X_rot, Y_ref, H = prepare_calibration_data(W, X, convrot=False, convrot_group_size=4, device="cpu")
    assert torch.allclose(X_rot, X)
    assert torch.allclose(Y_ref, X @ W.T)
    assert H is None

    # Test with convrot
    X_rot, Y_ref, H = prepare_calibration_data(W, X, convrot=True, convrot_group_size=4, device="cpu")
    assert H is not None
    assert H.shape == (4, 4)
    assert X_rot.shape == X.shape

    # Check that Y_ref is mathematically correct. Since weight wasn't rotated, Y_ref is X_rot @ W.T
    assert torch.allclose(Y_ref, X_rot @ W.T, atol=1e-5)

    # Test with dynamic calib_scale < 1.0
    calib_scale = 0.5
    X_rot_scaled, Y_ref_scaled, H_scaled = prepare_calibration_data(
        W, X, convrot=True, convrot_group_size=4, device="cpu", calib_scale=calib_scale
    )
    # 10 * 0.5 = 5 samples
    assert X_rot_scaled.shape == (5, 16)
    assert Y_ref_scaled.shape == (5, 8)
    assert torch.allclose(X_rot_scaled, X_rot[:5])
    assert torch.allclose(Y_ref_scaled, Y_ref[:5])

    # Test dynamic calib_scale where result is at least 1 sample (e.g., 0.01)
    X_rot_scaled_min, _, _ = prepare_calibration_data(
        W, X, convrot=False, convrot_group_size=4, device="cpu", calib_scale=0.01
    )
    assert X_rot_scaled_min.shape == (1, 16)

