import torch
import pytest
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight, rotate_activation

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
