"""Stable fingerprints for behavior-locking characterization tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    """Describe logical tensor metadata and hash its exact storage bytes."""
    value = tensor.detach().contiguous().cpu()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_map_fingerprint(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    """Fingerprint a tensor mapping in stable key order."""
    return {key: tensor_fingerprint(tensors[key]) for key in sorted(tensors)}


def safetensors_fingerprint(path: str | Path) -> dict[str, Any]:
    """Fingerprint safetensors contents without depending on file/header ordering."""
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensors = {key: tensor_fingerprint(handle.get_tensor(key)) for key in sorted(handle.keys())}
        metadata = handle.metadata() or {}

    normalized_metadata = {
        key: json.loads(value) if key == "_quantization_metadata" else value
        for key, value in sorted(metadata.items())
    }
    return {"tensors": tensors, "metadata": normalized_metadata}
