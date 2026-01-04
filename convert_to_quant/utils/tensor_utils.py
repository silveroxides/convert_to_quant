"""
Tensor utility functions for convert_to_quant.

Provides serialization helpers for dictionary/tensor conversion and scale normalization.
"""
import json
import torch
from typing import Dict, Tuple


def dict_to_tensor(data_dict: dict) -> torch.Tensor:
    """
    Convert a dictionary to a torch.uint8 tensor containing JSON bytes.

    Args:
        data_dict: Dictionary to serialize

    Returns:
        torch.uint8 tensor containing UTF-8 encoded JSON
    """
    json_str = json.dumps(data_dict)
    byte_data = json_str.encode("utf-8")
    tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)
    return tensor_data


def tensor_to_dict(tensor_data: torch.Tensor) -> dict:
    """
    Convert a torch.uint8 tensor containing JSON bytes to a dictionary.

    Args:
        tensor_data: Tensor containing UTF-8 encoded JSON bytes

    Returns:
        Parsed dictionary
    """
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode("utf-8")
    data_dict = json.loads(json_str)
    return data_dict


def normalize_tensorwise_scales(
    tensors: Dict[str, torch.Tensor],
    enabled: bool = True,
) -> Tuple[Dict[str, torch.Tensor], int]:
    """
    Normalize 1-element scale tensors to scalars in-place.

    Tensorwise quantization should use scalar scales, not 1-element arrays.
    This ensures consistency with ComfyUI loader expectations.

    Args:
        tensors: Dictionary of tensors to normalize (modified in-place)
        enabled: If False, skip normalization and return immediately

    Returns:
        Tuple of (tensors dict, count of normalized tensors)
    """
    if not enabled:
        return tensors, 0

    normalized_count = 0
    scale_suffixes = (".input_scale", ".weight_scale", ".scale_input", ".scale_weight")

    for key in list(tensors.keys()):
        if any(key.endswith(suffix) for suffix in scale_suffixes):
            tensor = tensors[key]
            # Only normalize if it's a 1-element array (shape like (1,) or (1,1))
            if tensor.numel() == 1 and tensor.ndim > 0:
                tensors[key] = tensor.squeeze()  # Convert to scalar
                normalized_count += 1

    return tensors, normalized_count
