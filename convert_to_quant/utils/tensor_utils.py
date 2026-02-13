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

def generate_calibration_data(
    tensors: Dict[str, torch.Tensor],
    calib_samples: int,
    seed: int,
    device: str,
    compute_dtype: torch.dtype = torch.float32,
) -> Dict[int, torch.Tensor]:
    """
    Generate random calibration data for each unique input dimension.

    Used for bias correction during quantization - creates synthetic
    activation samples for each unique weight input dimension found.

    Args:
        tensors: Dictionary of model tensors to scan for weight shapes
        calib_samples: Number of calibration samples to generate per dimension
        seed: Random seed for reproducibility
        device: Device to create tensors on ('cuda' or 'cpu')
        compute_dtype: Data type for calibration tensors (default: float32)

    Returns:
        Dict mapping input_features -> calibration tensor of shape (calib_samples, input_features)
    """
    seed_generator = torch.Generator(device=device)
    seed_generator.manual_seed(seed)

    calibration_data_cache: Dict[int, torch.Tensor] = {}

    for key, tensor in tensors.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples,
                    in_features,
                    dtype=compute_dtype,
                    generator=seed_generator,
                    device=device,
                )

    return calibration_data_cache


def compute_bias_correction(
    original_weight: torch.Tensor,
    dequantized_weight: torch.Tensor,
    original_bias: torch.Tensor,
    calibration_data: torch.Tensor,
    device: str,
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, bool]:
    """
    Compute bias correction based on weight quantization error.

    Uses calibration data to estimate the expected output error from
    weight quantization, then corrects the bias to compensate.

    Args:
        original_weight: Original FP32 weight tensor (out_features, in_features)
        dequantized_weight: Dequantized weight after quantization
        original_bias: Original bias tensor
        calibration_data: Random calibration data (samples, in_features)
        device: Device to compute on ('cuda' or 'cpu')
        compute_dtype: Data type for computation (default: float32)

    Returns:
        Tuple of (corrected_bias, success_flag). If calibration data is missing,
        returns (original_bias, False).
    """
    with torch.no_grad():
        X_calib_dev = calibration_data.to(device=device)
        W_orig_dev = original_weight.to(device=device, dtype=compute_dtype)
        W_dequant_dev = dequantized_weight.to(device=device, dtype=compute_dtype)
        b_orig_dev = original_bias.to(device=device, dtype=compute_dtype)

        weight_error = W_orig_dev - W_dequant_dev
        output_error = X_calib_dev @ weight_error.T
        bias_correction = output_error.mean(dim=0)
        b_new = b_orig_dev - bias_correction

        result = b_new.to(device="cpu", dtype=original_bias.dtype)

        # Cleanup
        del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev
        del weight_error, output_error, bias_correction, b_new
        if device == "cuda":
            torch.cuda.empty_cache()

        return result, True
