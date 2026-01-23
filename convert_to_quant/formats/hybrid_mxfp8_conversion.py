"""
Hybrid MXFP8 conversion functions.

Converts MXFP8 models to Hybrid MXFP8 format by adding tensorwise scales
(calculated from block scales or imported from a separate tensorwise FP8 model).

The Hybrid MXFP8 layout adds a .weight_scalar tensor (tensorwise scale) to
standard MXFP8 models. This allows for a performant fallback on GPUs that
don't support native MXFP8 (e.g., Ada Lovelace/SM 8.9) by using the
tensorwise scale for standard FP8 matmul.
"""
import os
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Optional, Any

from ..utils.float_utils import e8m0_to_f32
from ..utils.tensor_utils import dict_to_tensor, tensor_to_dict
from ..utils.logging import info, warning, error, verbose


def _compute_tensorwise_scale(block_scales: torch.Tensor) -> torch.Tensor:
    """Compute optimal tensorwise scale from E8M0 block scales.

    Uses the maximum block scale to ensure all values fit within FP8 range.
    """
    # Input validation
    if block_scales.dim() != 2:
        # If flattened, try to proceed if 1D
        if block_scales.dim() == 1:
            pass
        else:
            raise ValueError(f"block_scales must be 1D or 2D, got {block_scales.dim()}D")

    if block_scales.numel() == 0:
        raise ValueError("block_scales cannot be empty")

    # Convert E8M0 to float32 (E8M0 is stored as uint8 exponents)
    scales_uint8 = block_scales.view(torch.uint8)
    scales_f32 = e8m0_to_f32(scales_uint8)

    # Get max scale
    max_scale = scales_f32.max()

    # Handle edge case: all-zero scales
    if max_scale == 0:
        max_scale = torch.tensor(1.0, device=block_scales.device, dtype=torch.float32)

    return max_scale.to(torch.float32).reshape(())


def convert_to_hybrid_mxfp8(
    input_file: str,
    output_file: str,
    tensor_scales_path: Optional[str] = None,
) -> None:
    """
    Convert an MXFP8 model to Hybrid MXFP8 format.

    Args:
        input_file: Path to existing MXFP8 safetensors model
        output_file: Path to save the hybrid model
        tensor_scales_path: Optional path to a tensorwise FP8 model to steal scales from.
                            If None, scales are computed from MXFP8 block scales.
    """
    info(f"Converting to Hybrid MXFP8: {input_file}")
    info(f"Output: {output_file}")

    if tensor_scales_path:
        info(f"Using tensorwise scales from: {tensor_scales_path}")

    # Load input model tensors
    tensors = {}
    with safe_open(input_file, framework="pt", device="cpu") as f:
        input_metadata = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    # Load tensorwise scales if provided
    tensor_scales = {}
    if tensor_scales_path:
        with safe_open(tensor_scales_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".weight_scale"):
                    base = k[:-13] # remove .weight_scale
                    tensor_scales[base] = f.get_tensor(k)
        info(f"Loaded {len(tensor_scales)} scales from external model")

    # Process tensors
    output_tensors = {}
    converted_count = 0

    # Track keys to iterate (copy of keys since we might add new ones)
    keys_to_process = list(tensors.keys())

    for key in keys_to_process:
        # Pass through existing tensors by default
        if key not in output_tensors:
            output_tensors[key] = tensors[key]

        # Look for .comfy_quant tensors to identify quantized layers
        if key.endswith(".comfy_quant"):
            base_key = key[:-12]

            try:
                config = tensor_to_dict(tensors[key])
            except Exception as e:
                warning(f"Failed to parse {key}: {e}")
                continue

            # Check if it's MXFP8
            if config.get("format") != "mxfp8":
                continue

            # Check if we have the corresponding block scales
            scale_key = f"{base_key}.weight_scale"
            if scale_key not in tensors:
                warning(f"Found comfy_quant but missing scale for {base_key}")
                continue

            # Compute or retrieve tensorwise scale
            scalar = None

            if tensor_scales_path:
                # Retrieve from external model
                if base_key in tensor_scales:
                    scalar = tensor_scales[base_key]
                    # Ensure it's a scalar (normalize)
                    if scalar.numel() == 1 and scalar.ndim > 0:
                        scalar = scalar.reshape(())
                    elif scalar.numel() > 1:
                        # Fallback to max if external scale is not scalar (e.g. rowwise)
                        # But user said "tensorwise model", so it should be scalar
                        scalar = scalar.max().reshape(())
                else:
                    warning(f"Missing external scale for {base_key}, falling back to computed")

            if scalar is None:
                # Compute from block scales
                block_scales = tensors[scale_key]
                scalar = _compute_tensorwise_scale(block_scales)

            # Store as .weight_scalar
            scalar_key = f"{base_key}.weight_scalar"
            output_tensors[scalar_key] = scalar

            # Update .comfy_quant config
            config["format"] = "hybrid_mxfp8"
            output_tensors[key] = dict_to_tensor(config)

            converted_count += 1
            verbose(f"Converted {base_key} to hybrid_mxfp8 (scalar={scalar.item():.6f})")

    # Update _quantization_metadata header
    output_metadata = {}
    if input_metadata:
        output_metadata = dict(input_metadata)

    if "_quantization_metadata" in output_metadata:
        try:
            quant_meta = json.loads(output_metadata["_quantization_metadata"])
            if "layers" in quant_meta:
                for layer_name, layer_meta in quant_meta["layers"].items():
                    # If we converted this layer, update metadata
                    if f"{layer_name}.weight_scalar" in output_tensors:
                        layer_meta["format"] = "hybrid_mxfp8"

            output_metadata["_quantization_metadata"] = json.dumps(quant_meta)
        except Exception as e:
            warning(f"Failed to update quantization metadata header: {e}")

    info("-" * 60)
    info(f"Converted {converted_count} layers to Hybrid MXFP8")
    info(f"Saving to {output_file}...")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(output_tensors, output_file, metadata=output_metadata)
    info("Done!")
