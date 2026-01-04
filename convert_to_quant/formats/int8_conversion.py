"""
INT8 conversion utilities for convert_to_quant.

Converts legacy INT8 quantized models to comfy_quant format.
"""
import gc
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..constants import (
    TARGET_INT8_DTYPE,
    SCALE_DTYPE,
    NORMALIZE_SCALES_ENABLED,
)
from ..utils.tensor_utils import normalize_tensorwise_scales
from ..utils.comfy_quant import create_comfy_quant_tensor

def convert_int8_to_comfy_quant(
    input_file: str,
    output_file: str,
    block_size: int = 128,
    include_input_scale: bool = False,
    save_quant_metadata: bool = False,
):
    """
    Convert legacy INT8 quantized models to comfy_quant format.

    This is a format conversion only - NO quantization is performed.
    INT8 layers are detected by weight dtype (torch.int8).
    High-precision layers may have dummy .scale_weight which are removed.

    Args:
        input_file: Path to input INT8 safetensors file
        output_file: Path to output comfy_quant safetensors file
        block_size: Block size to use in comfy_quant metadata (default 128)
        include_input_scale: If True, add input_scale tensor (1.0 fp32) when missing
    """
    print(f"Converting INT8 model to comfy_quant format: {input_file}")
    print("-" * 60)
    print(f"Block size: {block_size}")
    print("-" * 60)

    # Load input tensors
    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            print(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Initialize metadata collection if enabled
    quant_metadata_layers = {} if save_quant_metadata else None

    # Group tensors by layer base name
    layer_info: Dict[str, Dict[str, torch.Tensor]] = {}
    other_tensors: Dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        # Skip any existing markers
        if key in ("scaled_fp8", "scaled_int8"):
            continue

        # Parse layer and suffix
        if key.endswith(".weight"):
            base = key[: -len(".weight")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["weight"] = tensor
        elif key.endswith(".scale_weight"):
            base = key[: -len(".scale_weight")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["scale_weight"] = tensor
        elif key.endswith(".scale_input"):
            base = key[: -len(".scale_input")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["scale_input"] = tensor
        elif key.endswith(".weight_scale"):
            # Already in new format - might be partial conversion
            base = key[: -len(".weight_scale")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["weight_scale"] = tensor
        elif key.endswith(".input_scale"):
            base = key[: -len(".input_scale")]
            if base not in layer_info:
                layer_info[base] = {}
            layer_info[base]["input_scale"] = tensor
        else:
            other_tensors[key] = tensor

    # Process layers
    output_tensors: Dict[str, torch.Tensor] = {}
    int8_layers = []
    hp_layers = []
    already_converted = []
    detected_formats = {}  # Track format detection counts

    for base_name, layer_data in tqdm(layer_info.items(), desc="Processing layers"):
        weight = layer_data.get("weight")
        scale_weight = layer_data.get("scale_weight")
        scale_input = layer_data.get("scale_input")
        weight_scale = layer_data.get("weight_scale")  # New format
        input_scale = layer_data.get("input_scale")  # New format

        if weight is None:
            # No weight tensor - just copy any scales through (unusual case)
            if scale_weight is not None:
                print(f"  WARNING: {base_name} has scale_weight but no weight tensor")
                output_tensors[f"{base_name}.scale_weight"] = scale_weight
            if weight_scale is not None:
                output_tensors[f"{base_name}.weight_scale"] = weight_scale
            continue

        # Detect if this is an INT8 layer by weight dtype
        is_int8 = weight.dtype == torch.int8

        # Check if already has new format keys (already converted)
        has_new_format = weight_scale is not None

        if is_int8:
            int8_layers.append(base_name)
            output_tensors[f"{base_name}.weight"] = weight

            if has_new_format:
                # Already converted - preserve existing
                already_converted.append(base_name)
                output_tensors[f"{base_name}.weight_scale"] = weight_scale
                if input_scale is not None:
                    output_tensors[f"{base_name}.input_scale"] = input_scale
                elif include_input_scale:
                    # No input_scale but flag is set - add default (fp32, 1.0 scalar)
                    output_tensors[f"{base_name}.input_scale"] = torch.tensor(
                        1.0, dtype=torch.float32
                    )
                # Detect INT8 format and block_size from weight_scale shape
                # Scale shape conventions from quant_ops.py layouts:
                # - BlockWiseINT8Layout: (M//bs, N//bs) - 2D block grid
                M, K = weight.shape[0], weight.shape[1] if weight.ndim >= 2 else 1

                detected_format = "int8_blockwise"
                if weight_scale is None:
                    detected_block_size = block_size
                    print(
                        f"    → Format: {detected_format} (no scale, assumed blockwise)"
                    )
                elif weight_scale.ndim == 2:
                    scale_dim0, scale_dim1 = weight_scale.shape
                    # Check if it's blockwise: (M//bs, N//bs)
                    if M % scale_dim0 == 0 and K % scale_dim1 == 0:
                        bs_M = M // scale_dim0
                        bs_K = K // scale_dim1
                        detected_block_size = bs_M if bs_M == bs_K else min(bs_M, bs_K)
                        print(
                            f"    → Format: {detected_format} (scale shape={weight_scale.shape}, bs={detected_block_size})"
                        )
                    else:
                        detected_block_size = block_size
                        print(
                            f"    → Format: {detected_format} (scale 2D, cannot identify layout)"
                        )
                else:
                    detected_block_size = block_size
                    print(
                        f"    → Format: {detected_format} (scale ndim={weight_scale.ndim}, assumed blockwise)"
                    )

                # Check if .comfy_quant already exists in other_tensors
                if f"{base_name}.comfy_quant" not in other_tensors:
                    detected_formats[detected_format] = (
                        detected_formats.get(detected_format, 0) + 1
                    )
                    comfy_quant_tensor = create_comfy_quant_tensor(
                        detected_format,
                        block_size=detected_block_size,
                        full_precision_matrix_mult=None,
                    )
                    output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

                    # Collect metadata if enabled
                    if save_quant_metadata:
                        meta_entry = {"format": detected_format}
                        # int8 is always blockwise in this context, so check is simple or implicit
                        if detected_block_size is not None:
                            meta_entry["group_size"] = detected_block_size

                        quant_metadata_layers[base_name] = meta_entry
            else:
                # Convert old format to new format
                if scale_weight is not None:
                    output_tensors[f"{base_name}.weight_scale"] = scale_weight
                else:
                    print(f"  WARNING: INT8 layer {base_name} missing scale_weight")

                # Handle scale_input -> input_scale
                if scale_input is not None:
                    output_tensors[f"{base_name}.input_scale"] = scale_input
                elif include_input_scale:
                    # No scale_input but flag is set - add default input_scale (fp32, 1.0 scalar)
                    output_tensors[f"{base_name}.input_scale"] = torch.tensor(
                        1.0, dtype=torch.float32
                    )

                # Detect INT8 format and block_size from scale_weight shape
                # Scale shape conventions from quant_ops.py layouts:
                # - BlockWiseINT8Layout: (M//bs, N//bs) - 2D block grid
                M, K = weight.shape[0], weight.shape[1] if weight.ndim >= 2 else 1

                detected_format = "int8_blockwise"
                if scale_weight is None:
                    detected_block_size = block_size
                    print(
                        f"    → Format: {detected_format} (no scale, assumed blockwise)"
                    )
                elif scale_weight.ndim == 2:
                    scale_dim0, scale_dim1 = scale_weight.shape
                    # Check if it's blockwise: (M//bs, N//bs)
                    if M % scale_dim0 == 0 and K % scale_dim1 == 0:
                        bs_M = M // scale_dim0
                        bs_K = K // scale_dim1
                        detected_block_size = bs_M if bs_M == bs_K else min(bs_M, bs_K)
                        print(
                            f"    → Format: {detected_format} (scale shape={scale_weight.shape}, bs={detected_block_size})"
                        )
                    else:
                        detected_block_size = block_size
                        print(
                            f"    → Format: {detected_format} (scale 2D, cannot identify layout)"
                        )
                else:
                    detected_block_size = block_size
                    print(
                        f"    → Format: {detected_format} (scale ndim={scale_weight.ndim}, assumed blockwise)"
                    )

                # Create .comfy_quant metadata
                detected_formats[detected_format] = (
                    detected_formats.get(detected_format, 0) + 1
                )
                comfy_quant_tensor = create_comfy_quant_tensor(
                    detected_format,
                    block_size=detected_block_size,
                    full_precision_matrix_mult=None,
                )
                output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

                # Collect metadata if enabled
                if save_quant_metadata:
                    meta_entry = {"format": detected_format}
                    if detected_block_size is not None:
                        meta_entry["group_size"] = detected_block_size

                    quant_metadata_layers[base_name] = meta_entry

        else:
            # High-precision layer: keep weight, remove dummy scales
            hp_layers.append(base_name)
            output_tensors[f"{base_name}.weight"] = weight

            if scale_weight is not None:
                print(
                    f"  Removing dummy scale_weight from high-precision layer: {base_name}"
                )
            if scale_input is not None:
                print(
                    f"  Removing dummy scale_input from high-precision layer: {base_name}"
                )

    # Add other tensors (bias, norms, etc.) - also fix any incorrect comfy_quant structures
    fixed_comfy_quant_count = 0
    for key, tensor in other_tensors.items():
        if key.endswith(".comfy_quant"):
            fixed_tensor, was_fixed = fix_comfy_quant_params_structure(tensor)
            if was_fixed:
                fixed_comfy_quant_count += 1
            output_tensors[key] = fixed_tensor
        else:
            output_tensors[key] = tensor

    # Summary
    print("\n" + "-" * 60)
    print("Conversion Summary:")
    print(f"  INT8 layers:           {len(int8_layers)}")
    if already_converted:
        print(f"    (already converted): {len(already_converted)}")
    print(f"  High-precision layers: {len(hp_layers)}")
    print(f"  Other tensors:         {len(other_tensors)}")
    print(f"  Total output tensors:  {len(output_tensors)}")
    if detected_formats:
        print("  Detected formats:")
        for fmt, count in sorted(detected_formats.items(), key=lambda x: -x[1]):
            print(f"    {fmt}: {count} layers")
    else:
        print(f"  INT8 format (CLI):     {detected_format}")
        print(f"  Block size (CLI):      {detected_block_size}")
    if fixed_comfy_quant_count > 0:
        print(
            f"  Fixed comfy_quant:     {fixed_comfy_quant_count} (nested params → flat)"
        )
    print("-" * 60)

    # Save output
    print(f"\nSaving to {output_file}...")
    try:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            print(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(output_tensors, output_file)

        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return
