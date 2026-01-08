"""
Format migration utilities for convert_to_quant.

Converts between legacy quantization formats and comfy_quant format.
"""
import gc
import json
import os
import re
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any, Optional
from tqdm import tqdm


from ..constants import (
    TARGET_FP8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    NORMALIZE_SCALES_ENABLED,
)
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales
from ..utils.comfy_quant import create_comfy_quant_tensor, fix_comfy_quant_params_structure

def convert_fp8_scaled_to_comfy_quant(
    input_file: str,
    output_file: str,
    hp_filter: Optional[str] = None,
    full_precision_mm: bool = False,
    include_input_scale: bool = False,
    save_quant_metadata: bool = False,
):
    """
    Convert legacy fp8_scaled format to comfy_quant format.

    This is a format conversion only - NO quantization is performed.
    FP8 layers are detected by weight dtype (float8_e4m3fn), not by scale presence.
    High-precision layers may have dummy .scale_weight which are removed.

    Args:
        input_file: Path to input fp8_scaled safetensors file
        output_file: Path to output comfy_quant safetensors file
        hp_filter: Optional regex pattern to validate high-precision layers
        full_precision_mm: If True, set full_precision_matrix_mult in .comfy_quant
        include_input_scale: If True, add input_scale tensor (1.0 fp32) when missing
    """
    print("Converting fp8_scaled to comfy_quant format")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-" * 60)

    # Load input tensors and preserve original metadata
    tensors: Dict[str, torch.Tensor] = {}
    original_metadata: Dict[str, str] = {}
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            # Preserve original file metadata
            original_metadata = f.metadata() or {}
            if original_metadata:
                print(f"Preserving {len(original_metadata)} original metadata entries")
            
            print(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Initialize metadata collection if enabled
    quant_metadata_layers = {} if save_quant_metadata else None

    # Verify this is an fp8_scaled model
    if "scaled_fp8" not in tensors:
        print(
            "ERROR: This does not appear to be an fp8_scaled model (missing 'scaled_fp8' marker)"
        )
        print("       Use this mode only for legacy fp8_scaled format models.")
        return

    print("Verified: Input is fp8_scaled format")

    # Compile hp_filter regex if provided
    hp_pattern = None
    if hp_filter:
        try:
            hp_pattern = re.compile(hp_filter)
            print(f"High-precision filter: {hp_filter}")
        except re.error as e:
            print(f"ERROR: Invalid regex pattern '{hp_filter}': {e}")
            return

    # Group tensors by layer base name
    # Find all .weight tensors and their associated scales
    layer_info: Dict[str, Dict[str, torch.Tensor]] = {}
    other_tensors: Dict[str, torch.Tensor] = {}

    for key, tensor in tensors.items():
        if key == "scaled_fp8":
            continue  # Skip marker, will be removed

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
        else:
            other_tensors[key] = tensor

    # Process layers
    output_tensors: Dict[str, torch.Tensor] = {}
    fp8_layers = []
    hp_layers = []

    for base_name, layer_data in tqdm(layer_info.items(), desc="Processing layers"):
        weight = layer_data.get("weight")
        scale_weight = layer_data.get("scale_weight")
        scale_input = layer_data.get("scale_input")

        if weight is None:
            # No weight tensor - just copy any scales through (unusual case)
            if scale_weight is not None:
                print(f"  WARNING: {base_name} has scale_weight but no weight tensor")
                output_tensors[f"{base_name}.scale_weight"] = scale_weight
            if scale_input is not None:
                output_tensors[f"{base_name}.scale_input"] = scale_input
            continue

        # Detect if this is an FP8 layer by weight dtype
        is_fp8 = weight.dtype == TARGET_FP8_DTYPE

        if is_fp8:
            # FP8 layer: rename scales and add .comfy_quant
            fp8_layers.append(base_name)
            output_tensors[f"{base_name}.weight"] = weight

            if scale_weight is not None:
                output_tensors[f"{base_name}.weight_scale"] = scale_weight
            else:
                print(f"  WARNING: FP8 layer {base_name} missing scale_weight")

            # Handle scale_input -> input_scale
            if scale_input is not None:
                output_tensors[f"{base_name}.input_scale"] = scale_input
            elif include_input_scale:
                # No scale_input but flag is set - add default input_scale (scalar)
                output_tensors[f"{base_name}.input_scale"] = torch.tensor(
                    1.0, dtype=torch.float32
                )

            # Detect format and block_size from scale_weight tensor shape
            # Scale shape conventions from quant_ops.py layouts:
            # - TensorCoreFP8Layout: () or (1,) - scalar, single global scale
            # - RowWiseFP8Layout: (M,) - 1D, one scale per output row
            # - BlockWiseFP8Layout: (M//bs, N//bs) - 2D grid, one scale per tile
            # - Block3DFP8Layout: (M, N//bs, 1) - 3D, per-row-block scaling
            M, N = weight.shape[0], weight.shape[1] if weight.ndim >= 2 else 1

            if scale_weight is None:
                # No scale tensor - assume tensor-wise (this shouldn't happen for valid FP8 models)
                format_type = "float8_e4m3fn"
                block_size = None
                print(
                    f"    → Format: {format_type} (missing scale, assumed tensor-wise)"
                )
            elif scale_weight.numel() == 1:
                # Scalar or single-element tensor → tensor-wise scaling
                format_type = "float8_e4m3fn"
                block_size = None
                print(f"    → Format: {format_type} (scale numel=1)")
            elif scale_weight.ndim == 1:
                # 1D scale tensor - check if it matches row count
                if scale_weight.shape[0] == M:
                    # One scale per row → row-wise
                    format_type = "float8_e4m3fn_rowwise"
                    block_size = None
                    print(
                        f"    → Format: {format_type} (scale shape={scale_weight.shape}, M={M})"
                    )
                else:
                    # 1D but doesn't match M - could be flattened block scale
                    # Try to infer block_size: scale_count = (M//bs) * (N//bs) = M*N / bs^2
                    # So bs = sqrt(M*N / scale_count)
                    scale_count = scale_weight.shape[0]
                    total_elements = M * N
                    if scale_count > 0 and total_elements % scale_count == 0:
                        bs_squared = total_elements // scale_count
                        bs = int(bs_squared**0.5)
                        if bs * bs == bs_squared and M % bs == 0 and N % bs == 0:
                            format_type = "float8_e4m3fn_blockwise"
                            block_size = bs
                            print(
                                f"    → Format: {format_type} (scale 1D flattened, inferred bs={bs})"
                            )
                        else:
                            format_type = "float8_e4m3fn"
                            block_size = None
                            print(
                                f"    → Format: {format_type} (scale 1D unknown pattern, fallback)"
                            )
                    else:
                        format_type = "float8_e4m3fn"
                        block_size = None
                        print(
                            f"    → Format: {format_type} (scale 1D, cannot infer block)"
                        )
            elif scale_weight.ndim == 2:
                # 2D scale - most likely block-wise: (M//bs, N//bs)
                scale_M, scale_N = scale_weight.shape
                if M % scale_M == 0 and N % scale_N == 0:
                    bs_M = M // scale_M
                    bs_N = N // scale_N
                    if bs_M == bs_N:
                        # Square blocks
                        format_type = "float8_e4m3fn_blockwise"
                        block_size = bs_M
                        print(
                            f"    → Format: {format_type} (scale 2D, bs={block_size})"
                        )
                    else:
                        # Non-square blocks - use smaller dimension as block_size
                        format_type = "float8_e4m3fn_blockwise"
                        block_size = min(bs_M, bs_N)
                        print(
                            f"    → Format: {format_type} (scale 2D non-square, bs={block_size})"
                        )
                else:
                    # Doesn't divide evenly - fallback
                    format_type = "float8_e4m3fn"
                    block_size = None
                    print(
                        f"    → Format: {format_type} (scale 2D but dims don't divide)"
                    )
            elif scale_weight.ndim == 3:
                # 3D scale - likely Block3DFP8Layout: (M, N//bs, 1)
                scale_M, scale_blocks, scale_last = scale_weight.shape
                if scale_M == M and scale_last == 1 and N % scale_blocks == 0:
                    format_type = "float8_e4m3fn_block3d"
                    block_size = N // scale_blocks
                    print(f"    → Format: {format_type} (scale 3D, bs={block_size})")
                else:
                    format_type = "float8_e4m3fn"
                    block_size = None
                    print(f"    → Format: {format_type} (scale 3D unknown pattern)")
            else:
                # Unknown ndim
                format_type = "float8_e4m3fn"
                block_size = None
                print(
                    f"    → Format: {format_type} (scale ndim={scale_weight.ndim} unknown)"
                )

            # Create .comfy_quant metadata
            comfy_quant_tensor = create_comfy_quant_tensor(
                format_type,
                block_size=block_size,
                full_precision_matrix_mult=full_precision_mm
                if full_precision_mm
                else None,
            )
            output_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor

            # Collect metadata if enabled
            if save_quant_metadata:
                meta_entry = {"format": format_type}
                block_based_formats = {"int8_blockwise", "float8_e4m3fn_blockwise"}
                if block_size is not None and format_type in block_based_formats:
                    meta_entry["group_size"] = block_size
                if full_precision_mm:
                    meta_entry["full_precision_matrix_mult"] = True

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

    # Validate hp_filter if provided
    if hp_pattern:
        print("\nValidating high-precision filter...")
        violations = []
        for base_name in fp8_layers:
            if hp_pattern.search(base_name):
                violations.append(base_name)

        if violations:
            print(
                "ERROR: The following layers matched hp-filter but are FP8 (not high-precision):"
            )
            for v in violations:
                print(f"  - {v}")
            print(
                "\nThese layers have float8_e4m3fn weights. If they should be high-precision,"
            )
            print(
                "the input model needs to be regenerated with correct layer exclusions."
            )
            return

        # Report matched hp layers
        matched_hp = [b for b in hp_layers if hp_pattern.search(b)]
        if matched_hp:
            print(f"  Validated {len(matched_hp)} high-precision layers match filter")

    # Summary
    print("\n" + "-" * 60)
    print("Conversion Summary:")
    print(f"  FP8 layers:            {len(fp8_layers)}")
    print(f"  High-precision layers: {len(hp_layers)}")
    print(f"  Other tensors:         {len(other_tensors)}")
    print(f"  Total output tensors:  {len(output_tensors)}")
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

        # Prepare metadata args - merge original metadata with new quantization metadata
        output_metadata = dict(original_metadata)  # Start with original metadata
        
        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            output_metadata["_quantization_metadata"] = json.dumps(full_metadata)
            print(
                f"  Adding quantization metadata for {len(quant_metadata_layers)} layers"
            )
        
        save_kwargs = {"metadata": output_metadata} if output_metadata else {}

        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            print(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(output_tensors, output_file, **save_kwargs)

        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return
