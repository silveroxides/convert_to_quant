"""
Legacy format cleanup utilities for convert_to_quant.

Handles legacy fp8_scaled format cleanup and input_scale addition.
"""
import gc
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any
from tqdm import tqdm

from ..constants import (
    TARGET_FP8_DTYPE,
    SCALE_DTYPE,
    NORMALIZE_SCALES_ENABLED,
)
from ..utils.tensor_utils import normalize_tensorwise_scales
from ..utils.logging import info, verbose, debug, minimal, warning, error, log_debug

def add_legacy_input_scale(
    input_file: str,
    output_file: str,
):
    """
    Add .scale_input tensors to legacy fp8_scaled models.

    This modifies a legacy fp8_scaled model to add .scale_input = [1.0] (fp32)
    for every FP8 layer that has .scale_weight but no .scale_input.
    Also converts the scaled_fp8 marker to a single-element [1] tensor.

    This does NOT convert to comfy_quant format - it keeps the legacy format
    but adds the missing input scales.

    Args:
        input_file: Path to input fp8_scaled safetensors file
        output_file: Path to output safetensors file
    """
    info("Adding .scale_input to legacy fp8_scaled model")
    info(f"Input: {input_file}")
    info(f"Output: {output_file}")
    info("-" * 60)

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
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Verify this is an fp8_scaled model
    if "scaled_fp8" not in tensors:
        error(
            "ERROR: This does not appear to be an fp8_scaled model (missing 'scaled_fp8' marker)"
        )
        error("       Use this mode only for legacy fp8_scaled format models.")
        return

    info("Verified: Input is fp8_scaled format")

    # Find all .scale_weight keys and check for corresponding .weight and .scale_input
    scale_weight_keys = [k for k in tensors.keys() if k.endswith(".scale_weight")]

    output_tensors: Dict[str, torch.Tensor] = {}
    added_scale_input = 0
    skipped_non_fp8 = 0
    already_has_input = 0

    for key, tensor in tensors.items():
        # Handle scaled_fp8 marker - convert to single-element vector
        if key == "scaled_fp8":
            output_tensors[key] = torch.tensor([1], dtype=tensor.dtype)
            verbose("  Converted scaled_fp8 marker to single-element tensor")
            continue

        # Copy all tensors through
        output_tensors[key] = tensor

    # Now add .scale_input for each .scale_weight that doesn't have one
    for scale_weight_key in scale_weight_keys:
        base_name = scale_weight_key[: -len(".scale_weight")]
        weight_key = f"{base_name}.weight"
        scale_input_key = f"{base_name}.scale_input"

        # Check if .scale_input already exists
        if scale_input_key in tensors:
            already_has_input += 1
            continue

        # Check if corresponding .weight exists and is FP8
        if weight_key not in tensors:
            continue

        weight = tensors[weight_key]
        if weight.dtype != torch.float8_e4m3fn:
            skipped_non_fp8 += 1
            continue

        # Add .scale_input = 1.0 fp32 (scalar format)
        output_tensors[scale_input_key] = torch.tensor(1.0, dtype=torch.float32)
        added_scale_input += 1

    # Summary
    info("\n" + "-" * 60)
    info("Conversion Summary:")
    info(f"  Total tensors input:     {len(tensors)}")
    info(f"  Total tensors output:    {len(output_tensors)}")
    info(f"  scale_input added:       {added_scale_input}")
    if already_has_input > 0:
        info(f"  Already had scale_input: {already_has_input}")
    if skipped_non_fp8 > 0:
        info(f"  Skipped (not FP8):       {skipped_non_fp8}")
    info("-" * 60)

    # Save output
    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            verbose(f"  Normalized {normalized_count} scale tensors to scalars")
        
        # Save with preserved metadata
        save_kwargs = {"metadata": original_metadata} if original_metadata else {}
        save_file(output_tensors, output_file, **save_kwargs)

        info("Conversion complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return

def cleanup_fp8_scaled(
    input_file: str,
    output_file: str,
    marker_size: int = 0,
    add_scale_input: bool = False,
):
    """
    Clean up legacy fp8_scaled models.

    This modifies a legacy fp8_scaled model to:
    - Set scaled_fp8 marker to empty((marker_size)) where marker_size is 0 or 2
    - Remove orphaned scale_weight/scale_input tensors (where weight is NOT FP8)
    - Optionally add missing .scale_input tensors for FP8 layers
    - Normalize 1-element scale tensors to scalars
    - Keep legacy format (no comfy_quant, no metadata)

    Args:
        input_file: Path to input fp8_scaled safetensors file
        output_file: Path to output safetensors file
        marker_size: Size of scaled_fp8 marker tensor (0 or 2)
        add_scale_input: If True, add .scale_input = 1.0 for FP8 layers missing it
    """
    info("Cleaning up legacy fp8_scaled model")
    info(f"Input: {input_file}")
    info(f"Output: {output_file}")
    info(f"scaled_fp8 marker size: {marker_size}")
    if add_scale_input:
        info("Adding missing scale_input: Yes")
    info("-" * 60)

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
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    # Build map of weight keys to their dtypes
    weight_dtypes: Dict[str, torch.dtype] = {}
    for key in tensors.keys():
        if key.endswith(".weight"):
            base_name = key[:-7]  # Remove '.weight'
            weight_dtypes[base_name] = tensors[key].dtype

    # Process tensors
    output_tensors: Dict[str, torch.Tensor] = {}
    removed_scale_weight = 0
    removed_scale_input = 0
    kept_scale_weight = 0
    kept_scale_input = 0

    for key, tensor in tensors.items():
        # Handle scaled_fp8 marker
        if key == "scaled_fp8":
            output_tensors[key] = torch.empty((marker_size,), dtype=TARGET_FP8_DTYPE)
            verbose(f"  Set scaled_fp8 marker to empty(({marker_size}))")
            continue

        # Check if this is a scale tensor
        if key.endswith(".scale_weight"):
            base_name = key[:-13]  # Remove '.scale_weight'
            if base_name in weight_dtypes:
                if weight_dtypes[base_name] == TARGET_FP8_DTYPE:
                    output_tensors[key] = tensor
                    kept_scale_weight += 1
                else:
                    removed_scale_weight += 1
                    verbose(f"  Removing orphaned: {key} (weight is {weight_dtypes[base_name]})")
            else:
                removed_scale_weight += 1
                verbose(f"  Removing orphaned: {key} (no weight found)")
            continue

        if key.endswith(".scale_input"):
            base_name = key[:-12]  # Remove '.scale_input'
            if base_name in weight_dtypes:
                if weight_dtypes[base_name] == TARGET_FP8_DTYPE:
                    output_tensors[key] = tensor
                    kept_scale_input += 1
                else:
                    removed_scale_input += 1
                    verbose(f"  Removing orphaned: {key} (weight is {weight_dtypes[base_name]})")
            else:
                removed_scale_input += 1
                verbose(f"  Removing orphaned: {key} (no weight found)")
            continue

        # Keep all other tensors
        output_tensors[key] = tensor

    # Add missing scale_input tensors for FP8 layers if requested
    added_scale_input = 0
    if add_scale_input:
        for base_name, dtype in weight_dtypes.items():
            if dtype == TARGET_FP8_DTYPE:
                scale_input_key = f"{base_name}.scale_input"
                if scale_input_key not in output_tensors:
                    output_tensors[scale_input_key] = torch.tensor(1.0, dtype=torch.float32)
                    added_scale_input += 1

    # Summary
    info("\n" + "-" * 60)
    info("Cleanup Summary:")
    info(f"  Total tensors input:      {len(tensors)}")
    info(f"  Total tensors output:     {len(output_tensors)}")
    if kept_scale_weight > 0:
        info(f"  scale_weight kept:        {kept_scale_weight}")
    if kept_scale_input > 0:
        info(f"  scale_input kept:         {kept_scale_input}")
    if added_scale_input > 0:
        info(f"  scale_input added:        {added_scale_input}")
    if removed_scale_weight > 0:
        info(f"  scale_weight removed:     {removed_scale_weight}")
    if removed_scale_input > 0:
        info(f"  scale_input removed:      {removed_scale_input}")
    info("-" * 60)

    # Save output
    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            verbose(f"  Normalized {normalized_count} scale tensors to scalars")
        
        # Save with preserved metadata
        save_kwargs = {"metadata": original_metadata} if original_metadata else {}
        save_file(output_tensors, output_file, **save_kwargs)

        info("Cleanup complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return
