"""
ComfyUI quantization metadata utilities for convert_to_quant.

Handles .comfy_quant tensor creation, parsing, editing, and performance heuristics.
"""
import json
import os
import re
import torch
from typing import Dict, Tuple, Optional, Any, List
from safetensors import safe_open
from safetensors.torch import save_file

from .tensor_utils import dict_to_tensor, tensor_to_dict, normalize_tensorwise_scales
from ..constants import NORMALIZE_SCALES_ENABLED
from .logging import info, verbose, warning, error, minimal


# Block-based formats that require group_size
BLOCK_BASED_FORMATS = (
    "int8_blockwise",
    "float8_e4m3fn_blockwise",
)

def create_comfy_quant_tensor(
    format_type: str,
    block_size: Optional[int] = None,
    full_precision_matrix_mult: Optional[bool] = None,
) -> torch.Tensor:
    """
    Create a .comfy_quant layer configuration tensor for ComfyUI.

    Args:
        format_type: One of "float8_e4m3fn", "float8_e4m3fn_rowwise", "float8_e4m3fn_blockwise",
                     or "int8_blockwise"
        block_size: Block/group size for quantization (for block-based formats)
        full_precision_matrix_mult: If True, adds "full_precision_matrix_mult": True.
                                    If False or None, this key is omitted.

    Returns:
        torch.uint8 tensor containing JSON-encoded layer configuration
    """
    comfy_quant = {"format": format_type}

    # Add group_size directly (not nested in params) for block-based formats
    if block_size is not None and format_type in BLOCK_BASED_FORMATS:
        comfy_quant["group_size"] = block_size

    if full_precision_matrix_mult is True:
        comfy_quant["full_precision_matrix_mult"] = True

    return dict_to_tensor(comfy_quant)

def fix_comfy_quant_params_structure(
    comfy_quant_tensor: torch.Tensor,
) -> Tuple[torch.Tensor, bool]:
    """
    Check and fix comfy_quant config with incorrect nested params structure.

    Old buggy format: {"format": "...", "params": {"group_size": 128}}
    Correct format:   {"format": "...", "group_size": 128}

    Args:
        comfy_quant_tensor: Existing .comfy_quant tensor from model

    Returns:
        Tuple of (fixed_tensor, was_fixed)
        - fixed_tensor: Fixed tensor (or original if no fix needed)
        - was_fixed: True if the structure was fixed
    """
    try:
        config = tensor_to_dict(comfy_quant_tensor)
    except Exception:
        return comfy_quant_tensor, False

    if "params" not in config:
        return comfy_quant_tensor, False

    # Fix nested params structure
    params = config.pop("params")
    if isinstance(params, dict):
        # Move group_size to root level
        if "group_size" in params:
            config["group_size"] = params["group_size"]
        # Move any other params to root (future-proofing)
        for key, value in params.items():
            if key != "group_size" and key not in config:
                config[key] = value

    return dict_to_tensor(config), True

def parse_add_keys_string(add_keys_str: str) -> Dict[str, Any]:
    """
    Parse a Python-like key:value string into a dictionary.

    Format: "'key': 'string_value', 'key2': true, 'key3': 123"
    - Quoted values ('value') → string
    - true/false → bool
    - Numbers → int/float

    Args:
        add_keys_str: String in format "'key': value, 'key2': value2"

    Returns:
        Dictionary of parsed key-value pairs
    """
    result = {}
    if not add_keys_str or not add_keys_str.strip():
        return result

    pattern = r"'([^']+)':\s*([^,]+?)(?:,|$)"
    matches = re.findall(pattern, add_keys_str.strip())

    for key, value in matches:
        value = value.strip()
        # Parse value type
        if value.startswith("'") and value.endswith("'"):
            # String value
            result[key] = value[1:-1]
        elif value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        else:
            # Try to parse as number
            try:
                if "." in value:
                    result[key] = float(value)
                else:
                    result[key] = int(value)
            except ValueError:
                # Fallback to string
                result[key] = value

    return result

def edit_comfy_quant(
    input_file: str,
    output_file: str,
    remove_keys: Optional[List[str]] = None,
    add_keys_str: Optional[str] = None,
    layer_filter: Optional[str] = None,
    save_quant_metadata: bool = False,
):
    """
    Edit comfy_quant layer configurations and _quantization_metadata in a model.

    This loads a safetensors model and modifies both:
    1. The .comfy_quant metadata tensors (per-layer JSON configs)
    2. The _quantization_metadata header entry (aggregated layer configs)

    Both are edited in sync when present. Useful for batch editing layer
    configurations without re-quantizing.

    Args:
        input_file: Path to input safetensors file
        output_file: Path to output safetensors file
        remove_keys: List of key names to remove from configs
        add_keys_str: Python-like string of key:value pairs to add
        layer_filter: Regex pattern to filter which layers to edit (None = all)
        save_quant_metadata: If True, generate _quantization_metadata from tensors
    """
    info("ComfyQuant Layer & Metadata Editor")
    info("=" * 60)
    info(f"Input:  {input_file}")
    info(f"Output: {output_file}")

    # Parse add_keys string
    add_keys = parse_add_keys_string(add_keys_str) if add_keys_str else {}

    if remove_keys:
        info(f"Keys to remove: {remove_keys}")
    if add_keys:
        info(f"Keys to add: {add_keys}")
    if layer_filter:
        info(f"Layer filter: {layer_filter}")
        try:
            layer_regex = re.compile(layer_filter)
        except re.error as e:
            error(f"FATAL: Invalid regex pattern '{layer_filter}': {e}")
            return
    else:
        layer_regex = None

    info("-" * 60)

    # Load all tensors and existing header metadata
    tensors = {}
    existing_metadata: Optional[Dict[str, str]] = None
    with safe_open(input_file, framework="pt", device="cpu") as f:
        existing_metadata = f.metadata()
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    # Parse _quantization_metadata from header if present
    quant_metadata: Optional[Dict[str, Any]] = None
    quant_metadata_modified = False
    if existing_metadata and "_quantization_metadata" in existing_metadata:
        try:
            quant_metadata = json.loads(existing_metadata["_quantization_metadata"])
            info(
                f"Found _quantization_metadata header with "
                f"{len(quant_metadata.get('layers', {}))} layer entries"
            )
        except json.JSONDecodeError as e:
            warning(f"  WARNING: Failed to parse _quantization_metadata: {e}")
            quant_metadata = None

    # Track statistics for .comfy_quant tensors
    edited_count = 0
    skipped_filter = 0
    skipped_no_change = 0
    total_comfy_quant = 0
    keys_removed: Dict[str, int] = {}
    keys_added: Dict[str, int] = {}

    # Track statistics for metadata entries
    metadata_edited_count = 0
    metadata_keys_removed: Dict[str, int] = {}
    metadata_keys_added: Dict[str, int] = {}

    # Process .comfy_quant tensors
    for key in list(tensors.keys()):
        if not key.endswith(".comfy_quant"):
            continue

        total_comfy_quant += 1
        base_name = key[:-12]  # Remove '.comfy_quant' suffix

        # Apply layer filter
        if layer_regex and not layer_regex.search(base_name):
            skipped_filter += 1
            continue

        # Decode existing config from tensor
        try:
            config = tensor_to_dict(tensors[key])
        except Exception as e:
            warning(f"  WARNING: Failed to decode {key}: {e}")
            continue

        original_config = config.copy()

        # Remove keys from tensor config
        if remove_keys:
            for k in remove_keys:
                if k in config:
                    del config[k]
                    keys_removed[k] = keys_removed.get(k, 0) + 1

        # Add keys to tensor config
        if add_keys:
            for k, v in add_keys.items():
                if k not in config or config[k] != v:
                    config[k] = v
                    keys_added[k] = keys_added.get(k, 0) + 1

        # Check if tensor config changed
        tensor_changed = config != original_config

        # Also edit corresponding metadata entry if present
        if quant_metadata and "layers" in quant_metadata:
            meta_layers = quant_metadata["layers"]
            if base_name in meta_layers:
                meta_entry = meta_layers[base_name]
                original_meta = meta_entry.copy()

                # Remove keys from metadata entry
                if remove_keys:
                    for k in remove_keys:
                        if k in meta_entry:
                            del meta_entry[k]
                            metadata_keys_removed[k] = (
                                metadata_keys_removed.get(k, 0) + 1
                            )

                # Add keys to metadata entry
                if add_keys:
                    for k, v in add_keys.items():
                        if k not in meta_entry or meta_entry[k] != v:
                            meta_entry[k] = v
                            metadata_keys_added[k] = metadata_keys_added.get(k, 0) + 1

                # Track if metadata changed
                if meta_entry != original_meta:
                    quant_metadata_modified = True
                    metadata_edited_count += 1

        # If tensor config didn't change, skip re-encoding
        if not tensor_changed:
            skipped_no_change += 1
            continue

        # Re-encode and store updated tensor
        tensors[key] = dict_to_tensor(config)
        edited_count += 1

    # Auto-create .comfy_quant tensors for layers in metadata but missing tensors
    created_count = 0
    if quant_metadata and "layers" in quant_metadata:
        meta_layers = quant_metadata["layers"]
        for layer_name, meta_entry in meta_layers.items():
            comfy_quant_key = f"{layer_name}.comfy_quant"

            # Skip if tensor already exists
            if comfy_quant_key in tensors:
                continue

            # Apply layer filter
            if layer_regex and not layer_regex.search(layer_name):
                continue

            # Check if the base weight tensor exists (sanity check)
            weight_key = f"{layer_name}.weight"
            if weight_key not in tensors:
                continue

            # Build config from metadata entry
            config = {}
            if "format" in meta_entry:
                config["format"] = meta_entry["format"]
            else:
                # Try to infer format from weight dtype
                weight = tensors[weight_key]
                if weight.dtype == torch.float8_e4m3fn:
                    config["format"] = "float8_e4m3fn"
                elif weight.dtype == torch.int8:
                    # Default to blockwise for INT8 if unknown
                    config["format"] = "int8_blockwise"
                else:
                    continue  # Can't determine format

            # Copy relevant keys from metadata
            for key in ["group_size", "full_precision_matrix_mult"]:
                if key in meta_entry:
                    config[key] = meta_entry[key]

            # Apply add_keys if specified
            if add_keys:
                for k, v in add_keys.items():
                    config[k] = v
                    keys_added[k] = keys_added.get(k, 0) + 1

            # Create the tensor
            tensors[comfy_quant_key] = dict_to_tensor(config)
            created_count += 1
            verbose(f"  Created: {comfy_quant_key} (format={config.get('format', 'unknown')})")

    # Summary for .comfy_quant tensors
    info("-" * 60)
    info("Edit Summary (.comfy_quant tensors):")
    info(f"  Total tensors:              {total_comfy_quant}")
    info(f"  Edited:                     {edited_count}")
    if created_count > 0:
        info(f"  Created from metadata:      {created_count}")
    if skipped_filter > 0:
        info(f"  Skipped (filter):           {skipped_filter}")
    if skipped_no_change > 0:
        info(f"  Skipped (no change):        {skipped_no_change}")
    if keys_removed:
        info("  Keys removed:")
        for k, count in sorted(keys_removed.items()):
            info(f"    {k}: {count} layers")
    if keys_added:
        info("  Keys added:")
        for k, count in sorted(keys_added.items()):
            info(f"    {k}: {count} layers")

    # Summary for _quantization_metadata header
    if quant_metadata:
        info("-" * 60)
        info("Edit Summary (_quantization_metadata header):")
        total_meta_layers = len(quant_metadata.get("layers", {}))
        info(f"  Total layer entries:        {total_meta_layers}")
        info(f"  Edited:                     {metadata_edited_count}")
        if metadata_keys_removed:
            info("  Keys removed:")
            for k, count in sorted(metadata_keys_removed.items()):
                info(f"    {k}: {count} entries")
        if metadata_keys_added:
            info("  Keys added:")
            for k, count in sorted(metadata_keys_added.items()):
                info(f"    {k}: {count} entries")
    else:
        info("-" * 60)
        info("Note: No _quantization_metadata header found in input file")

    # Generate metadata from .comfy_quant tensors if requested
    if save_quant_metadata:
        info("-" * 60)
        info("Generating _quantization_metadata from .comfy_quant tensors...")
        generated_layers = {}
        for key in tensors.keys():
            if not key.endswith(".comfy_quant"):
                continue
            base_name = key[:-12]  # Remove '.comfy_quant' suffix
            try:
                config = tensor_to_dict(tensors[key])
                # Build metadata entry from config
                meta_entry = {"format": config.get("format", "unknown")}
                if "group_size" in config:
                    meta_entry["group_size"] = config["group_size"]
                if config.get("full_precision_matrix_mult"):
                    meta_entry["full_precision_matrix_mult"] = True
                generated_layers[base_name] = meta_entry
            except Exception as e:
                warning(f"  WARNING: Failed to parse {key}: {e}")

        if generated_layers:
            quant_metadata = {"format_version": "1.0", "layers": generated_layers}
            quant_metadata_modified = True
            info(f"  Generated metadata for {len(generated_layers)} layers")
        else:
            info("  No .comfy_quant tensors found to generate metadata from")

    info("-" * 60)

    # Prepare save kwargs with updated metadata
    save_kwargs: Dict[str, Any] = {}
    if quant_metadata and quant_metadata_modified:
        # Preserve any other existing metadata keys and update quant metadata
        output_metadata: Dict[str, str] = {}
        if existing_metadata:
            # Copy all existing metadata
            output_metadata = dict(existing_metadata)
        # Update the quantization metadata with our edits
        output_metadata["_quantization_metadata"] = json.dumps(quant_metadata)
        save_kwargs["metadata"] = output_metadata
        info("  Updated _quantization_metadata in output file")
    elif existing_metadata:
        # No quant_metadata changes but preserve existing metadata as-is
        save_kwargs["metadata"] = existing_metadata

    # Save output
    info(f"\nSaving to {output_file}...")
    try:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        # Normalize any 1-element scale tensors to scalars
        tensors, normalized_count = normalize_tensorwise_scales(tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            info(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(tensors, output_file, **save_kwargs)

        info("Edit complete!")
    except Exception as e:
        error(f"FATAL: Error saving file '{output_file}': {e}")
        return

def should_skip_layer_for_performance(
    tensor: torch.Tensor, block_size: int
) -> Tuple[bool, str]:
    """
    Check if a layer should be skipped based on performance heuristics.

    Args:
        tensor: Weight tensor to evaluate
        block_size: Block size for quantization

    Returns:
        Tuple of (should_skip, reason)
    """
    if tensor.ndim != 2:
        return True, "not 2D"

    rows, cols = tensor.shape

    # Skip if any dimension is smaller than block_size
    if rows < block_size or cols < block_size:
        return True, f"dimension smaller than block_size ({block_size})"

    # Skip if dimensions are not divisible by block_size
    if rows % block_size != 0 or cols % block_size != 0:
        return True, f"dimensions not divisible by block_size ({block_size})"

    return False, ""
