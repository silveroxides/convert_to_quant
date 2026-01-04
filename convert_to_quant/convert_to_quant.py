import argparse
import os
import re
import sys
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple, Optional, Any, List
from tqdm import tqdm

# fnmatch removed - using regex for layer config patterns
import gc
import math
import json
from torch.optim import AdamW, RAdam
from .comfy.quant_ops import BlockWiseINT8Layout
from .pinned_transfer import transfer_to_gpu_pinned, get_pinned_transfer_stats, set_verbose as set_pinned_verbose

# --- Constants and Configuration ---
torch.set_printoptions(precision=8)
AVOID_KEY_NAMES = [
    "norm",
    "bias",
    "embed_tokens",
    "lm_head",
    "shared",
    "patch_embedding",
    "audio_model.patch_embedding",
    "ref_conv",
    "control_adapter",
    "motion_encoder.enc.net_app",
    "face_encoder.conv",
    "pose_patch_embedding",
    "motion_encoder.enc.fc",
    "img_emb.proj",
    "k_norm",
    "q_norm",
    "motion_encoder.dec",
    "head.modulation",
    "casual_audio_encoder",
    "cond_encoder",
    "frame_packer",
    "norm_k",
    "norm_q",
    "tekken_model",
    "multi_modal_projector",
    "patch_conv",
    "ln_pre",
    "input_layernorm",
    "attention_norm",
    "post_attention_layernorm",
]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
VISUAL_AVOID_KEY_NAMES = ["mlp.down_proj", "mlp.up_proj", "mlp.gate_proj"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
HUNYUAN_AVOID_KEY_NAMES = [
    "layernorm",
    "img_attn_k_norm",
    "img_attn_q_norm",
    "txt_attn_k_norm",
    "txt_attn_q_norm",
    "norm1",
    "norm2",
    "vision_in.proj.0",
    "vision_in.proj.4",
    "img_in.proj",
    "cond_type_embedding",
]
ZIMAGE_AVOID_KEY_NAMES = [
    "cap_embedder.0",
    "cap_pad_token",
    "attention_norm1",
    "attention_norm2",
    "ffn_norm1",
    "ffn_norm2",
    "k_norm",
    "q_norm",
    "x_pad_token",
]
FLUX2_LAYER_KEYNAMES = [
    "stream_modulation",
    "guidance_in",
    "time_in",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "final_layer",
    "img_in",
    "txt_in",
]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
    "txt_in",
]
NERF_LAYER_KEYNAMES_SMALL = [
    "distilled_guidance_layer",
    "nerf_blocks",
    "nerf_image_embedder",
]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv", "__x0__"]
WAN_LAYER_KEYNAMES = [
    "text_embedding",
    "time_embedding",
    "audio_model.text_embedding",
    "casual_audio_encoder",
    "frame_packer",
    "trainable_cond_mask",
    "cond_encoder",
    "audio_model.time_embedding",
    "time_projection",
    "video_model.time_projection",
    "head.head",
    "face_encoder.out_proj",
    "face_adapter",
    "audio_injector",
]
QWEN_LAYER_KEYNAMES = [
    "time_text_embed",
    "img_in",
    "norm_out",
    "proj_out",
    "transformer_blocks.0.img_mod.1",
    "txt_in",
]
ZIMAGE_LAYER_KEYNAMES = [
    "x_embedder",
    "clip_text_pooled_proj",
    "final_layer",
    "cap_embedder.1",
    "adaLN_modulation",
    "t_embedder",
    "time_text_embed",
]
ZIMAGE_REFINER_LAYER_KEYNAMES = ["context_refiner", "noise_refiner"]
TARGET_FP8_DTYPE = torch.float8_e4m3fn
TARGET_INT8_DTYPE = torch.int8
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

# FP8 constants
FP8_MIN = float(torch.finfo(TARGET_FP8_DTYPE).min)
FP8_MAX = float(torch.finfo(TARGET_FP8_DTYPE).max)
FP8_MIN_POS = float(torch.finfo(TARGET_FP8_DTYPE).tiny)

# INT8 constants (using symmetric range [-127, 127] for symmetric quantization)
INT8_MIN = int(torch.iinfo(TARGET_INT8_DTYPE).min)  # -128
INT8_MAX = int(torch.iinfo(TARGET_INT8_DTYPE).max)  # 127
INT8_SYMMETRIC_MAX = min(abs(INT8_MIN), INT8_MAX)  # 127 (symmetric range)


def dict_to_tensor(data_dict):
    json_str = json.dumps(data_dict)
    byte_data = json_str.encode("utf-8")
    tensor_data = torch.tensor(list(byte_data), dtype=torch.uint8)
    return tensor_data


def tensor_to_dict(tensor_data):
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode("utf-8")
    data_dict = json.loads(json_str)
    return data_dict


# Valid quantization formats (maps to QUANT_ALGOS in quant_ops.py)
VALID_QUANT_FORMATS = {
    "float8_e4m3fn",
    "float8_e4m3fn_rowwise",
    "float8_e4m3fn_blockwise",
    "float8_e4m3fn_block3d",
    "int8_blockwise",
}

# Global config: normalize 1-element scale arrays to scalars (set from CLI)
NORMALIZE_SCALES_ENABLED = True


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




def pattern_specificity(pattern: str) -> tuple:
    """
    Calculate specificity score for a regex pattern.
    Higher score = more specific pattern.

    Priority rules:
    1. Tier 0 (highest): Pattern has explicit numbers (``\\d`` or literal digits) AND 8+ literal chars
    2. Tier 1: Longer literal length

    Returns:
        (tier, score) tuple for sorting - lower tier and higher score = more specific
    """
    if pattern.startswith("_"):
        return (999, 0)  # Special keys like _default have lowest priority

    # Count literal characters (exclude regex metacharacters)
    # Remove common regex patterns to get approximate literal length
    literal_pattern = re.sub(r"\\.|\[.*?\]|\(.*?\)|[.*+?^${}|\\]", "", pattern)
    literal_len = len(literal_pattern)

    # Check if pattern has explicit numbers (literal digits or \d patterns)
    has_number = bool(re.search(r"\d|\\d", pattern))

    # Tier 0: numbers + 8+ literal chars (very specific layers)
    tier = 0 if (has_number and literal_len >= 8) else 1

    return (tier, literal_len)


def load_layer_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate layer configuration from JSON file.

    Patterns are now regex patterns (using re.search matching).

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config has invalid format, unknown quant format, or invalid regex

    Returns:
        Validated config dict with compiled regex patterns
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Layer config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = json.load(f)

    if not isinstance(config, dict):
        raise ValueError(
            f"Layer config must be a JSON object, got {type(config).__name__}"
        )

    # Validate each entry and compile regex patterns
    compiled_patterns = {}
    for key, settings in config.items():
        if key.startswith("_"):
            # Validate _default has valid format if format is specified
            if key == "_default" and isinstance(settings, dict):
                if "format" in settings:
                    fmt = settings["format"]
                    if not fmt:  # Empty string check
                        raise ValueError(
                            "_default has empty 'format' field. Use skip:true to skip, or specify a valid format."
                        )
                    if fmt not in VALID_QUANT_FORMATS:
                        raise ValueError(
                            f"_default has invalid format '{fmt}'. Valid formats: {sorted(VALID_QUANT_FORMATS)}"
                        )
            continue

        if not isinstance(settings, dict):
            raise ValueError(
                f"Layer config entry '{key}' must be an object, got {type(settings).__name__}"
            )

        # Validate regex pattern
        try:
            compiled_patterns[key] = re.compile(key)
        except re.error as e:
            raise ValueError(
                f"Layer config entry '{key}' has invalid regex pattern: {e}"
            )

        # skip:true entries don't need format
        if settings.get("skip", False):
            continue

        if "format" not in settings:
            raise ValueError(
                f"Layer config entry '{key}' missing required 'format' field (or set skip:true)"
            )

        fmt = settings["format"]
        if not fmt:  # Empty string check
            raise ValueError(
                f"Layer config entry '{key}' has empty 'format' field. Use skip:true to skip, or specify a valid format."
            )
        if fmt not in VALID_QUANT_FORMATS:
            raise ValueError(
                f"Layer config entry '{key}' has invalid format '{fmt}'. "
                f"Valid formats: {sorted(VALID_QUANT_FORMATS)}"
            )

    # Store compiled patterns in config for reuse
    config["_compiled_patterns"] = compiled_patterns

    print(
        f"Loaded layer config with {len([k for k in config if not k.startswith('_')])} layer patterns (regex mode)"
    )
    return config


def get_layer_settings(
    layer_key: str, config: Dict[str, Any], fullmatch: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Find the most specific matching config entry for a layer using regex.

    Args:
        layer_key: Full layer name (e.g., "double_blocks.0.img_attn.proj.weight")
        config: Layer config dict (with _compiled_patterns from load_layer_config)
        fullmatch: If True, use re.fullmatch (pattern must match entire string).
                   If False (default), use re.search (pattern matches anywhere).

    Returns:
        Settings dict for the layer, or None if no match and no _default
    """
    # Strip .weight suffix for matching
    base_key = layer_key[:-7] if layer_key.endswith(".weight") else layer_key

    # Get compiled patterns (or compile on demand for backwards compatibility)
    compiled_patterns = config.get("_compiled_patterns", {})

    # Find all matching patterns using regex
    matches = []
    for pattern, settings in config.items():
        if pattern.startswith("_"):
            continue

        # Use pre-compiled pattern if available, otherwise compile
        regex = compiled_patterns.get(pattern)
        if regex is None:
            try:
                regex = re.compile(pattern)
            except re.error:
                continue  # Skip invalid patterns

        # Use fullmatch or search based on flag
        match_result = (
            regex.fullmatch(base_key) if fullmatch else regex.search(base_key)
        )
        if match_result:
            specificity = pattern_specificity(pattern)
            matches.append((specificity, pattern, settings))

    if matches:
        # Sort by (tier asc, score desc) - most specific first
        matches.sort(key=lambda x: (x[0][0], -x[0][1]))
        _, best_pattern, best_settings = matches[0]
        return best_settings

    # Fall back to _default if present
    return config.get("_default")


def generate_config_template(input_file: str, output_path: str, block_size: int = 128):
    """
    Generate a JSON config template from model, with viable layers and empty format.

    Args:
        input_file: Path to input safetensors file
        output_path: Path to write the template JSON
        block_size: Block size to check divisibility against (for block-based formats)

    Returns:
        Tuple of (viable_count, skipped_count)
    """
    print(f"Generating layer config template from: {input_file}")
    print("-" * 60)

    # Load tensor metadata (not full tensors)
    try:
        with safe_open(input_file, framework="pt", device="cpu") as f:
            all_keys = f.keys()
            weight_keys = [k for k in all_keys if k.endswith(".weight")]
    except Exception as e:
        raise RuntimeError(f"Error reading model file: {e}")

    print(f"Found {len(weight_keys)} weight tensors")

    config = {"_default": {"format": ""}, "_exclusions": []}

    viable_count = 0
    skipped_count = 0
    skipped_reasons = {}

    with safe_open(input_file, framework="pt", device="cpu") as f:
        for key in tqdm(weight_keys, desc="Analyzing layers"):
            tensor = f.get_tensor(key)
            base_name = key[:-7]  # Remove .weight suffix

            # Check viability
            skip_reason = None
            if tensor.numel() == 0:
                skip_reason = "empty tensor"
            elif tensor.ndim != 2:
                skip_reason = f"non-2D ({tensor.ndim}D)"
            elif tensor.shape[0] < 16 or tensor.shape[1] < 16:
                skip_reason = f"too small ({tensor.shape})"

            if skip_reason:
                skipped_count += 1
                skipped_reasons.setdefault(skip_reason, []).append(base_name)
                continue

            # Add to template with shape info as comment
            config[base_name] = {"format": "", "_shape": list(tensor.shape)}
            viable_count += 1

    # Write template
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)

    print("-" * 60)
    print("Template Summary:")
    print(f"  Viable layers:  {viable_count}")
    print(f"  Skipped layers: {skipped_count}")
    if skipped_reasons:
        print("  Skip reasons:")
        for reason, layers in skipped_reasons.items():
            print(f"    {reason}: {len(layers)} layers")
    print(f"\nTemplate written to: {output_path}")
    print(
        "Edit the template to set 'format' for each layer (float8_e4m3fn, int8_blockwise, etc.)"
    )

    return viable_count, skipped_count


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
    BLOCK_BASED_FORMATS = (
        "int8_blockwise",
        "float8_e4m3fn_blockwise",
    )

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

    # Split by comma, but handle quoted strings properly
    # Use regex to match 'key': value patterns
    import re

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
    """
    from safetensors import safe_open

    print("ComfyQuant Layer & Metadata Editor")
    print("=" * 60)
    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")

    # Parse add_keys string
    add_keys = parse_add_keys_string(add_keys_str) if add_keys_str else {}

    if remove_keys:
        print(f"Keys to remove: {remove_keys}")
    if add_keys:
        print(f"Keys to add: {add_keys}")
    if layer_filter:
        print(f"Layer filter: {layer_filter}")
        try:
            layer_regex = re.compile(layer_filter)
        except re.error as e:
            print(f"FATAL: Invalid regex pattern '{layer_filter}': {e}")
            return
    else:
        layer_regex = None

    print("-" * 60)

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
            print(
                f"Found _quantization_metadata header with "
                f"{len(quant_metadata.get('layers', {}))} layer entries"
            )
        except json.JSONDecodeError as e:
            print(f"  WARNING: Failed to parse _quantization_metadata: {e}")
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
            print(f"  WARNING: Failed to decode {key}: {e}")
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

    # Summary for .comfy_quant tensors
    print("-" * 60)
    print("Edit Summary (.comfy_quant tensors):")
    print(f"  Total tensors:              {total_comfy_quant}")
    print(f"  Edited:                     {edited_count}")
    if skipped_filter > 0:
        print(f"  Skipped (filter):           {skipped_filter}")
    if skipped_no_change > 0:
        print(f"  Skipped (no change):        {skipped_no_change}")
    if keys_removed:
        print("  Keys removed:")
        for k, count in sorted(keys_removed.items()):
            print(f"    {k}: {count} layers")
    if keys_added:
        print("  Keys added:")
        for k, count in sorted(keys_added.items()):
            print(f"    {k}: {count} layers")

    # Summary for _quantization_metadata header
    if quant_metadata:
        print("-" * 60)
        print("Edit Summary (_quantization_metadata header):")
        total_meta_layers = len(quant_metadata.get("layers", {}))
        print(f"  Total layer entries:        {total_meta_layers}")
        print(f"  Edited:                     {metadata_edited_count}")
        if metadata_keys_removed:
            print("  Keys removed:")
            for k, count in sorted(metadata_keys_removed.items()):
                print(f"    {k}: {count} entries")
        if metadata_keys_added:
            print("  Keys added:")
            for k, count in sorted(metadata_keys_added.items()):
                print(f"    {k}: {count} entries")
    else:
        print("-" * 60)
        print("Note: No _quantization_metadata header found in input file")

    # Generate metadata from .comfy_quant tensors if requested
    if save_quant_metadata:
        print("-" * 60)
        print("Generating _quantization_metadata from .comfy_quant tensors...")
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
                print(f"  WARNING: Failed to parse {key}: {e}")
        
        if generated_layers:
            quant_metadata = {"format_version": "1.0", "layers": generated_layers}
            quant_metadata_modified = True
            print(f"  Generated metadata for {len(generated_layers)} layers")
        else:
            print("  No .comfy_quant tensors found to generate metadata from")

    print("-" * 60)

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
        print("  Updated _quantization_metadata in output file")
    elif existing_metadata:
        # No quant_metadata changes but preserve existing metadata as-is
        save_kwargs["metadata"] = existing_metadata

    # Save output
    print(f"\nSaving to {output_file}...")
    try:
        os.makedirs(
            os.path.dirname(output_file) if os.path.dirname(output_file) else ".",
            exist_ok=True,
        )
        # Normalize any 1-element scale tensors to scalars
        tensors, normalized_count = normalize_tensorwise_scales(tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            print(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(tensors, output_file, **save_kwargs)

        print("Edit complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
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

    # Skip highly rectangular layers (aspect ratio > 4.0 with small dimension < 5120)
    aspect_ratio = max(rows, cols) / max(min(rows, cols), 1)
    small_dim = min(rows, cols)
    if aspect_ratio > 4.0 and small_dim < 5120:
        return (
            True,
            f"highly rectangular (aspect ratio {aspect_ratio:.1f}, small dim {small_dim})",
        )

    # Skip small-ish layers (< 15M params, aspect ratio < 2.0)
    num_params = rows * cols
    if num_params < 15_000_000 and aspect_ratio < 2.0:
        return (
            True,
            f"small layer ({num_params:,} params, aspect ratio {aspect_ratio:.1f})",
        )

    # Skip layers with max dimension < 4096
    if max(rows, cols) < 4096:
        return True, f"max dimension < 4096 ({max(rows, cols)})"

    return False, ""


class LearnedRoundingConverter:
    """
    Implements advanced quantization using learned adaptive rounding.
    Provides a highly effective optimization strategy.
    Supports both FP8 and INT8 quantization formats.
    """

    def __init__(
        self,
        optimizer="original",
        num_iter=500,
        top_p=0.01,
        min_k=1,
        max_k=16,
        scaling_mode="tensor",
        block_size=64,
        full_matrix=False,
        target_format="fp8",
        no_learned_rounding=False,
        lr_schedule="adaptive",
        lr_gamma=0.99,
        lr_patience=50,
        lr_factor=0.5,
        lr_min=1e-8,
        lr_cooldown=0,
        lr_threshold=0.0,
        lr_adaptive_mode="simple-reset",
        lr_shape_influence=1.0,
        lr_threshold_mode="rel",
        early_stop_loss=1e-8,
        early_stop_lr=1e-10,
        early_stop_stall=1000,
        **kwargs,
    ):
        self.num_iter = num_iter
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.block_size = block_size
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.optimizer_choice = optimizer
        self.full_matrix = full_matrix
        self.target_format = target_format
        self.no_learned_rounding = no_learned_rounding
        self.optimizer_kwargs = kwargs

        # LR schedule configuration (for 'original' optimizer)
        self.lr_schedule = lr_schedule
        self.lr_gamma = lr_gamma
        self.lr_patience = lr_patience
        self.lr_factor = lr_factor
        self.lr_min = lr_min
        self.lr_cooldown = lr_cooldown
        self.lr_threshold = lr_threshold
        self.lr_adaptive_mode = lr_adaptive_mode

        # Shape-adaptive LR (for plateau schedule)
        self.lr_shape_influence = lr_shape_influence
        self.lr_threshold_mode = lr_threshold_mode  # 'rel' or 'abs'

        # Early stopping thresholds
        self.early_stop_loss = early_stop_loss
        self.early_stop_lr = early_stop_lr
        self.early_stop_stall = early_stop_stall

        # INT8 always uses block-wise scaling
        if target_format == "int8":
            scaling_mode = "block"
        # Normalize block3d alias to block
        if scaling_mode == "block3d":
            scaling_mode = "block"
        self.scaling_mode = scaling_mode

        # Set format-specific max values and dtype
        if self.target_format == "int8":
            # INT8 uses integer symmetric range [-127, 127]
            self.target_dtype = TARGET_INT8_DTYPE
            self.f8_max_val = None  # Not applicable to INT8
        else:
            # FP8 uses floating point range constants
            self.target_dtype = TARGET_FP8_DTYPE
            self.f8_max_val = FP8_MAX  # Used in FP8 scale calculation and clamping

        print(f"LearnedRoundingConverter initialized on device: {self.device}")
        print(f"  - Target format: {self.target_format}")
        print(
            f"  - Using optimizer: '{self.optimizer_choice}'"
            + (" (disabled - simple quant)" if self.no_learned_rounding else "")
        )
        if self.optimizer_choice == "original":
            print(f"  - LR schedule: {self.lr_schedule}")
        print(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode in ("block", "block2d", "block3d"):
            print(f"    - Block size: {self.block_size}")

    def _optimize_adamw(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """FP8 optimization using AdamW optimizer with manual LR scheduling."""
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = AdamW([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (AdamW-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()

            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            # 'adaptive' mode: fixed LR (AdamW handles momentum internally)

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr < self.early_stop_lr:
                    print("      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping early.")
                elif best_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_radam(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """FP8 optimization using RAdam optimizer with manual LR scheduling."""
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        delta = torch.zeros_like(W_rounded, requires_grad=True)
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = RAdam([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (RAdam-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()
            W_q_refined = W_rounded + delta

            current_dq = W_q_refined / scale
            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()

            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            # 'adaptive' mode: fixed LR (RAdam handles momentum internally)

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr < self.early_stop_lr:
                    print("      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping early.")
                elif best_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping early.")
                break

        pbar.close()
        return W_rounded + best_delta

    def _optimize_original(
        self,
        W_float32: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        plateau_counter = 0  # For plateau schedule
        cooldown_counter = 0  # For plateau cooldown
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        if W_float32.shape[0] == W_float32.shape[1]:
            small_mult = 0.95
        else:
            small_mult = 1.0

        schedule_name = self.lr_schedule

        # Shape-aware plateau parameters
        rows, cols = W_float32.shape
        aspect_ratio = max(rows, cols) / min(rows, cols)

        if schedule_name == "plateau" and self.lr_shape_influence > 0:
            # Scale factor based on aspect ratio, modulated by influence
            # influence=1.0: full effect, influence=0.0: no effect (use raw values)
            # Elongated tensors need MORE AGGRESSIVE decay (lower factor)
            ar_factor = math.sqrt(aspect_ratio)  # e.g., 1.0 for square, 2.0 for AR=4
            blend = self.lr_shape_influence

            # Keep patience unchanged per user feedback
            effective_patience = self.lr_patience

            # More aggressive factor for elongated tensors: factor^ar_factor makes it smaller
            # E.g., 0.92^2 = 0.846 for AR=4, 0.92^2.45 = 0.808 for AR=6
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend

            # Cooldown unchanged
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing (Original-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            with torch.no_grad():
                current_dq = W_q_refined / scale
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()
            # Check if improvement exceeds threshold (supports rel/abs mode like PyTorch ReduceLROnPlateau)
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                    # Relative: significant if loss < best * (1 - threshold)
                    improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:  # 'abs'
                    # Absolute: significant if improvement > threshold
                    improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            # Store counter before potential reset (for no-reset adaptive mode)
            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss = current_loss
                best_tensor = W_q_refined.clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
                # no-reset mode: worse_loss_counter preserved for tier calculation
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update based on schedule
            if schedule_name == "exponential":
                # ExponentialLR: lr = lr * gamma per step
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
                # ReduceLROnPlateau with cooldown (shape-aware)
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= effective_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - tier-based schedule
                # For no-reset mode, use counter value before reset for tier calculation
                counter_for_tier = (
                    prev_worse_counter
                    if (improved and self.lr_adaptive_mode == "no-reset")
                    else worse_loss_counter
                )

                if improved and counter_for_tier < 50:
                    curr_lr = min(curr_lr * (1.25 * small_mult), 100.0)
                elif improved and counter_for_tier >= 50 and counter_for_tier < 75:
                    curr_lr = min(curr_lr * (1.375 * small_mult), 100.0)
                elif improved and counter_for_tier >= 75 and counter_for_tier < 100:
                    curr_lr = min(curr_lr * (1.5 * small_mult), 100.0)
                elif improved and counter_for_tier >= 100 and counter_for_tier < 125:
                    curr_lr = min(curr_lr * (1.75 * small_mult), 100.0)
                elif improved and counter_for_tier >= 125 and counter_for_tier < 150:
                    curr_lr = min(curr_lr * (2.0 * small_mult), 100.0)
                elif improved and counter_for_tier >= 150 and counter_for_tier < 200:
                    curr_lr = min(curr_lr * (2.25 * small_mult), 100.0)
                elif improved and counter_for_tier >= 200 and counter_for_tier < 250:
                    curr_lr = min(curr_lr * (2.5 * small_mult), 100.0)
                elif improved and counter_for_tier >= 250 and counter_for_tier < 300:
                    curr_lr = min(curr_lr * (2.75 * small_mult), 100.0)
                elif improved and counter_for_tier >= 300:
                    curr_lr = min(curr_lr * (3.0 * small_mult), 100.0)
                elif not improved and worse_loss_counter < 26:
                    curr_lr = max(curr_lr * (0.95 * small_mult), 9e-8)
                elif worse_loss_counter >= 26 and worse_loss_counter < 51:
                    curr_lr = max(curr_lr * (0.97 * small_mult), 8e-8)
                elif worse_loss_counter >= 51 and worse_loss_counter < 76:
                    curr_lr = max(curr_lr * (0.985 * small_mult), 7e-8)
                elif worse_loss_counter >= 76 and worse_loss_counter < 101:
                    curr_lr = max(curr_lr * (0.9875 * small_mult), 6e-8)
                elif worse_loss_counter >= 101 and worse_loss_counter < 151:
                    curr_lr = max(curr_lr * (0.98875 * small_mult), 5e-8)
                elif worse_loss_counter >= 151 and worse_loss_counter < 201:
                    curr_lr = max(curr_lr * (0.99 * small_mult), 4e-8)
                elif worse_loss_counter >= 201 and worse_loss_counter < 251:
                    curr_lr = max(curr_lr * (0.99125 * small_mult), 3e-8)
                elif worse_loss_counter >= 251 and worse_loss_counter < 301:
                    curr_lr = max(curr_lr * (0.9925 * small_mult), 2e-8)
                else:  # worse_loss_counter >= 301
                    curr_lr = max(curr_lr * (0.995 * small_mult), 5e-9)

                # Reset counter after boost in no-reset mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix(
                {
                    "loss": f"{current_loss:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                }
            )

            # Early stopping conditions (configurable thresholds)
            if (
                current_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if (
                    curr_lr < self.early_stop_lr * 1.75
                    and worse_loss_counter > self.early_stop_stall * 0.95
                ):
                    print(
                        "      - Loss has stalled and learning rate has bottomed out. Stopping."
                    )
                elif (
                    current_loss < self.early_stop_loss
                    and curr_lr < self.early_stop_lr * 1.75
                ):
                    print(
                        "      - Learning Rate has bottomed out and loss is negligible. Stopping."
                    )
                elif (
                    worse_loss_counter > self.early_stop_stall * 0.95
                    and current_loss > self.early_stop_loss * 2
                ):
                    print("      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping.")
                elif curr_lr < self.early_stop_lr:
                    print("      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping.")
                break

            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * (grad_direction * scale)

        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(
        self, W_orig: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            print("  - Tensor is all zeros, skipping optimization.")
            quantized_tensor = torch.zeros_like(W_float32, dtype=self.target_dtype)
            dequant_scale = None

            if W_float32.ndim == 2:
                out_features, in_features = W_float32.shape

                if self.target_format == "int8":
                    # INT8 uses 2D block scaling (M//block_size, N//block_size)
                    num_blocks_m = out_features // self.block_size
                    num_blocks_n = in_features // self.block_size
                    dequant_scale = torch.ones(
                        num_blocks_m,
                        num_blocks_n,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                elif self.scaling_mode == "row":
                    # Row-wise: one scale per row
                    dequant_scale = torch.ones(
                        out_features, device=self.device, dtype=SCALE_DTYPE
                    )
                elif (
                    self.scaling_mode in ("block", "block2d")
                    and out_features % self.block_size == 0
                    and in_features % self.block_size == 0
                ):
                    # 2D block-wise: (M//bs, N//bs) - 'block' is primary, 'block2d' deprecated alias
                    num_blocks_m = out_features // self.block_size
                    num_blocks_n = in_features // self.block_size
                    dequant_scale = torch.ones(
                        num_blocks_m,
                        num_blocks_n,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                elif (
                    self.scaling_mode == "block3d"
                    and in_features > 0
                    and in_features % self.block_size == 0
                ):
                    # Per-row-group 3D: (out_features, num_blocks, 1)
                    num_blocks = in_features // self.block_size
                    dequant_scale = torch.ones(
                        out_features,
                        num_blocks,
                        1,
                        device=self.device,
                        dtype=SCALE_DTYPE,
                    )
                else:
                    # Tensor-wise: single scale
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)

            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32)

        # INT8 quantization path
        if self.target_format == "int8":
            return self._convert_int8(W_float32)

        # FP8 quantization path - route based on scaling_mode
        if self.scaling_mode == "row":
            return self._convert_fp8_rowwise(W_float32)
        elif self.scaling_mode in ("block", "block2d"):
            # 2D block-wise - 'block' is primary, 'block2d' is deprecated alias
            return self._convert_fp8_block2d(W_float32)
        elif self.scaling_mode == "block3d":
            # 3D per-row-group mode (legacy)
            return self._convert_fp8(W_float32)
        else:
            # 'tensor' mode
            return self._convert_fp8(W_float32)

    def _convert_int8(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        INT8 block-wise quantization using BlockWiseINT8Layout or Lode-Wise kernels.

        INT8 block-wise quantization differs from FP8:
        - Uses symmetric quantization with range [-127, 127]
        - Scale is per-block (2D grid): shape (M//block_size, N//block_size)
        - Requires dimensions divisible by block_size
        """
        M, N = W_float32.shape

        # Validate dimensions are divisible by block_size
        if M % self.block_size != 0 or N % self.block_size != 0:
            raise ValueError(
                f"INT8 block-wise quantization requires dimensions divisible by block_size={self.block_size}. "
                f"Got shape ({M}, {N}). Consider using --skip_inefficient_layers or a different block_size."
            )

        # Select quantization backend
        # Use BlockWiseINT8Layout (blockwise backend from quant_ops.py)
        qdata, layout_params = BlockWiseINT8Layout.quantize(
            W_float32, block_size=self.block_size, is_weight=True
        )
        scale = layout_params["scale"]  # Shape: (M//block_size, N//block_size)

        # Optional: Apply learned rounding optimization for INT8
        if not self.no_learned_rounding and self.num_iter > 0:
            print("    - Applying learned rounding optimization for INT8...")
            qdata, scale = self._optimize_int8_learned_rounding(W_float32, qdata, scale)

        # Dequantize to get the reconstructed weight for bias correction
        dequantized_weight = BlockWiseINT8Layout.dequantize(
            qdata, scale, self.block_size, is_weight=True, orig_dtype=COMPUTE_DTYPE
        )

        # Clean up
        del W_float32
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (
            qdata,
            scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized_weight,
        )

    def _int8_dequantize_blockwise(
        self, qdata: torch.Tensor, scale: torch.Tensor, M: int, N: int, block_size: int
    ) -> torch.Tensor:
        """
        Differentiable block-wise INT8 dequantization for optimization.
        Matches BlockWiseINT8Layout._weight_quantize_pytorch logic.

        Args:
            qdata: Quantized values (can be float during optimization), shape (M, N)
            scale: Per-block scales, shape (M//block_size, N//block_size)
            M, N: Original tensor dimensions
            block_size: Block size for quantization

        Returns:
            Dequantized tensor, shape (M, N)
        """
        # Reshape to blocks: (M//bs, bs, N//bs, bs)
        q_blocked = qdata.reshape(
            M // block_size, block_size, N // block_size, block_size
        )
        # Permute to: (M//bs, N//bs, bs, bs)
        q_blocked = q_blocked.permute(0, 2, 1, 3)
        # Broadcast scale: (M//bs, N//bs, 1, 1)
        scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
        # Apply scale
        dequantized = q_blocked * scale_broadcast
        # Permute back and reshape: (M, N)
        dequantized = dequantized.permute(0, 2, 1, 3).reshape(M, N)
        return dequantized

    def _optimize_int8_learned_rounding(
        self, W_float32: torch.Tensor, qdata: torch.Tensor, scale: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply learned rounding optimization for INT8 quantization.
        Uses SVD-based optimization similar to FP8 but adapted for INT8.
        Supports multiple optimizer choices: original, adamw, radam.
        """
        M, N = W_float32.shape

        # Compute SVD for the optimization
        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(
            f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components."
        )

        if self.full_matrix:
            print("    - Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                print("    - Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(
                    W_float32, q=min(k + 10, max_rank), niter=4
                )
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        # Route to appropriate optimizer
        if self.optimizer_choice == "original":
            final_qdata = self._optimize_int8_original(
                W_float32, qdata, scale, U_k, Vh_k
            )
        elif self.optimizer_choice == "adamw":
            final_qdata = self._optimize_int8_adamw(W_float32, qdata, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_qdata = self._optimize_int8_radam(W_float32, qdata, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        del U, Vh, U_k, Vh_k
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return final_qdata, scale

    def _optimize_int8_adamw(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 optimization using AdamW optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)

        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = AdamW([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT8 (AdamW-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(
                q_refined, scale, M, N, block_size
            )

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()

            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            # 'adaptive' mode: fixed LR (AdamW handles momentum internally)

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr < self.early_stop_lr:
                    print("      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping early.")
                elif best_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        final_qdata = (
            (qdata_float + best_delta)
            .clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_radam(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 optimization using RAdam optimizer with manual LR scheduling."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        delta = torch.zeros_like(qdata_float, requires_grad=True)

        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        optimizer = RAdam([delta], lr=curr_lr)

        schedule_name = self.lr_schedule
        best_loss = float("inf")
        best_delta = delta.detach().clone()
        worse_loss_counter = 0
        plateau_counter = 0
        cooldown_counter = 0

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT8 (RAdam-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            optimizer.zero_grad()

            q_refined = qdata_float + delta
            current_dq = self._int8_dequantize_blockwise(
                q_refined, scale, M, N, block_size
            )

            error = current_dq - W_float32
            projected_error = U_k.T @ error @ Vh_k.T
            loss = torch.linalg.norm(projected_error)

            loss.backward()
            optimizer.step()

            current_loss_val = loss.item()

            if current_loss_val < best_loss:
                best_loss = current_loss_val
                best_delta = delta.detach().clone()
                worse_loss_counter = 0
                plateau_counter = 0
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # Manual LR update based on schedule (matching _optimize_int8_original)
            if schedule_name == "exponential":
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = curr_lr
            elif schedule_name == "plateau":
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= self.lr_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * self.lr_factor, self.lr_min)
                        for param_group in optimizer.param_groups:
                            param_group["lr"] = curr_lr
                        cooldown_counter = self.lr_cooldown
                    plateau_counter = 0
            # 'adaptive' mode: fixed LR (RAdam handles momentum internally)

            pbar.set_postfix(
                {
                    "loss": f"{current_loss_val:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                }
            )

            # Early stopping conditions
            if (
                best_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if curr_lr < self.early_stop_lr:
                    print("      - Learning rate bottomed out. Stopping early.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping early.")
                elif best_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping early.")
                break

        pbar.close()

        final_qdata = (
            (qdata_float + best_delta)
            .clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, delta
        return final_qdata

    def _optimize_int8_original(
        self,
        W_float32: torch.Tensor,
        qdata: torch.Tensor,
        scale: torch.Tensor,
        U_k: torch.Tensor,
        Vh_k: torch.Tensor,
    ) -> torch.Tensor:
        """INT8 optimization using original gradient-based optimizer (no autograd)."""
        M, N = W_float32.shape
        block_size = self.block_size

        qdata_float = qdata.to(COMPUTE_DTYPE)
        q_refined = qdata_float.clone()

        best_loss = float("inf")
        best_tensor = None
        worse_loss_counter = 0
        plateau_counter = 0  # For plateau schedule
        cooldown_counter = 0  # For plateau cooldown
        curr_lr = self.optimizer_kwargs.get("lr", 8.077300000003e-3)
        if M == N:
            small_mult = 0.95
        else:
            small_mult = 1.0

        schedule_name = self.lr_schedule

        # Shape-aware plateau parameters
        aspect_ratio = max(M, N) / min(M, N)

        if schedule_name == "plateau" and self.lr_shape_influence > 0:
            # Scale factor based on aspect ratio, modulated by influence
            # Elongated tensors need MORE AGGRESSIVE decay (lower factor)
            ar_factor = math.sqrt(aspect_ratio)
            blend = self.lr_shape_influence

            # Patience unchanged per user feedback
            effective_patience = self.lr_patience

            # More aggressive factor for elongated tensors: factor^ar_factor makes it smaller
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor**ar_factor
            effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend

            # Cooldown unchanged
            effective_cooldown = self.lr_cooldown
        else:
            effective_patience = self.lr_patience
            effective_factor = self.lr_factor
            effective_cooldown = self.lr_cooldown

        pbar = tqdm(
            range(self.num_iter),
            desc=f"    Optimizing INT8 (Original-{schedule_name})",
            leave=False,
            dynamic_ncols=True,
        )
        for i in pbar:
            with torch.no_grad():
                current_dq = self._int8_dequantize_blockwise(
                    q_refined, scale, M, N, block_size
                )
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            current_loss = loss.item()
            # Check if improvement exceeds threshold (supports rel/abs mode like PyTorch ReduceLROnPlateau)
            if self.lr_threshold > 0:
                if self.lr_threshold_mode == "rel":
                    # Relative: significant if loss < best * (1 - threshold)
                    improved = current_loss < best_loss * (1.0 - self.lr_threshold)
                else:  # 'abs'
                    # Absolute: significant if improvement > threshold
                    improved = (best_loss - current_loss) > self.lr_threshold
            else:
                improved = current_loss < best_loss

            # Store counter before potential reset (for no-reset adaptive mode)
            prev_worse_counter = worse_loss_counter

            if improved:
                best_loss = current_loss
                best_tensor = q_refined.clone()
                plateau_counter = 0
                if self.lr_adaptive_mode == "simple-reset":
                    worse_loss_counter = 0
                # no-reset mode: worse_loss_counter preserved for tier calculation
            else:
                worse_loss_counter += 1
                plateau_counter += 1

            # LR update based on schedule
            if schedule_name == "exponential":
                # ExponentialLR: lr = lr * gamma per step
                curr_lr = max(curr_lr * self.lr_gamma, self.lr_min)
            elif schedule_name == "plateau":
                # ReduceLROnPlateau with cooldown (shape-aware)
                if cooldown_counter > 0:
                    cooldown_counter -= 1
                elif plateau_counter >= effective_patience:
                    if curr_lr > self.lr_min:
                        curr_lr = max(curr_lr * effective_factor, self.lr_min)
                        cooldown_counter = effective_cooldown
                    plateau_counter = 0
            else:  # 'adaptive' - tier-based schedule
                # For no-reset mode, use counter value before reset for tier calculation
                counter_for_tier = (
                    prev_worse_counter
                    if (improved and self.lr_adaptive_mode == "no-reset")
                    else worse_loss_counter
                )

                if improved and counter_for_tier < 50:
                    curr_lr = min(curr_lr * (1.25 * small_mult), 100.0)
                elif improved and counter_for_tier >= 50 and counter_for_tier < 75:
                    curr_lr = min(curr_lr * (1.375 * small_mult), 100.0)
                elif improved and counter_for_tier >= 75 and counter_for_tier < 100:
                    curr_lr = min(curr_lr * (1.5 * small_mult), 100.0)
                elif improved and counter_for_tier >= 100 and counter_for_tier < 125:
                    curr_lr = min(curr_lr * (1.75 * small_mult), 100.0)
                elif improved and counter_for_tier >= 125 and counter_for_tier < 150:
                    curr_lr = min(curr_lr * (2.0 * small_mult), 100.0)
                elif improved and counter_for_tier >= 150 and counter_for_tier < 200:
                    curr_lr = min(curr_lr * (2.25 * small_mult), 100.0)
                elif improved and counter_for_tier >= 200 and counter_for_tier < 250:
                    curr_lr = min(curr_lr * (2.5 * small_mult), 100.0)
                elif improved and counter_for_tier >= 250 and counter_for_tier < 300:
                    curr_lr = min(curr_lr * (2.75 * small_mult), 100.0)
                elif improved and counter_for_tier >= 300:
                    curr_lr = min(curr_lr * (3.0 * small_mult), 100.0)
                elif not improved and worse_loss_counter < 26:
                    curr_lr = max(curr_lr * (0.95 * small_mult), 9e-8)
                elif worse_loss_counter >= 26 and worse_loss_counter < 51:
                    curr_lr = max(curr_lr * (0.97 * small_mult), 8e-8)
                elif worse_loss_counter >= 51 and worse_loss_counter < 76:
                    curr_lr = max(curr_lr * (0.985 * small_mult), 7e-8)
                elif worse_loss_counter >= 76 and worse_loss_counter < 101:
                    curr_lr = max(curr_lr * (0.9875 * small_mult), 6e-8)
                elif worse_loss_counter >= 101 and worse_loss_counter < 151:
                    curr_lr = max(curr_lr * (0.98875 * small_mult), 5e-8)
                elif worse_loss_counter >= 151 and worse_loss_counter < 201:
                    curr_lr = max(curr_lr * (0.99 * small_mult), 4e-8)
                elif worse_loss_counter >= 201 and worse_loss_counter < 251:
                    curr_lr = max(curr_lr * (0.99125 * small_mult), 3e-8)
                elif worse_loss_counter >= 251 and worse_loss_counter < 301:
                    curr_lr = max(curr_lr * (0.9925 * small_mult), 2e-8)
                else:  # worse_loss_counter >= 301
                    curr_lr = max(curr_lr * (0.995 * small_mult), 5e-9)

                # Reset counter after boost in no-reset mode
                if improved and self.lr_adaptive_mode == "no-reset":
                    worse_loss_counter = 0

            pbar.set_postfix(
                {
                    "loss": f"{current_loss:.3e}",
                    "best": f"{best_loss:.3e}",
                    "lr": f"{curr_lr:.2e}",
                    "worse_count": f"{worse_loss_counter}",
                }
            )

            # Early stopping conditions (configurable thresholds)
            if (
                current_loss < self.early_stop_loss
                or curr_lr < self.early_stop_lr
                or worse_loss_counter > self.early_stop_stall
            ):
                if (
                    curr_lr < self.early_stop_lr * 1.75
                    and worse_loss_counter > self.early_stop_stall * 0.95
                ):
                    print(
                        "      - Loss has stalled and learning rate has bottomed out. Stopping."
                    )
                elif (
                    current_loss < self.early_stop_loss
                    and curr_lr < self.early_stop_lr * 1.75
                ):
                    print(
                        "      - Learning Rate has bottomed out and loss is negligible. Stopping."
                    )
                elif (
                    worse_loss_counter > self.early_stop_stall * 0.95
                    and current_loss > self.early_stop_loss * 2
                ):
                    print("      - Loss is negligible and loss has stalled. Stopping.")
                elif current_loss < self.early_stop_loss:
                    print("      - Loss is negligible. Stopping.")
                elif curr_lr < self.early_stop_lr:
                    print("      - Learning Rate has bottomed out. Stopping.")
                elif worse_loss_counter > self.early_stop_stall:
                    print("      - Loss has stalled. Stopping.")
                break

            with torch.no_grad():
                # Compute gradient direction in INT8 quantized space
                #
                # Math derivation:
                # - Dequantization: dq = Q * scale (per-block)
                # - Loss L is computed on dq
                # - By chain rule: ∂L/∂Q = ∂L/∂dq * ∂dq/∂Q = ∂L/∂dq * scale
                #
                # So we need to MULTIPLY the weight-space gradient by scale to get Q-space gradient
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k

                # Transform gradient through block-wise structure
                # Reshape grad to blocks, multiply by scale (chain rule), then reshape back
                grad_blocked = grad_direction.reshape(
                    M // block_size, block_size, N // block_size, block_size
                )
                grad_blocked = grad_blocked.permute(0, 2, 1, 3)
                scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1)
                grad_scaled = grad_blocked * scale_broadcast
                grad_scaled = grad_scaled.permute(0, 2, 1, 3).reshape(M, N)

                q_refined -= curr_lr * grad_scaled

        pbar.close()

        final_tensor = best_tensor if best_tensor is not None else q_refined
        final_qdata = (
            final_tensor.clamp(-INT8_SYMMETRIC_MAX, INT8_SYMMETRIC_MAX)
            .round()
            .to(TARGET_INT8_DTYPE)
        )
        del qdata_float, q_refined
        return final_qdata

    def _convert_fp8(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Original FP8 quantization path."""

        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == "block":
            if (
                W_float32.ndim == 2
                and W_float32.shape[1] > 0
                and W_float32.shape[1] % self.block_size == 0
            ):
                print(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(
                    out_features, in_features
                )
            else:
                print(
                    f"    - WARNING: Tensor shape {list(W_float32.shape)} not suitable for block size {self.block_size}. Falling back to 'tensor' scaling."
                )
                current_scaling_mode = "tensor"

        if current_scaling_mode == "tensor":
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale

        assert (
            scale is not None
        ), "scale should not be None after scaling mode selection"

        # Skip SVD optimization if no_learned_rounding is set
        if self.no_learned_rounding:
            print("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_f8 = (
                    (W_float32 * scale)
                    .clamp(-self.f8_max_val, self.f8_max_val)
                    .to(TARGET_FP8_DTYPE)
                )
                if compact_scale is None:
                    dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
                else:
                    if current_scaling_mode == "block":
                        dequant_scale = compact_scale.reciprocal()
                    else:
                        dequant_scale = compact_scale.reciprocal()
                    dequant_scale = dequant_scale.to(
                        device=self.device, dtype=SCALE_DTYPE
                    )
                dequantized_weight_tensor = (
                    W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
                )
            del W_float32, scale, compact_scale
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return W_f8, dequant_scale, dequantized_weight_tensor

        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(
            f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components."
        )

        if self.full_matrix:
            print("Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                print("Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(
                    W_float32, q=min(k + 10, max_rank), niter=4
                )
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        #    final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        #    final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            # Ensure compact_scale is valid before calling reciprocal; fall back to ones if missing.
            if compact_scale is None:
                print(
                    "    - WARNING: compact_scale is None, falling back to torch.ones for dequant_scale."
                )
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            else:
                if current_scaling_mode == "block":
                    dequant_scale = compact_scale.reciprocal()
                else:
                    dequant_scale = compact_scale.reciprocal()
                dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized_weight_tensor = (
                W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale
            )
        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return (
            W_f8,
            dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized_weight_tensor,
        )

    def _convert_fp8_rowwise(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Row-wise FP8 quantization - one scale per row.

        Scale shape: (out_features,)
        Good balance between accuracy and memory for most weight matrices.
        """
        M, N = W_float32.shape
        print("    - Using row-wise FP8 scaling (1 scale per row).")

        # Compute per-row max
        row_max = W_float32.abs().amax(dim=1, keepdim=True)  # (M, 1)
        quant_scale = self.f8_max_val / row_max.clamp_min_(1e-12)  # (M, 1)

        if self.no_learned_rounding:
            print("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                W_scaled = W_float32 * quant_scale
                W_f8 = W_scaled.to(TARGET_FP8_DTYPE)
                dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)
                dequantized = W_f8.to(COMPUTE_DTYPE) / quant_scale

            del W_float32
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return (
                W_f8,
                dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
                dequantized,
            )

        # With learned rounding optimization
        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(
            f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components."
        )

        if self.full_matrix:
            print("Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                print("Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(
                    W_float32, q=min(k + 10, max_rank), niter=4
                )
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        # Use the appropriate optimizer with row-wise scale
        scale = quant_scale  # (M, 1) for broadcast
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            dequant_scale = (1.0 / quant_scale).squeeze(1)  # (M,)
            dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / quant_scale

        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, quant_scale
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return W_f8, dequant_scale, dequantized

    def _convert_fp8_block2d(
        self, W_float32: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        True 2D block-wise FP8 quantization - one scale per block_size x block_size tile.

        Scale shape: (M // block_size, N // block_size)
        Similar to INT8 block-wise scaling, optimized for tiled GEMM inference.
        """
        M, N = W_float32.shape
        bs = self.block_size

        # Validate dimensions
        if M % bs != 0 or N % bs != 0:
            print(
                f"    - WARNING: Dimensions ({M}, {N}) not divisible by block_size={bs}. Falling back to row-wise."
            )
            return self._convert_fp8_rowwise(W_float32)

        print(f"    - Using 2D block-wise FP8 scaling with block size {bs}.")

        # Reshape to 2D blocks
        W_blocked = W_float32.reshape(M // bs, bs, N // bs, bs).permute(
            0, 2, 1, 3
        )  # (M//bs, N//bs, bs, bs)
        block_max = W_blocked.abs().amax(dim=(2, 3))  # (M//bs, N//bs)
        quant_scale = self.f8_max_val / block_max.clamp_min_(1e-12)  # (M//bs, N//bs)

        if self.no_learned_rounding:
            print("    - Simple quantization (no learned rounding).")
            with torch.no_grad():
                # Apply scale per-block
                scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(
                    -1
                )  # (M//bs, N//bs, 1, 1)
                W_scaled_blocked = W_blocked * scale_broadcast
                W_f8_blocked = W_scaled_blocked.to(TARGET_FP8_DTYPE)
                W_f8 = W_f8_blocked.permute(0, 2, 1, 3).reshape(M, N)

                # Dequant scale is reciprocal
                dequant_scale = 1.0 / quant_scale  # (M//bs, N//bs)

                # Dequantize for bias correction
                dequant_broadcast = dequant_scale.unsqueeze(-1).unsqueeze(-1)
                dequantized_blocked = W_f8_blocked.to(COMPUTE_DTYPE) * dequant_broadcast
                dequantized = dequantized_blocked.permute(0, 2, 1, 3).reshape(M, N)

            del W_float32, W_blocked
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()

            return (
                W_f8,
                dequant_scale.to(device=self.device, dtype=SCALE_DTYPE),
                dequantized,
            )

        # With learned rounding - expand scale to full tensor for optimization
        scale_broadcast = quant_scale.unsqueeze(-1).unsqueeze(
            -1
        )  # (M//bs, N//bs, 1, 1)
        scale_full_blocked = scale_broadcast.expand(-1, -1, bs, bs)
        scale_full = scale_full_blocked.permute(0, 2, 1, 3).reshape(M, N)

        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(
            f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components."
        )

        if self.full_matrix:
            print("Using torch.linalg.svd with full_matrices=True")
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver="gesvd")
        else:
            try:
                print("Trying svd_lowrank")
                U, _, Vh = torch.svd_lowrank(
                    W_float32, q=min(k + 10, max_rank), niter=4
                )
                Vh = Vh.T
            except RuntimeError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        # Use the optimizer with the expanded scale
        if self.optimizer_choice == "adamw":
            final_tensor_scaled = self._optimize_adamw(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "radam":
            final_tensor_scaled = self._optimize_radam(W_float32, scale_full, U_k, Vh_k)
        elif self.optimizer_choice == "original":
            final_tensor_scaled = self._optimize_original(
                W_float32, scale_full, U_k, Vh_k
            )
        else:
            raise ValueError(f"Unknown optimizer: '{self.optimizer_choice}'")

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            dequant_scale = 1.0 / quant_scale  # (M//bs, N//bs)
            dequant_scale = dequant_scale.to(device=self.device, dtype=SCALE_DTYPE)
            dequantized = W_f8.to(COMPUTE_DTYPE) / scale_full

        del (
            W_float32,
            W_blocked,
            scale_full,
            scale_broadcast,
            U,
            Vh,
            U_k,
            Vh_k,
            final_tensor_scaled,
            quant_scale,
        )
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return W_f8, dequant_scale, dequantized


# --- Main script execution functions ---


def convert_to_fp8_scaled(
    input_file: str,
    output_file: str,
    comfy_quant: bool,
    t5xxl: bool,
    mistral: bool,
    visual: bool,
    flux2: bool,
    distillation_large: bool,
    distillation_small: bool,
    nerf_large: bool,
    nerf_small: bool,
    radiance: bool,
    wan: bool,
    qwen: bool,
    hunyuan: bool,
    zimage: bool,
    zimage_refiner: bool,
    calib_samples: int,
    seed: int,
    int8: bool = False,
    fallback: Optional[str] = None,
    custom_layers: Optional[str] = None,
    custom_type: Optional[str] = None,
    custom_block_size: Optional[int] = None,
    custom_scaling_mode: Optional[str] = None,
    custom_simple: bool = False,
    custom_heur: bool = False,
    fallback_block_size: Optional[int] = None,
    fallback_simple: bool = False,
    full_precision_matrix_mult: bool = False,
    skip_inefficient_layers: bool = False,
    include_input_scale: bool = False,
    no_learned_rounding: bool = False,
    save_quant_metadata: bool = False,
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    **converter_kwargs,
):
    # Determine target format (priority: int8 > fp8)
    if int8:
        target_format = "int8"
        format_name = "INT8"
    else:
        target_format = "fp8"
        format_name = "FP8"

    print(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    print("-" * 60)
    if int8:
        print("Target format: INT8 (block-wise quantization)")
        print(f"INT8 Range: [{-INT8_SYMMETRIC_MAX}, {INT8_SYMMETRIC_MAX}]")
    else:
        print(
            f"Target FP8 format: {TARGET_FP8_DTYPE}\nFP8 Range: [{FP8_MIN}, {FP8_MAX}]"
        )
    print("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    if comfy_quant:
        print(
            "Comfy quantization mode enabled: Using comfy_quant layer names and settings."
        )
        comfy_quant = True
    else:
        comfy_quant = False

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

    # Add target_format and no_learned_rounding to converter kwargs
    converter_kwargs["target_format"] = target_format
    converter_kwargs["no_learned_rounding"] = no_learned_rounding

    # Extract block_size for comfy_quant format
    block_size = converter_kwargs.get("block_size", 64)

    # Helper function to create converter for a specific format type
    def create_converter_for_format(
        fmt: str, overrides: dict = None
    ) -> LearnedRoundingConverter:
        kwargs = converter_kwargs.copy()
        kwargs["target_format"] = fmt
        if overrides:
            kwargs.update(overrides)
        return LearnedRoundingConverter(**kwargs)

    # Helper function to get format metadata
    def get_format_info(fmt: str) -> dict:
        """Returns dtype and format name for a quantization format."""
        format_map = {
            "int8": {"dtype": TARGET_INT8_DTYPE, "name": "INT8"},
            "fp8": {"dtype": TARGET_FP8_DTYPE, "name": "FP8"},
        }
        return format_map.get(fmt, format_map["fp8"])

    # Create converters for each format type used
    converters = {"primary": create_converter_for_format(target_format)}

    # Create fallback converter with optional overrides
    if fallback:
        fallback_overrides = {}
        if fallback_block_size is not None:
            fallback_overrides["block_size"] = fallback_block_size
        if fallback_simple:
            fallback_overrides["no_learned_rounding"] = True
        converters["fallback"] = create_converter_for_format(
            fallback, fallback_overrides if fallback_overrides else None
        )
        override_note = (
            f" (block_size={fallback_block_size})" if fallback_block_size else ""
        )
        override_note += " (simple)" if fallback_simple else ""
        print(
            f"Fallback quantization enabled: {fallback.upper()}{override_note} for excluded layers"
        )

    # Create custom converter with optional overrides
    if custom_layers and custom_type:
        custom_overrides = {}
        if custom_block_size is not None:
            custom_overrides["block_size"] = custom_block_size
        if custom_scaling_mode is not None:
            custom_overrides["scaling_mode"] = custom_scaling_mode
        if custom_simple:
            custom_overrides["no_learned_rounding"] = True
        converters["custom"] = create_converter_for_format(
            custom_type, custom_overrides if custom_overrides else None
        )
        override_note = (
            f" (block_size={custom_block_size})" if custom_block_size else ""
        )
        override_note += (
            f" (scaling_mode={custom_scaling_mode})" if custom_scaling_mode else ""
        )
        override_note += " (simple)" if custom_simple else ""
        print(
            f"Custom layer quantization enabled: {custom_type.upper()}{override_note} for pattern '{custom_layers}'"
        )

    # Compile custom_layers regex pattern
    custom_pattern = None
    if custom_layers:
        try:
            custom_pattern = re.compile(custom_layers)
        except re.error as e:
            print(f"ERROR: Invalid regex pattern '{custom_layers}': {e}")
            return

    print("\nScanning model and generating simulated calibration data...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith(".weight") and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples,
                    in_features,
                    dtype=COMPUTE_DTYPE,
                    generator=seed_generator,
                    device=seed_device,
                )
    print("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted(
        [
            key
            for key in tensors.keys()
            if key.endswith(".weight") and tensors[key].ndim == 2
        ]
    )
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0
    custom_count = 0
    fallback_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")
    print("-" * 60)

    for i, key in enumerate(weight_keys):
        exclusion_reason = ""
        use_custom = False
        use_fallback = False
        use_layer_config = False
        layer_format = target_format  # default to primary
        layer_settings = None  # Per-layer settings from config

        # T5XXL decoder tensors are always removed (not quantized, not kept)
        if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            skipped_count += 1
            continue

        # Check layer_config FIRST (highest priority)
        if layer_config:
            layer_settings = get_layer_settings(
                key, layer_config, fullmatch=layer_config_fullmatch
            )
            if layer_settings:
                if layer_settings.get("skip", False):
                    print(f"({i+1}/{total_weights}) Skipping (layer-config): {key}")
                    original_tensor = tensors[key]
                    new_tensors[key] = original_tensor.to(
                        device="cpu", dtype=original_tensor.dtype
                    )
                    skipped_count += 1
                    continue
                use_layer_config = True
                # Map format to layer_format type
                fmt = layer_settings["format"]
                if fmt.startswith("float8"):
                    layer_format = "fp8"
                elif fmt.startswith("int8"):
                    layer_format = "int8"
                else:
                    layer_format = "fp8"  # fallback

        # Check for custom pattern match (second priority, only if no layer_config match)
        if not use_layer_config and custom_pattern and custom_pattern.search(key):
            use_custom = True
            layer_format = custom_type

        # Check exclusion filters (only matters if not custom matched and not layer_config matched)
        if not use_custom and not use_layer_config:
            if (t5xxl or mistral) and any(n in key for n in AVOID_KEY_NAMES):
                exclusion_reason = "T5XXL/Mistral exclusion"
            elif visual and any(n in key for n in VISUAL_AVOID_KEY_NAMES):
                exclusion_reason = "Visual exclusion"
            elif radiance and any(n in key for n in RADIANCE_LAYER_KEYNAMES):
                exclusion_reason = "Radiance exclusion"
            elif wan and any(n in key for n in AVOID_KEY_NAMES):
                exclusion_reason = "WAN exclusion"
            elif qwen and any(n in key for n in QWEN_AVOID_KEY_NAMES):
                exclusion_reason = "Qwen Image exclusion"
            elif (zimage or zimage_refiner) and any(
                n in key for n in ZIMAGE_AVOID_KEY_NAMES
            ):
                exclusion_reason = "Z-Image exclusion"
            elif hunyuan and any(n in key for n in HUNYUAN_AVOID_KEY_NAMES):
                exclusion_reason = "Hunyuan Video 1.5 exclusion"
            elif flux2 and any(n in key for n in FLUX2_LAYER_KEYNAMES):
                exclusion_reason = "Flux2 exclusion and keep in high precision"
            elif distillation_large and any(
                n in key for n in DISTILL_LAYER_KEYNAMES_LARGE
            ):
                exclusion_reason = (
                    "Distillation layer and Chroma keep in high precision"
                )
            elif distillation_small and any(
                n in key for n in DISTILL_LAYER_KEYNAMES_SMALL
            ):
                exclusion_reason = (
                    "Distillation layer and Chroma keep in high precision"
                )
            elif nerf_large and any(n in key for n in NERF_LAYER_KEYNAMES_LARGE):
                exclusion_reason = (
                    "NeRF layer, distillation layer and txt_in keep in high precision"
                )
            elif nerf_small and any(n in key for n in NERF_LAYER_KEYNAMES_SMALL):
                exclusion_reason = (
                    "NeRF layer and distillation layer keep in high precision"
                )
            elif wan and any(n in key for n in WAN_LAYER_KEYNAMES):
                exclusion_reason = "WAN layer keep in high precision"
            elif qwen and any(n in key for n in QWEN_LAYER_KEYNAMES):
                exclusion_reason = "Qwen Image layer keep in high precision"
            elif zimage and any(n in key for n in ZIMAGE_LAYER_KEYNAMES):
                exclusion_reason = "Z-Image layer keep in high precision"
            elif zimage_refiner and any(
                n in key for n in ZIMAGE_REFINER_LAYER_KEYNAMES
            ):
                exclusion_reason = "Z-Image refiner layer keep in high precision"

        # Handle excluded layers: use fallback if available, otherwise skip
        if exclusion_reason and not use_custom and not use_layer_config:
            if fallback:
                use_fallback = True
                layer_format = fallback
                print(
                    f"({i+1}/{total_weights}) Processing (fallback {fallback.upper()}): {key} (was: {exclusion_reason})"
                )
            else:
                print(
                    f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})"
                )
                original_tensor = tensors[key]
                new_tensors[key] = original_tensor.to(
                    device="cpu", dtype=original_tensor.dtype
                )
                skipped_count += 1
                continue

        # Log what we're doing
        if use_layer_config:
            fmt = layer_settings["format"]
            print(f"({i+1}/{total_weights}) Processing (config {fmt}): {key}")
            custom_count += 1  # Count layer_config as custom
        elif use_custom:
            print(
                f"({i+1}/{total_weights}) Processing (custom {custom_type.upper()}): {key}"
            )
            custom_count += 1
        elif use_fallback:
            fallback_count += 1
        else:
            print(f"({i+1}/{total_weights}) Processing ({format_name}): {key}")

        processed_count += 1
        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = original_tensor.to(
                device="cpu", dtype=original_tensor.dtype
            )
            continue

        # Check performance heuristics for inefficient layers
        # Custom layers use custom_heur flag, others use global skip_inefficient_layers
        apply_heur = custom_heur if use_custom else skip_inefficient_layers
        if apply_heur:
            should_skip, skip_perf_reason = should_skip_layer_for_performance(
                original_tensor, block_size
            )
            if should_skip:
                print(f"  - Skipping for performance: {skip_perf_reason}")
                new_tensors[key] = original_tensor.to(
                    device="cpu", dtype=original_tensor.dtype
                )
                skipped_count += 1
                continue

        # Select the appropriate converter based on layer format
        if use_layer_config:
            # Create converter dynamically from layer_config settings
            cfg_overrides = {}
            cfg_block_size = layer_settings.get("block_size")
            cfg_scaling_mode = layer_settings.get("scaling_mode")
            cfg_simple = layer_settings.get("simple", False)
            if cfg_block_size is not None:
                cfg_overrides["block_size"] = cfg_block_size
            if cfg_scaling_mode is not None:
                cfg_overrides["scaling_mode"] = cfg_scaling_mode
            if cfg_simple:
                cfg_overrides["no_learned_rounding"] = True
            converter = create_converter_for_format(
                layer_format, cfg_overrides if cfg_overrides else None
            )
        elif use_custom:
            converter = converters["custom"]
        elif use_fallback:
            converter = converters["fallback"]
        else:
            converter = converters["primary"]

        # Determine format type for this layer
        is_int8 = layer_format == "int8"

        q_tensor, dequant_s, dequant_w = converter.convert(original_tensor)
        new_tensors[key] = q_tensor.to(device="cpu")
        base_name = key[: key.rfind(".weight")]
        bias_key = f"{base_name}.bias"

        if comfy_quant is True:
            # Use the converter's block_size (respects custom/fallback overrides)
            layer_block_size = converter.block_size

            # Determine full_precision_matrix_mult: per-layer config takes priority over global
            layer_full_precision_mm = full_precision_matrix_mult
            if use_layer_config and "full_precision_matrix_mult" in layer_settings:
                layer_full_precision_mm = layer_settings["full_precision_matrix_mult"]

            # Variables for metadata collection
            comfy_quant_format = None
            block_size_for_meta = None

            # Use appropriate scale key name based on format
            if is_int8:
                new_tensors[f"{base_name}.weight_scale"] = (
                    dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                )
                comfy_quant_format = "int8_blockwise"
                block_size_for_meta = layer_block_size
                # Use int8_blockwise format
                comfy_quant_tensor = create_comfy_quant_tensor(
                    "int8_blockwise",
                    block_size=layer_block_size,
                    full_precision_matrix_mult=layer_full_precision_mm
                    if layer_full_precision_mm
                    else None,
                )
                # Always add input_scale for INT8 (matches reference behavior)
                new_tensors[f"{base_name}.input_scale"] = torch.tensor(
                    1.0, dtype=torch.float32, device="cpu"
                )
            else:
                # FP8 format - determine format based on scaling_mode or layer_config
                new_tensors[f"{base_name}.weight_scale"] = (
                    dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                )

                # Select FP8 format type based on layer_config or scaling mode
                if use_layer_config:
                    # Use format directly from layer_config
                    fp8_format = layer_settings["format"]
                    fp8_block_size = layer_settings.get("block_size", layer_block_size)
                elif converter.scaling_mode == "row":
                    fp8_format = "float8_e4m3fn_rowwise"
                    fp8_block_size = None
                elif converter.scaling_mode in ("block", "block2d"):
                    # 2D block-wise - 'block' is primary, 'block2d' is deprecated alias
                    fp8_format = "float8_e4m3fn_blockwise"
                    fp8_block_size = layer_block_size
                elif converter.scaling_mode == "block3d":
                    # 3D per-row-group uses base format (not recommended)
                    fp8_format = "float8_e4m3fn"
                    fp8_block_size = None
                else:
                    # 'tensor' mode
                    fp8_format = "float8_e4m3fn"
                    fp8_block_size = None

                comfy_quant_format = fp8_format
                block_size_for_meta = fp8_block_size

                comfy_quant_tensor = create_comfy_quant_tensor(
                    fp8_format,
                    block_size=fp8_block_size,
                    full_precision_matrix_mult=layer_full_precision_mm
                    if layer_full_precision_mm
                    else None,
                )
                # Add input_scale for FP8: use weight_scale for t5xxl/mistral, 1.0 otherwise
                if include_input_scale or t5xxl or mistral or visual:
                    if t5xxl or mistral or visual:
                        new_tensors[f"{base_name}.input_scale"] = (
                            dequant_s.to(device="cpu", dtype=SCALE_DTYPE)
                            .detach()
                            .clone()
                        )
                    else:
                        new_tensors[f"{base_name}.input_scale"] = torch.tensor(
                            1.0, dtype=torch.float32, device="cpu"
                        )
            new_tensors[f"{base_name}.comfy_quant"] = comfy_quant_tensor.to(
                device="cpu"
            )

            # Collect metadata if enabled
            if save_quant_metadata:
                # Reconstruct the dict that was used to create the tensor
                meta_entry = {"format": comfy_quant_format}
                block_based_formats = {"int8_blockwise", "float8_e4m3fn_blockwise"}
                if (
                    block_size_for_meta is not None
                    and comfy_quant_format in block_based_formats
                ):
                    meta_entry["group_size"] = block_size_for_meta
                if layer_full_precision_mm:
                    meta_entry["full_precision_matrix_mult"] = True

                quant_metadata_layers[base_name] = meta_entry

        else:
            # Non-comfy (legacy) path - FP8/INT8 only
            new_tensors[f"{base_name}.scale_weight"] = (
                dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
            )
            # Add scale_input for non-comfy mode: use dequant_s for t5xxl/mistral, ones for others
            if include_input_scale or t5xxl or mistral or visual:
                if t5xxl or mistral or visual:
                    new_tensors[f"{base_name}.scale_input"] = (
                        dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                    )
                else:
                    # Shape matches scale_weight, filled with 1.0
                    new_tensors[f"{base_name}.scale_input"] = torch.ones_like(
                        dequant_s, dtype=SCALE_DTYPE, device="cpu"
                    )

        # Determine if this layer uses simple mode (skip bias correction to save memory)
        layer_uses_simple = (
            custom_simple
            if use_custom
            else (fallback_simple if use_fallback else no_learned_rounding)
        )

        if bias_key in tensors:
            if layer_uses_simple:
                # Skip bias correction for simple mode (saves memory, avoids OOM on large layers)
                print(f"  - Keeping original bias (simple mode): {bias_key}")
                new_tensors[bias_key] = tensors[bias_key]
            else:
                print(f"  - Adjusting corresponding bias: {bias_key}")
                with torch.no_grad():
                    original_bias = tensors[bias_key]
                    in_features = original_tensor.shape[1]
                    if in_features not in calibration_data_cache:
                        print("  - WARNING: No calibration data for bias correction.")
                        new_tensors[bias_key] = original_bias
                    else:
                        X_calib_dev = calibration_data_cache[in_features].to(
                            device=device
                        )
                        W_orig_dev = original_tensor.to(
                            device=device, dtype=COMPUTE_DTYPE
                        )
                        W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                        b_orig_dev = original_bias.to(
                            device=device, dtype=COMPUTE_DTYPE
                        )
                        weight_error = W_orig_dev - W_dequant_dev
                        output_error = X_calib_dev @ weight_error.T
                        bias_correction = output_error.mean(dim=0)
                        b_new = b_orig_dev - bias_correction
                        new_tensors[bias_key] = b_new.to(
                            device="cpu", dtype=original_bias.dtype
                        )
                        print(
                            f"    - Original bias mean : {original_bias.mean().item():.6f}\n    - Corrected bias mean: {new_tensors[bias_key].mean().item():.6f}"
                        )
                        del (
                            W_orig_dev,
                            W_dequant_dev,
                            X_calib_dev,
                            b_orig_dev,
                            weight_error,
                            output_error,
                            bias_correction,
                            b_new,
                        )
                        if device == "cuda":
                            torch.cuda.empty_cache()

        # T5XXL/Mistral fallback: ensure input scale exists with correct key format
        if t5xxl or mistral or visual:
            if comfy_quant and f"{base_name}.input_scale" not in new_tensors:
                new_tensors[f"{base_name}.input_scale"] = (
                    dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                )
            elif not comfy_quant and f"{base_name}.scale_input" not in new_tensors:
                new_tensors[f"{base_name}.scale_input"] = (
                    dequant_s.to(device="cpu", dtype=SCALE_DTYPE).detach().clone()
                )

        # Get scale key name based on comfy_quant mode
        scale_key = (
            f"{base_name}.weight_scale" if comfy_quant else f"{base_name}.scale_weight"
        )
        if scale_key in new_tensors:
            new_scale = new_tensors[scale_key]
            if dequant_s.ndim == 1:
                print(
                    f"    - Final Dequant Scale value: {new_scale}\n    - Final Weight shape       : {q_tensor.shape}"
                )
            else:
                print(
                    f"    - Final Dequant Scale shape: {new_scale.shape}\n    - Final Weight shape       : {q_tensor.shape}"
                )
        print("-" * 60)

    for key, tensor in tensors.items():
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and t5xxl:
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor

    # Add scaled_fp8 marker only for legacy non-comfy_quant FP8 format
    # Use empty((0)) when input_scale is present (t5xxl, mistral, or --input_scale flag)
    if (
        not comfy_quant
        and not int8
        and not custom_layers
        and "scaled_fp8" not in new_tensors
    ):
        new_tensors["scaled_fp8"] = (
            torch.empty((0), dtype=TARGET_FP8_DTYPE)
            if (t5xxl or mistral or visual or include_input_scale)
            else torch.empty((2), dtype=TARGET_FP8_DTYPE)
        )

    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Prepare metadata args
        save_kwargs = {}
        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            save_kwargs["metadata"] = {
                "_quantization_metadata": json.dumps(full_metadata)
            }
            print(
                f"  Adding quantization metadata for {len(quant_metadata_layers)} layers"
            )

        # Normalize any 1-element scale tensors to scalars
        new_tensors, normalized_count = normalize_tensorwise_scales(new_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            print(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(new_tensors, output_file, **save_kwargs)

        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return

    print("-" * 60)
    print("Summary:")
    summary_parts = [
        f"  - Original tensor count : {len(tensors)}",
        f"  - Weights processed     : {processed_count}",
    ]
    if custom_count > 0:
        summary_parts.append(f"    - Custom type layers  : {custom_count}")
    if fallback_count > 0:
        summary_parts.append(f"    - Fallback type layers: {fallback_count}")
    summary_parts.extend(
        [
            f"  - Weights skipped       : {skipped_count}",
            f"  - Final tensor count    : {len(new_tensors)}",
        ]
    )
    print("\n".join(summary_parts))
    print("-" * 60)


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

        # Prepare metadata args
        save_kwargs = {}
        if save_quant_metadata and quant_metadata_layers:
            full_metadata = {"format_version": "1.0", "layers": quant_metadata_layers}
            save_kwargs["metadata"] = {
                "_quantization_metadata": json.dumps(full_metadata)
            }
            print(
                f"  Adding quantization metadata for {len(quant_metadata_layers)} layers"
            )

        # Normalize any 1-element scale tensors to scalars
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors, NORMALIZE_SCALES_ENABLED)
        if normalized_count > 0:
            print(f"  Normalized {normalized_count} scale tensors to scalars")
        save_file(output_tensors, output_file, **save_kwargs)

        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return


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
    print("Adding .scale_input to legacy fp8_scaled model")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
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

    # Verify this is an fp8_scaled model
    if "scaled_fp8" not in tensors:
        print(
            "ERROR: This does not appear to be an fp8_scaled model (missing 'scaled_fp8' marker)"
        )
        print("       Use this mode only for legacy fp8_scaled format models.")
        return

    print("Verified: Input is fp8_scaled format")

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
            print("  Converted scaled_fp8 marker to single-element tensor")
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
    print("\n" + "-" * 60)
    print("Conversion Summary:")
    print(f"  Total tensors input:     {len(tensors)}")
    print(f"  Total tensors output:    {len(output_tensors)}")
    print(f"  scale_input added:       {added_scale_input}")
    if already_has_input > 0:
        print(f"  Already had scale_input: {already_has_input}")
    if skipped_non_fp8 > 0:
        print(f"  Skipped (not FP8):       {skipped_non_fp8}")
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
    print("Cleaning up legacy fp8_scaled model")
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print(f"scaled_fp8 marker size: {marker_size}")
    if add_scale_input:
        print("Adding missing scale_input: Yes")
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
            print(f"  Set scaled_fp8 marker to empty(({marker_size}))")
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
                    print(f"  Removing orphaned: {key} (weight is {weight_dtypes[base_name]})")
            else:
                removed_scale_weight += 1
                print(f"  Removing orphaned: {key} (no weight found)")
            continue

        if key.endswith(".scale_input"):
            base_name = key[:-12]  # Remove '.scale_input'
            if base_name in weight_dtypes:
                if weight_dtypes[base_name] == TARGET_FP8_DTYPE:
                    output_tensors[key] = tensor
                    kept_scale_input += 1
                else:
                    removed_scale_input += 1
                    print(f"  Removing orphaned: {key} (weight is {weight_dtypes[base_name]})")
            else:
                removed_scale_input += 1
                print(f"  Removing orphaned: {key} (no weight found)")
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
    print("\n" + "-" * 60)
    print("Cleanup Summary:")
    print(f"  Total tensors input:      {len(tensors)}")
    print(f"  Total tensors output:     {len(output_tensors)}")
    if kept_scale_weight > 0:
        print(f"  scale_weight kept:        {kept_scale_weight}")
    if kept_scale_input > 0:
        print(f"  scale_input kept:         {kept_scale_input}")
    if added_scale_input > 0:
        print(f"  scale_input added:        {added_scale_input}")
    if removed_scale_weight > 0:
        print(f"  scale_weight removed:     {removed_scale_weight}")
    if removed_scale_input > 0:
        print(f"  scale_input removed:      {removed_scale_input}")
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

        print("Cleanup complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return


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


# --- CLI Help Sections ---
# Arguments categorized for multi-section help output

EXPERIMENTAL_ARGS = {
    "int8",
    "fallback",
    "custom_layers",
    "custom_type",
    "custom_block_size",
    "custom_scaling_mode",
    "custom_simple",
    "custom_heur",
    "fallback_block_size",
    "fallback_simple",
    "heur",
    "scaling_mode",
    "block_size",
}

FILTER_ARGS = {
    "t5xxl",
    "mistral",
    "visual",
    "flux2",
    "distillation_large",
    "distillation_small",
    "nerf_large",
    "nerf_small",
    "radiance",
    "wan",
    "qwen",
    "hunyuan",
    "zimage",
    "zimage_refiner",
}

ADVANCED_ARGS = {
    "lr_shape_influence",
    "lr_threshold_mode",
    "early_stop_loss",
    "early_stop_lr",
    "early_stop_stall",
}


class MultiHelpArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with multiple help sections for experimental and filter args."""

    def __init__(
        self,
        *args,
        experimental_args=None,
        filter_args=None,
        advanced_args=None,
        **kwargs,
    ):
        self._experimental_args = experimental_args or set()
        self._filter_args = filter_args or set()
        self._advanced_args = advanced_args or set()
        self._all_actions = []  # Track all actions for section-specific help
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if hasattr(self, "_all_actions"):
            self._all_actions.append(action)
        return action

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Check for special help flags before parsing
        if "--help-experimental" in args or "-he" in args:
            self._print_experimental_help()
            sys.exit(0)
        elif "--help-filters" in args or "-hf" in args:
            self._print_filters_help()
            sys.exit(0)
        elif "--help-advanced" in args or "-ha" in args:
            self._print_advanced_help()
            sys.exit(0)

        return super().parse_args(args, namespace)

    def _get_dest_name(self, action):
        """Get the destination name for an action."""
        return action.dest

    def _format_action_help(self, action):
        """Format a single action for help output."""
        # Get option strings
        opts = (
            ", ".join(action.option_strings) if action.option_strings else action.dest
        )

        # Get help text
        help_text = action.help or ""
        if help_text == argparse.SUPPRESS:
            return None

        # Format default if present and not suppressed
        if action.default is not None and action.default != argparse.SUPPRESS:
            if action.default is not False and action.default != "":
                if isinstance(action.default, str):
                    help_text += f" (default: '{action.default}')"
                else:
                    help_text += f" (default: {action.default})"

        # Format choices if present
        if action.choices:
            choices_str = ", ".join(str(c) for c in action.choices)
            help_text += f" [choices: {choices_str}]"

        return f"  {opts:30s} {help_text}"

    def _print_experimental_help(self):
        """Print help for experimental features."""
        print("Experimental Quantization Features")
        print("=" * 60)
        print()
        print("These are advanced/experimental options for non-default quantization")
        print("formats and fine-grained control. Use --help for standard options.")
        print()
        print("Alternative Quantization Formats:")
        print("-" * 40)

        format_args = [
            "int8",
            "fallback",
            "block_size",
            "scaling_mode",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in format_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Custom Layer Quantization:")
        print("-" * 40)

        custom_args = [
            "custom_layers",
            "custom_type",
            "custom_block_size",
            "custom_scaling_mode",
            "custom_simple",
            "custom_heur",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in custom_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Fallback Layer Options:")
        print("-" * 40)

        fallback_args = ["fallback_block_size", "fallback_simple"]
        for action in self._all_actions:
            if self._get_dest_name(action) in fallback_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Performance Tuning:")
        print("-" * 40)

        perf_args = ["heur"]
        for action in self._all_actions:
            if self._get_dest_name(action) in perf_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def _print_filters_help(self):
        """Print help for model-specific filter presets."""
        print("Model-Specific Exclusion Filters")
        print("=" * 60)
        print()
        print("These flags keep certain model-specific layers in high precision")
        print("(not quantized). Multiple filters can be combined.")
        print()
        print("Text Encoders:")
        print("-" * 40)

        text_args = ["t5xxl", "mistral", "visual"]
        for action in self._all_actions:
            if self._get_dest_name(action) in text_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Diffusion Models (Flux-style):")
        print("-" * 40)

        diffusion_args = [
            "flux2",
            "distillation_large",
            "distillation_small",
            "nerf_large",
            "nerf_small",
            "radiance",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in diffusion_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Video Models:")
        print("-" * 40)

        video_args = ["wan", "hunyuan"]
        for action in self._all_actions:
            if self._get_dest_name(action) in video_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Image Models:")
        print("-" * 40)

        image_args = ["qwen", "zimage", "zimage_refiner"]
        for action in self._all_actions:
            if self._get_dest_name(action) in image_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def _print_advanced_help(self):
        """Print help for advanced LR tuning and early stopping options."""
        print("Advanced LR Tuning & Early Stopping Options")
        print("=" * 60)
        print()
        print("These options provide fine-grained control over the optimizer")
        print("learning rate schedules and early stopping behavior.")
        print()
        print("Shape-Adaptive LR (Plateau Schedule):")
        print("-" * 40)

        shape_args = ["lr_shape_influence", "lr_threshold_mode"]
        for action in self._all_actions:
            if self._get_dest_name(action) in shape_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Early Stopping Thresholds:")
        print("-" * 40)

        early_args = ["early_stop_loss", "early_stop_lr", "early_stop_stall"]
        for action in self._all_actions:
            if self._get_dest_name(action) in early_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def format_help(self):
        """Override to add section hints and hide experimental/filter args."""
        # Build custom help output
        formatter = self._get_formatter()

        # Add standard arguments only (filter out experimental and filter args)
        standard_actions = []
        for action in self._actions:
            dest = self._get_dest_name(action)
            if (
                dest not in self._experimental_args
                and dest not in self._filter_args
                and dest not in self._advanced_args
            ):
                standard_actions.append(action)

        # Add usage with only standard actions
        formatter.add_usage(
            self.usage, standard_actions, self._mutually_exclusive_groups
        )

        # Add description
        formatter.add_text(self.description)

        # Group standard actions
        formatter.start_section("Standard Options")
        formatter.add_arguments(standard_actions)
        formatter.end_section()

        # Add section hints
        formatter.add_text("")
        formatter.add_text("Additional Help Sections:")
        formatter.add_text(
            "  --help-experimental, -he    Show experimental quantization options"
        )
        formatter.add_text(
            "                              (int8, custom-layers, scaling_mode, etc.)"
        )
        formatter.add_text(
            "  --help-filters, -hf         Show model-specific exclusion filters"
        )
        formatter.add_text(
            "                              (t5xxl, hunyuan, wan, qwen, etc.)"
        )
        formatter.add_text(
            "  --help-advanced, -ha        Show advanced LR tuning and early stopping"
        )
        formatter.add_text(
            "                              (lr-shape-influence, early-stop-*, etc.)"
        )

        # Add epilog
        formatter.add_text(self.epilog)

        return formatter.format_help()


def main():
    parser = MultiHelpArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert safetensors weights to Scaled FP8 format.\n\n"
        "Default behavior: FP8 quantization with per-tensor scaling.\n"
        "For INT8 and other experimental options, see --help-experimental.\n"
        "For model-specific layer exclusions, see --help-filters.\n"
        "For advanced LR tuning and early stopping, see --help-advanced.",
        experimental_args=EXPERIMENTAL_ARGS,
        filter_args=FILTER_ARGS,
        advanced_args=ADVANCED_ARGS,
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input safetensors file path."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output safetensors file path. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--comfy_quant", action="store_true", help="Use Comfy quantization method."
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 block-wise quantization instead of FP8.",
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default=None,
        choices=["fp8", "int8"],
        help="Fallback quantization type for excluded layers (instead of keeping original precision).",
    )
    parser.add_argument(
        "--custom-layers",
        type=str,
        default=None,
        dest="custom_layers",
        help="Regex pattern for layers to quantize with custom type. Takes priority over exclusions.",
    )
    parser.add_argument(
        "--custom-type",
        type=str,
        default=None,
        dest="custom_type",
        choices=["fp8", "int8"],
        help="Quantization type for custom layer matches.",
    )
    # Custom-type parameter overrides
    parser.add_argument(
        "--custom-block-size",
        type=int,
        default=None,
        dest="custom_block_size",
        help="Block size for custom-type layers (default: inherit --block_size)",
    )
    parser.add_argument(
        "--custom-scaling-mode",
        type=str,
        default=None,
        dest="custom_scaling_mode",
        choices=["tensor", "row", "block", "block3d", "block2d"],
        help="FP8 scaling mode for custom-type layers (default: inherit --scaling_mode). 'block2d' is deprecated alias for 'block'.",
    )
    parser.add_argument(
        "--custom-simple",
        action="store_true",
        dest="custom_simple",
        help="Use simple quantization for custom-type layers",
    )
    parser.add_argument(
        "--custom-heur",
        action="store_true",
        dest="custom_heur",
        help="Apply performance heuristics to custom-type layers",
    )
    # Fallback-type parameter overrides
    parser.add_argument(
        "--fallback-block-size",
        type=int,
        default=None,
        dest="fallback_block_size",
        help="Block size for fallback-type layers (default: inherit --block_size)",
    )
    parser.add_argument(
        "--fallback-simple",
        action="store_true",
        dest="fallback_simple",
        help="Use simple quantization for fallback-type layers",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Skip SVD optimization, use simple quantization.",
    )
    parser.add_argument(
        "--full_precision_matrix_mult",
        action="store_true",
        help="Add full_precision_matrix_mult=True to .comfy_quant metadata.",
    )
    parser.add_argument(
        "--heur",
        action="store_true",
        help="Skip layers with poor quantization characteristics (aspect ratio, size).",
    )
    parser.add_argument(
        "--input_scale",
        action="store_true",
        help="Include input_scale tensor (fp32, 1.0) for quantized layers. Works with --convert-fp8-scaled and --convert-int8-scaled. Always enabled for T5XXL.",
    )
    parser.add_argument(
        "--t5xxl",
        action="store_true",
        help="Apply exclusions for T5XXL Text Encoder models.",
    )
    parser.add_argument(
        "--mistral",
        action="store_true",
        help="Apply exclusions for Mistral Text Encoder models.",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Apply exclusions for Visual Text Encoder models.",
    )
    parser.add_argument(
        "--flux2", action="store_true", help="Apply exclusions for Flux2 models."
    )
    parser.add_argument(
        "--distillation_large",
        action="store_true",
        help="Exclude known distillation layers and other sensitive.",
    )
    parser.add_argument(
        "--distillation_small",
        action="store_true",
        help="Exclude known distillation layers.",
    )
    parser.add_argument(
        "--nerf_large",
        action="store_true",
        help="Exclude known NeRF layers, distillation layers and txt_in.",
    )
    parser.add_argument(
        "--nerf_small",
        action="store_true",
        help="Exclude known NeRF layers and distillation layers.",
    )
    parser.add_argument(
        "--radiance", action="store_true", help="Exclude known Radiance Field layers."
    )
    parser.add_argument("--wan", action="store_true", help="Exclude known WAN layers.")
    parser.add_argument(
        "--qwen", action="store_true", help="Exclude known Qwen Image layers."
    )
    parser.add_argument(
        "--hunyuan", action="store_true", help="Exclude known Hunyuan Video 1.5 layers."
    )
    parser.add_argument(
        "--zimage", action="store_true", help="Exclude known Z-Image layers."
    )
    parser.add_argument(
        "--zimage_refiner",
        action="store_true",
        help="Exclude known Z-Image refiner layers (context_refiner, noise_refiner).",
    )
    parser.add_argument(
        "--full_matrix",
        action="store_true",
        help="If should use torch.linalg.svd with full matices instead of the torch.svd_lowrank.",
    )
    parser.add_argument(
        "--scaling_mode",
        type=str,
        default="tensor",
        choices=["tensor", "row", "block", "block3d", "block2d"],
        help="FP8 scaling mode: 'tensor' (1 global scale), 'row' (per-row scale), 'block' (2D tiles like INT8), 'block3d' (per-row-group 3D, legacy). 'block2d' is deprecated alias for 'block'.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for block-wise quantization (REQUIRED for INT8). Common values: 64, 128.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=6144,
        help="Number of random samples for bias correction.",
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=-1,
        help="Set a manual seed for reproducibility. Use -1 for random.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="original",
        choices=["original", "adamw", "radam"],
        help="Optimization algorithm.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1000,
        help="Total optimization iterations per tensor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8.077300000003e-3,
        help="[AdamW/RAdam/Original] Initial learning rate.",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="adaptive",
        choices=["adaptive", "exponential", "plateau"],
        help="LR schedule for 'original' optimizer: 'adaptive' (default custom), 'exponential' (gamma decay), 'plateau' (reduce on stall)",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="[exponential] Decay factor per step (default: 0.99)",
    )
    parser.add_argument(
        "--lr_patience", type=int, default=9, help="[plateau] Steps before decay"
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.92, help="[plateau] LR reduction factor"
    )
    parser.add_argument(
        "--lr_min", type=float, default=1e-10, help="[plateau] Minimum LR bound"
    )
    parser.add_argument(
        "--lr_cooldown",
        type=int,
        default=6,
        help="[plateau] Steps to wait after reduction",
    )
    parser.add_argument(
        "--lr_threshold",
        type=float,
        default=0.0,
        help="[plateau] Min improvement to reset patience",
    )
    parser.add_argument(
        "--lr_adaptive_mode",
        type=str,
        default="simple-reset",
        choices=["simple-reset", "no-reset"],
        help="[adaptive] Counter reset behavior (see MANUAL.md)",
    )
    # Advanced LR tuning (--help-advanced)
    parser.add_argument(
        "--lr-shape-influence",
        type=float,
        default=1.0,
        dest="lr_shape_influence",
        help="[plateau] Scale factor based on tensor aspect ratio. 0.0=disabled, 1.0=full effect. Elongated tensors get more aggressive decay. (default: 1.0)",
    )
    parser.add_argument(
        "--lr-threshold-mode",
        type=str,
        default="rel",
        choices=["rel", "abs"],
        dest="lr_threshold_mode",
        help="[plateau] How to interpret --lr_threshold: 'rel' (relative to best loss) or 'abs' (absolute). (default: rel)",
    )
    # Early stopping thresholds (--help-advanced)
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=1e-8,
        dest="early_stop_loss",
        help="Early stop when loss drops below this value. (default: 1e-8)",
    )
    parser.add_argument(
        "--early-stop-lr",
        type=float,
        default=1e-10,
        dest="early_stop_lr",
        help="Early stop when LR drops below this value. (default: 1e-10)",
    )
    parser.add_argument(
        "--early-stop-stall",
        type=int,
        default=1000,
        dest="early_stop_stall",
        help="Early stop when worse_loss_counter exceeds this. (default: 1000)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.2,
        help="Proportion of principal components (SVD) to use.",
    )
    parser.add_argument(
        "--min_k", type=int, default=64, help="Minimum number of principal components."
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=1024,
        help="Maximum number of principal components.",
    )

    # FP8 scaled to comfy_quant conversion mode
    parser.add_argument(
        "--convert-fp8-scaled",
        action="store_true",
        dest="convert_fp8_scaled",
        help="Convert fp8_scaled model to comfy_quant format (no quantization, just format conversion)",
    )
    parser.add_argument(
        "--hp-filter",
        type=str,
        default=None,
        dest="hp_filter",
        help="Regex pattern for high-precision layers to validate (error if they have FP8 weights)",
    )
    parser.add_argument(
        "--full-precision-mm",
        action="store_true",
        dest="full_precision_mm",
        help="Set full_precision_matrix_mult=True in .comfy_quant metadata (for --convert-fp8-scaled)",
    )

    # INT8 to comfy_quant conversion mode
    parser.add_argument(
        "--convert-int8-scaled",
        action="store_true",
        dest="convert_int8_scaled",
        help="Convert legacy INT8 model (.scale_weight) to comfy_quant format (.weight_scale + metadata)",
    )

    # Legacy input scale addition mode
    parser.add_argument(
        "--legacy_input_add",
        action="store_true",
        help="Add .scale_input tensors to legacy fp8_scaled models (keeps legacy format, adds missing input scales)",
    )

    # Legacy FP8 cleanup mode
    parser.add_argument(
        "--cleanup-fp8-scaled",
        action="store_true",
        dest="cleanup_fp8_scaled",
        help="Clean up legacy fp8_scaled model: remove orphaned scales, set scaled_fp8 marker, normalize scales",
    )
    parser.add_argument(
        "--scaled-fp8-marker",
        type=int,
        default=0,
        choices=[0, 2],
        dest="scaled_fp8_marker",
        help="Size for scaled_fp8 marker tensor: 0=empty((0)), 2=empty((2)). (default: 0)",
    )

    # Activation scale calibration mode
    parser.add_argument(
        "--actcal",
        action="store_true",
        dest="actcal",
        help="Calibrate input_scale values using simulated PTQ. Patches existing FP8 model with computed scales.",
    )
    parser.add_argument(
        "--actcal-samples",
        type=int,
        default=64,
        dest="actcal_samples",
        help="Number of calibration samples for --actcal (default: 64)",
    )
    parser.add_argument(
        "--actcal-percentile",
        type=float,
        default=99.9,
        dest="actcal_percentile",
        help="Percentile for absmax in calibration (default: 99.9, use 100 for true max)",
    )
    parser.add_argument(
        "--actcal-lora",
        dest="actcal_lora",
        help="LoRA file for informed calibration (uses LoRA_A as input directions)",
    )
    parser.add_argument(
        "--actcal-seed",
        type=int,
        default=42,
        dest="actcal_seed",
        help="Random seed for calibration (default: 42). Use for reproducible results.",
    )
    parser.add_argument(
        "--actcal-device",
        type=str,
        default=None,
        dest="actcal_device",
        help="Device for calibration: 'cpu', 'cuda', 'cuda:0', etc. (default: auto-detect CUDA)",
    )

    # Metadata saving option
    parser.add_argument(
        "--save-quant-metadata",
        action="store_true",
        dest="save_quant_metadata",
        help="Save quantization metadata in safetensors header (under _quantization_metadata key)",
    )

    # Scale normalization toggle (for testing)
    parser.add_argument(
        "--no-normalize-scales",
        action="store_true",
        dest="no_normalize_scales",
        help="Disable normalization of 1-element scale arrays to scalars (for testing/compatibility)",
    )


    # ComfyQuant layer config editing mode
    parser.add_argument(
        "--edit-quant",
        action="store_true",
        dest="edit_quant",
        help="Edit .comfy_quant tensors and _quantization_metadata header (add/remove keys)",
    )
    parser.add_argument(
        "--remove-keys",
        type=str,
        default=None,
        dest="remove_keys",
        help="Comma-separated keys to remove (e.g., 'full_precision_matrix_mult,group_size')",
    )
    parser.add_argument(
        "--add-keys",
        type=str,
        default=None,
        dest="add_keys",
        help="Python-like key:value pairs to add (e.g., \"'full_precision_matrix_mult': true, 'group_size': 64\")",
    )
    parser.add_argument(
        "--quant-filter",
        type=str,
        default=None,
        dest="quant_filter",
        help="Regex pattern to filter which layers to edit (default: all layers)",
    )

    # Per-layer quantization config (JSON file)
    parser.add_argument(
        "--layer-config",
        type=str,
        default=None,
        dest="layer_config",
        help="""Path to JSON file with per-layer quantization settings (regex patterns).
Example config:
{
  "_default": {"format": "float8_e4m3fn"},
  "attn": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true},
  "\\\\.0\\\\.img_mod": {"skip": true}
}
By default, patterns use re.search (substring match). Use --fullmatch for full string matching.
In JSON, backslashes must be doubled (\\\\. for literal dot). See DEVELOPMENT.md for details.""",
    )
    parser.add_argument(
        "--fullmatch",
        action="store_true",
        dest="layer_config_fullmatch",
        help="Use re.fullmatch instead of re.search for --layer-config patterns. "
        "With fullmatch, patterns must match the entire layer name (use .* for wildcards).",
    )

    # Dry run / template generation
    parser.add_argument(
        "--dry-run",
        type=str,
        nargs="?",
        const="analyze",
        default=None,
        dest="dry_run",
        choices=["analyze", "create-template"],
        help="Dry run mode: 'analyze' shows what would be processed, 'create-template' generates config template",
    )

    # Verbose output for pinned memory transfers
    parser.add_argument(
        "--verbose-pinned",
        action="store_true",
        dest="verbose_pinned",
        help="Print per-tensor pinned memory transfer details",
    )

    args = parser.parse_args()

    # Set global scale normalization flag from CLI
    global NORMALIZE_SCALES_ENABLED
    NORMALIZE_SCALES_ENABLED = not args.no_normalize_scales

    # Set pinned memory verbosity
    set_pinned_verbose(args.verbose_pinned)

    # Handle dry-run create-template mode (separate workflow)
    if args.dry_run == "create-template":
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        template_path = os.path.splitext(args.input)[0] + "_layer_config_template.json"
        generate_config_template(
            args.input, template_path, block_size=args.block_size or 128
        )
        return

    # Handle fp8_scaled conversion mode first (separate workflow)
    if args.convert_fp8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_fp8mixed.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        convert_fp8_scaled_to_comfy_quant(
            args.input,
            args.output,
            hp_filter=args.hp_filter,
            full_precision_mm=args.full_precision_mm,
            include_input_scale=args.input_scale,
        )
        return

    # Handle int8 to comfy_quant conversion mode (separate workflow)
    if args.convert_int8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_int8_comfy.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        # Use block_size from args or default to 128
        int8_block_size = args.block_size if args.block_size else 128

        convert_int8_to_comfy_quant(
            args.input,
            args.output,
            block_size=int8_block_size,
            include_input_scale=args.input_scale,
            save_quant_metadata=args.save_quant_metadata,
        )
        return

    # Handle legacy input scale addition mode (separate workflow)
    if args.legacy_input_add:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_with_input_scale.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        add_legacy_input_scale(args.input, args.output)
        return

    # Handle legacy FP8 cleanup mode (separate workflow)
    if args.cleanup_fp8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_cleaned.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        cleanup_fp8_scaled(
            args.input,
            args.output,
            marker_size=args.scaled_fp8_marker,
            add_scale_input=args.input_scale,
        )
        return

    # Handle activation scale calibration mode (separate workflow)
    if args.actcal:
        try:
            from .calibrate_activation_scales import calibrate_model, patch_model_with_scales, load_lora_tensors
        except ImportError:
            from calibrate_activation_scales import calibrate_model, patch_model_with_scales, load_lora_tensors
        
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_calibrated.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        print(f"Loading model: {args.input}")
        tensors = load_file(args.input)
        print(f"  Total tensors: {len(tensors)}")

        # Load LoRA if specified
        lora_tensors = None
        if args.actcal_lora:
            if not os.path.exists(args.actcal_lora):
                print(f"Error: LoRA file not found: {args.actcal_lora}")
                return
            print(f"\nLoading LoRA: {args.actcal_lora}")
            lora_tensors = load_lora_tensors(args.actcal_lora)
            print(f"  LoRA layers found: {len(lora_tensors)}")

        mode = "LoRA-informed" if lora_tensors else "random"
        print(f"\nCalibrating input_scale using {mode} PTQ ({args.actcal_samples} samples)...")
        scales = calibrate_model(
            tensors,
            calib_samples=args.actcal_samples,
            seed=args.actcal_seed,
            percentile=args.actcal_percentile,
            verbose=True,
            lora_tensors=lora_tensors,
            device=args.actcal_device,
        )
        print(f"\nCalibrated {len(scales)} layers")

        print(f"\nPatching model with calibrated scales...")
        patched = patch_model_with_scales(tensors, scales)

        print(f"Saving to: {args.output}")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_file(patched, args.output)
        print("Done!")
        return

    # Handle comfy_quant editing mode (separate workflow)
    if args.edit_quant:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_edited.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        if not args.remove_keys and not args.add_keys and not args.save_quant_metadata:
            print(
                "Error: --edit-quant requires at least one of --remove-keys, --add-keys, or --save-quant-metadata"
            )
            return

        # Parse remove_keys from comma-separated string
        remove_keys_list = None
        if args.remove_keys:
            remove_keys_list = [
                k.strip() for k in args.remove_keys.split(",") if k.strip()
            ]

        edit_comfy_quant(
            args.input,
            args.output,
            remove_keys=remove_keys_list,
            add_keys_str=args.add_keys,
            layer_filter=args.quant_filter,
            save_quant_metadata=args.save_quant_metadata,
        )
        return

    # Determine which formats require block_size
    primary_needs_block_size = args.int8
    custom_needs_block_size = args.custom_type == "int8"
    fallback_needs_block_size = args.fallback == "int8"

    # Validate block_size for primary format
    if primary_needs_block_size and args.block_size is None:
        print("Error: --block_size is required when using INT8 quantization.")
        print("       Example: --block_size 128")
        sys.exit(1)

    # Validate custom-block-size for custom format
    if args.custom_type and custom_needs_block_size and args.custom_block_size is None:
        print(
            f"Error: --custom-block-size is required when using --custom-type {args.custom_type}."
        )
        print("       Example: --custom-block-size 128")
        sys.exit(1)

    # Validate fallback-block-size for fallback format
    if args.fallback and fallback_needs_block_size and args.fallback_block_size is None:
        print(
            f"Error: --fallback-block-size is required when using --fallback {args.fallback}."
        )
        print("       Example: --fallback-block-size 128")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Auto-enable comfy_quant if custom-type is used (required for mixed precision)
    if args.custom_type and not args.comfy_quant:
        print(
            "Note: --comfy_quant auto-enabled (required for --custom-type mixed precision)"
        )
        args.comfy_quant = True

    # Only check FP8 support if not using INT8
    if not args.int8:
        try:
            _ = torch.zeros(
                1,
                dtype=TARGET_FP8_DTYPE,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except (RuntimeError, TypeError):
            print(
                "Error: This hardware/PyTorch version does not support the target FP8 dtype."
            )
            return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        if args.int8:
            format_str = "int8_blockwise"
            scaling_str = f"_bs{args.block_size}"
        else:
            format_str = TARGET_FP8_DTYPE.__str__().split(".")[-1]
            scaling_str = f"_{args.scaling_mode}"
        flags = "".join(
            [
                "_t5" if args.t5xxl else "",
                "_mistral" if args.mistral else "",
                "_visual" if args.visual else "",
                "_flux2" if args.flux2 else "",
                "_nodist_l" if args.distillation_large else "",
                "_nodist_s" if args.distillation_small else "",
                "_nonerf_l" if args.nerf_large else "",
                "_nonerf_s" if args.nerf_small else "",
                "_norad" if args.radiance else "",
            ]
        )
        output_file = f"{base}_{format_str}{scaling_str}{flags}_k{args.min_k}-{args.max_k}_p{args.top_p}_lr{args.lr}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = (
        int(torch.randint(0, 2**32 - 1, ()).item())
        if args.manual_seed == -1
        else args.manual_seed
    )
    print(f"Using seed: {seed}")

    # Separate converter kwargs from function kwargs
    excluded_keys = [
        "input",
        "output",
        "comfy_quant",
        "t5xxl",
        "mistral",
        "visual",
        "flux2",
        "distillation_large",
        "distillation_small",
        "nerf_large",
        "nerf_small",
        "radiance",
        "wan",
        "qwen",
        "hunyuan",
        "zimage",
        "zimage_refiner",
        "calib_samples",
        "manual_seed",
        "int8",
        "fallback",
        "custom_layers",
        "custom_type",
        "custom_block_size",
        "custom_scaling_mode",
        "custom_simple",
        "custom_heur",
        "fallback_block_size",
        "fallback_simple",
        "full_precision_matrix_mult",
        "heur",
        "input_scale",
        "simple",
        "layer_config",
        "layer_config_fullmatch",
        "save_quant_metadata",
    ]
    converter_kwargs = {k: v for k, v in vars(args).items() if k not in excluded_keys}

    # Load layer config if specified
    layer_config_data = None
    if args.layer_config:
        layer_config_data = load_layer_config(args.layer_config)

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.comfy_quant,
        args.t5xxl,
        args.mistral,
        args.visual,
        args.flux2,
        args.distillation_large,
        args.distillation_small,
        args.nerf_large,
        args.nerf_small,
        args.radiance,
        args.wan,
        args.qwen,
        args.hunyuan,
        args.zimage,
        args.zimage_refiner,
        args.calib_samples,
        seed,
        int8=args.int8,
        fallback=args.fallback,
        custom_layers=args.custom_layers,
        custom_type=args.custom_type,
        custom_block_size=args.custom_block_size,
        custom_scaling_mode=args.custom_scaling_mode,
        custom_simple=args.custom_simple,
        custom_heur=args.custom_heur,
        fallback_block_size=args.fallback_block_size,
        fallback_simple=args.fallback_simple,
        full_precision_matrix_mult=args.full_precision_matrix_mult,
        skip_inefficient_layers=args.heur,
        include_input_scale=args.input_scale,
        no_learned_rounding=args.simple,
        layer_config=layer_config_data,
        layer_config_fullmatch=args.layer_config_fullmatch,
        save_quant_metadata=args.save_quant_metadata,
        **converter_kwargs,
    )


if __name__ == "__main__":
    main()
