"""
Layer configuration loading and matching for convert_to_quant.

Provides regex-based layer pattern matching for per-layer quantization settings.
"""
import json
import os
import re
import torch
from typing import Dict, Any, Optional, Tuple
from safetensors import safe_open
from tqdm import tqdm

from ..constants import VALID_QUANT_FORMATS

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
    literal_pattern = re.sub(r"\\.|\\[.*?\\]|\\(.*?\\)|[.*+?^${}|\\\\]", "", pattern)
    literal_len = len(literal_pattern)

    # Check if pattern has explicit numbers (literal digits or \\d patterns)
    has_number = bool(re.search(r"\\d|\\\\d", pattern))

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
