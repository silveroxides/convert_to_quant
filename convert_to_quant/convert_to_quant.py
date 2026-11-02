"""
convert_to_quant - Quantization toolkit for safetensors models.

This module provides tools for converting model weights to FP8 and INT8
quantized formats with optional learned rounding optimization.

Main entry point for CLI and programmatic usage.
"""

# Re-export CLI
from .cli import main

# Re-export config
from .config import generate_config_template, get_layer_settings, load_layer_config, pattern_specificity

# Re-export constants for backward compatibility
from .constants import (  # New registry
    AVOID_KEY_NAMES,
    COMPUTE_DTYPE,
    DISTILL_LAYER_KEYNAMES_LARGE,
    DISTILL_LAYER_KEYNAMES_SMALL,
    FLUX2_LAYER_KEYNAMES,
    FP8_MAX,
    FP8_MIN,
    FP8_MIN_POS,
    HUNYUAN_AVOID_KEY_NAMES,
    INT8_MAX,
    INT8_MIN,
    INT8_SYMMETRIC_MAX,
    MODEL_FILTERS,
    NERF_LAYER_KEYNAMES_LARGE,
    NERF_LAYER_KEYNAMES_SMALL,
    NORMALIZE_SCALES_ENABLED,
    QWEN_AVOID_KEY_NAMES,
    QWEN_LAYER_KEYNAMES,
    RADIANCE_LAYER_KEYNAMES,
    SCALE_DTYPE,
    T5XXL_REMOVE_KEY_NAMES,
    TARGET_FP8_DTYPE,
    TARGET_INT8_DTYPE,
    VALID_QUANT_FORMATS,
    VISUAL_AVOID_KEY_NAMES,
    WAN_LAYER_KEYNAMES,
    ZIMAGE_AVOID_KEY_NAMES,
    ZIMAGE_LAYER_KEYNAMES,
    ZIMAGE_REFINER_LAYER_KEYNAMES,
    build_exclusion_patterns,
)

# Re-export converters
from .converters import LearnedRoundingConverter

# Re-export formats
from .formats import add_legacy_input_scale, cleanup_fp8_scaled, convert_fp8_scaled_to_comfy_quant, convert_int8_to_comfy_quant, convert_to_fp8_scaled

# Re-export utils
from .utils import create_comfy_quant_tensor, dict_to_tensor, edit_comfy_quant, fix_comfy_quant_params_structure, normalize_tensorwise_scales, parse_add_keys_string, should_skip_layer_for_performance, tensor_to_dict

def quantize(input: str, output: str = None, **kwargs):
    """
    Programmatic entry point for quantizing a model.
    Accepts the exact same parameters as the CLI tool.

    Args:
        input (str): Path to input safetensors file.
        output (str, optional): Path to output safetensors file. Auto-generated if not provided.
        **kwargs: Additional arguments matching CLI flags (e.g., int8=True, block_size=128, etc.).
    """
    import argparse
    from .cli.main import get_parser, run_conversion

    parser = get_parser()

    # Extract defaults from parser
    defaults = {}
    for action in parser._actions:
        if action.dest != "help":
            defaults[action.dest] = action.default

    # Update defaults with input/output
    defaults["input"] = input
    defaults["output"] = output

    # Update defaults with provided kwargs
    valid_keys = set(defaults.keys())
    for k, v in kwargs.items():
        if k not in valid_keys:
            raise ValueError(f"Unknown parameter: '{k}'. Valid parameters match CLI arguments: {list(valid_keys)}")
        defaults[k] = v

    # Create namespace
    args = argparse.Namespace(**defaults)

    # Run conversion
    run_conversion(args)


__all__ = [
    "quantize",
    # Constants
    "AVOID_KEY_NAMES",
    "T5XXL_REMOVE_KEY_NAMES",
    "VISUAL_AVOID_KEY_NAMES",
    "QWEN_AVOID_KEY_NAMES",
    "HUNYUAN_AVOID_KEY_NAMES",
    "ZIMAGE_AVOID_KEY_NAMES",
    "FLUX2_LAYER_KEYNAMES",
    "DISTILL_LAYER_KEYNAMES_LARGE",
    "DISTILL_LAYER_KEYNAMES_SMALL",
    "NERF_LAYER_KEYNAMES_LARGE",
    "NERF_LAYER_KEYNAMES_SMALL",
    "RADIANCE_LAYER_KEYNAMES",
    "WAN_LAYER_KEYNAMES",
    "QWEN_LAYER_KEYNAMES",
    "ZIMAGE_LAYER_KEYNAMES",
    "ZIMAGE_REFINER_LAYER_KEYNAMES",
    "TARGET_FP8_DTYPE",
    "TARGET_INT8_DTYPE",
    "COMPUTE_DTYPE",
    "SCALE_DTYPE",
    "FP8_MIN",
    "FP8_MAX",
    "FP8_MIN_POS",
    "INT8_MIN",
    "INT8_MAX",
    "INT8_SYMMETRIC_MAX",
    "VALID_QUANT_FORMATS",
    "NORMALIZE_SCALES_ENABLED",
    # New registry
    "MODEL_FILTERS",
    "build_exclusion_patterns",
    # Utils
    "dict_to_tensor",
    "tensor_to_dict",
    "normalize_tensorwise_scales",
    "create_comfy_quant_tensor",
    "fix_comfy_quant_params_structure",
    "parse_add_keys_string",
    "edit_comfy_quant",
    "should_skip_layer_for_performance",
    # Config
    "pattern_specificity",
    "load_layer_config",
    "get_layer_settings",
    "generate_config_template",
    # Converters
    "LearnedRoundingConverter",
    # Formats
    "convert_to_fp8_scaled",
    "convert_fp8_scaled_to_comfy_quant",
    "convert_int8_to_comfy_quant",
    "add_legacy_input_scale",
    "cleanup_fp8_scaled",
    # CLI
    "main",
]

if __name__ == "__main__":
    main()
