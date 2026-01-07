"""
convert_to_quant - Quantization toolkit for safetensors models.

This module provides tools for converting model weights to FP8 and INT8
quantized formats with optional learned rounding optimization.

Main entry point for CLI and programmatic usage.
"""
# Re-export constants for backward compatibility
from .constants import (
    AVOID_KEY_NAMES,
    T5XXL_REMOVE_KEY_NAMES,
    VISUAL_AVOID_KEY_NAMES,
    QWEN_AVOID_KEY_NAMES,
    HUNYUAN_AVOID_KEY_NAMES,
    ZIMAGE_AVOID_KEY_NAMES,
    FLUX2_LAYER_KEYNAMES,
    DISTILL_LAYER_KEYNAMES_LARGE,
    DISTILL_LAYER_KEYNAMES_SMALL,
    NERF_LAYER_KEYNAMES_LARGE,
    NERF_LAYER_KEYNAMES_SMALL,
    RADIANCE_LAYER_KEYNAMES,
    WAN_LAYER_KEYNAMES,
    QWEN_LAYER_KEYNAMES,
    ZIMAGE_LAYER_KEYNAMES,
    ZIMAGE_REFINER_LAYER_KEYNAMES,
    TARGET_FP8_DTYPE,
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MIN,
    FP8_MAX,
    FP8_MIN_POS,
    INT8_MIN,
    INT8_MAX,
    INT8_SYMMETRIC_MAX,
    VALID_QUANT_FORMATS,
    NORMALIZE_SCALES_ENABLED,
    # New registry
    MODEL_FILTERS,
    build_exclusion_patterns,
)

# Re-export utils
from .utils import (
    dict_to_tensor,
    tensor_to_dict,
    normalize_tensorwise_scales,
    create_comfy_quant_tensor,
    fix_comfy_quant_params_structure,
    parse_add_keys_string,
    edit_comfy_quant,
    should_skip_layer_for_performance,
)

# Re-export config
from .config import (
    pattern_specificity,
    load_layer_config,
    get_layer_settings,
    generate_config_template,
)

# Re-export converters
from .converters import LearnedRoundingConverter

# Re-export formats
from .formats import (
    convert_to_fp8_scaled,
    convert_fp8_scaled_to_comfy_quant,
    convert_int8_to_comfy_quant,
    add_legacy_input_scale,
    cleanup_fp8_scaled,
)

# Re-export CLI
from .cli import main

__all__ = [
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
