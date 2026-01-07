"""
FP8 conversion functions for convert_to_quant.

Main quantization function that processes safetensors files and applies
FP8/INT8 quantization with learned rounding optimization.
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
    TARGET_INT8_DTYPE,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    FP8_MIN,
    FP8_MAX,
    INT8_SYMMETRIC_MAX,
    AVOID_KEY_NAMES,
    T5XXL_REMOVE_KEY_NAMES,
    MODEL_FILTERS,
    VALID_QUANT_FORMATS,
    NORMALIZE_SCALES_ENABLED,
)
from ..converters.learned_rounding import LearnedRoundingConverter
from ..config.layer_config import get_layer_settings
from ..utils.tensor_utils import normalize_tensorwise_scales
from ..utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from ..utils.memory_efficient_loader import MemoryEfficientSafeOpen
from ..pinned_transfer import get_pinned_transfer_stats

def convert_to_fp8_scaled(
    input_file: str,
    output_file: str,
    comfy_quant: bool,
    filter_flags: Dict[str, bool],
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
    low_memory: bool = False,
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

    # Use unified loader (handles both standard and low-memory modes)
    try:
        loader = MemoryEfficientSafeOpen(input_file, low_memory=low_memory)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    all_keys = loader.keys()

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
    for key in all_keys:
        if key.endswith(".weight"):
            shape = loader.get_shape(key)
            if len(shape) == 2:
                in_features = shape[1]
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
            for key in all_keys
            if key.endswith(".weight") and loader.get_ndim(key) == 2
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

        # Pre-compute text encoder filter for input scale handling
        text_encoder_filter = (
            filter_flags.get("t5xxl") or
            filter_flags.get("mistral") or
            filter_flags.get("visual")
        )

        # T5XXL decoder tensors are always removed (not quantized, not kept)
        if filter_flags.get("t5xxl") and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
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
                    original_tensor = loader.get_tensor(key)
                    new_tensors[key] = original_tensor.to(
                        device="cpu", dtype=original_tensor.dtype
                    )
                    loader.mark_processed(key)
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
        # Uses MODEL_FILTERS registry for centralized filter definitions
        if not use_custom and not use_layer_config:
            # Use filter_flags dict passed from CLI
            active_filters = filter_flags

            # Check each active filter against the key
            for filter_name, is_active in active_filters.items():
                if not is_active:
                    continue
                cfg = MODEL_FILTERS[filter_name]

                # Check "exclude" patterns (layers to skip entirely)
                exclude_patterns = cfg.get("exclude", [])
                if exclude_patterns and any(n in key for n in exclude_patterns):
                    exclusion_reason = f"{filter_name} exclusion"
                    break

                # Check "highprec" patterns (layers to keep in high precision)
                highprec_patterns = cfg.get("highprec", [])
                if highprec_patterns and any(n in key for n in highprec_patterns):
                    exclusion_reason = f"{filter_name} keep in high precision"
                    break

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
                original_tensor = loader.get_tensor(key)
                new_tensors[key] = original_tensor.to(
                    device="cpu", dtype=original_tensor.dtype
                )
                loader.mark_processed(key)
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
        original_tensor = loader.get_tensor(key)

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
                loader.mark_processed(key)
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
            if include_input_scale or text_encoder_filter:
                if text_encoder_filter:
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

        if bias_key in all_keys:
            if layer_uses_simple:
                # Skip bias correction for simple mode (saves memory, avoids OOM on large layers)
                print(f"  - Keeping original bias (simple mode): {bias_key}")
                new_tensors[bias_key] = loader.get_tensor(bias_key)
            else:
                print(f"  - Adjusting corresponding bias: {bias_key}")
                with torch.no_grad():
                    original_bias = loader.get_tensor(bias_key)
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
        if text_encoder_filter:
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

    # Copy remaining tensors (bias, norms, etc.)
    for key in all_keys:
        if any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and t5xxl:
            continue
        if key not in new_tensors:
            new_tensors[key] = loader.get_tensor(key)

    # Close loader to release file handle
    loader.close()

    # Free calibration data and force garbage collection before save
    calibration_data_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Add scaled_fp8 marker only for legacy non-comfy_quant FP8 format
    # Use empty((0)) when input_scale is present (t5xxl, mistral, or --input_scale flag)
    if (
        not comfy_quant
        and not int8
        and not custom_layers
        and "scaled_fp8" not in new_tensors
    ):
        has_text_encoder_filter = (
            filter_flags.get("t5xxl") or
            filter_flags.get("mistral") or
            filter_flags.get("visual")
        )
        new_tensors["scaled_fp8"] = (
            torch.empty((0), dtype=TARGET_FP8_DTYPE)
            if (has_text_encoder_filter or include_input_scale)
            else torch.empty((2), dtype=TARGET_FP8_DTYPE)
        )

    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

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
        f"  - Original tensor count : {len(all_keys)}",
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
