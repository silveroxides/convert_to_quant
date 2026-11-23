"""
W6A8 ConvRot conversion functions for convert_to_quant.

Converts safetensors models to W6A8 (6-bit uniform ConvRot weights, INT8 activations)
format with group-wise Hadamard rotation and dual scaling (weight_s_rel + weight_s_channel)
for ComfyUI w6a8_int8 (AsymW4A8Int8Layout) inference.
"""

import json
import os
import re
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from ..constants import COMPUTE_DTYPE, NORMALIZE_SCALES_ENABLED, SCALE_DTYPE, W6A8_DEFAULT_CONVROT_GROUP_SIZE, W6A8_DEFAULT_GROUP_SIZE, W6A8_SCALE_REL_DTYPE, resolve_model_filter_patterns
from ..converters.w6a8_converter import W6A8Converter
from ..utils.comfy_quant import create_comfy_quant_tensor
from ..utils.convrot import resolve_w6a8_convrot_group_size
from ..utils.logging import error, info, log_debug, minimal
from ..utils.memory_efficient_loader import UnifiedSafetensorsLoader
from ..utils.output_dtype import cast_unquantized_weights, compile_preserve_layers, resolve_output_dtype
from ..utils.tensor_utils import normalize_tensorwise_scales


@log_debug
def convert_to_w6a8(
    input_file: str,
    output_file: str,
    # Filter flags
    filter_flags: Optional[Dict[str, bool]] = None,
    exclude_layers: Optional[str] = None,
    # Quantization options
    simple: bool = True,
    convrot: bool = True,
    group_size: int = W6A8_DEFAULT_GROUP_SIZE,
    convrot_group_size: int = W6A8_DEFAULT_CONVROT_GROUP_SIZE,
    dynamic_convrot: bool = True,
    scale_dtype: torch.dtype = W6A8_SCALE_REL_DTYPE,
    scale_search: bool = True,
    calib_scale: float = 1.0,
    device: Optional[str] = None,
    # Calibration options for bias correction
    calib_samples: int = 256,
    seed: int = 42,
    # Metadata & preservation
    comfy_quant: bool = True,
    save_quant_metadata: bool = True,
    full_precision_matrix_mult: bool = False,
    low_memory: bool = False,
    output_dtype: str = "bfloat16",
    preserve_layers: Optional[str] = None,
    # Layer config overrides
    layer_config: Optional[dict] = None,
    layer_config_fullmatch: bool = False,
    **kwargs,
):
    """
    Convert a safetensors checkpoint to ComfyUI W6A8 (w6a8_int8) format.
    """
    if not os.path.exists(input_file):
        error(f"ERROR: Input file does not exist: {input_file}")
        return

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    info("Target format: W6A8 (w6a8_int8, uniform 6-bit ConvRot + INT8 activations)")
    info(f"Group size: {group_size}")
    info(f"ConvRot group size: {convrot_group_size}")
    info(f"Device: {target_device}")
    info("-" * 60)

    filter_flags = filter_flags or {}
    exclude_pattern = None
    if exclude_layers:
        try:
            exclude_pattern = re.compile(exclude_layers)
            info(f"Layer exclusion pattern enabled: '{exclude_layers}'")
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{exclude_layers}': {e}")
            return

    resolved_output_dtype = resolve_output_dtype(output_dtype)
    preserve_compiled = compile_preserve_layers(preserve_layers)

    loader = UnifiedSafetensorsLoader(input_file, low_memory=low_memory)
    all_keys = loader.keys()
    resolved_filter_patterns = resolve_model_filter_patterns(filter_flags, all_keys)
    existing_meta = loader.metadata()

    # Pre-generate simulated calibration data by in_features if needed
    calibration_data_cache: Dict[int, torch.Tensor] = {}
    torch.manual_seed(seed)
    seed_generator = torch.Generator(device="cpu")
    seed_generator.manual_seed(seed)
    minimal("Scanning model and preparing calibration data for bias adjustments...")
    for key in all_keys:
        if key.endswith(".weight") and loader.get_ndim(key) == 2:
            shape = loader.get_shape(key)
            in_features = shape[1]
            if in_features not in calibration_data_cache:
                calib_tensor = torch.randn(
                    calib_samples,
                    in_features,
                    dtype=COMPUTE_DTYPE,
                    generator=seed_generator,
                    device="cpu",
                )
                calibration_data_cache[in_features] = calib_tensor

    converter = W6A8Converter(
        convrot=convrot,
        convrot_group_size=convrot_group_size,
        group_size=group_size,
        dynamic_convrot=dynamic_convrot,
        scale_dtype=scale_dtype,
        scale_search=scale_search,
        device=target_device,
        no_learned_rounding=simple,
        calib_scale=calib_scale,
        has_bias=True,
        **kwargs,
    )

    weight_keys = sorted([k for k in all_keys if k.endswith(".weight") and loader.get_ndim(k) == 2])
    total_weights = len(weight_keys)
    info(f"Found {total_weights} 2D weight tensors to process.")

    new_tensors: Dict[str, torch.Tensor] = {}
    quant_metadata_layers: Optional[Dict[str, dict]] = {} if save_quant_metadata else None
    quantized_weight_keys = set()
    skipped_count = 0
    processed_count = 0

    for i, key in enumerate(weight_keys):
        base_name = key[:-7]  # strip '.weight'
        bias_key = f"{base_name}.bias"
        has_bias = bias_key in all_keys

        # Check explicit layer config skip
        if layer_config:
            from ..config.layer_config import get_layer_settings

            cfg = get_layer_settings(key, layer_config, fullmatch=layer_config_fullmatch)
            if cfg and cfg.get("skip", False):
                info(f"({i + 1}/{total_weights}) Skipping (layer-config): {key}")
                new_tensors[key] = loader.get_tensor(key).to(device="cpu")
                loader.mark_processed(key)
                skipped_count += 1
                continue

        # Check filter exclusions
        is_excluded = False
        exclusion_reason = ""
        for filter_name, skip_patterns in resolved_filter_patterns.items():
            if skip_patterns and any(p in key for p in skip_patterns):
                is_excluded = True
                exclusion_reason = f"{filter_name} filter"
                break

        if not is_excluded and exclude_pattern and exclude_pattern.search(key):
            is_excluded = True
            exclusion_reason = "regex exclusion"

        if is_excluded:
            info(f"({i + 1}/{total_weights}) Skipping (exclusion: {exclusion_reason}): {key}")
            new_tensors[key] = loader.get_tensor(key).to(device="cpu")
            loader.mark_processed(key)
            skipped_count += 1
            continue

        original_tensor = loader.get_tensor(key)
        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            new_tensors[key] = original_tensor.to(device="cpu")
            skipped_count += 1
            continue

        in_features = original_tensor.shape[1]
        resolved_convrot_gs = resolve_w6a8_convrot_group_size(
            in_features,
            requested_group_size=convrot_group_size,
            dynamic=dynamic_convrot,
        )
        if in_features % 32 != 0 or in_features % group_size != 0 or resolved_convrot_gs is None:
            info(
                f"({i + 1}/{total_weights}) Skipping (incompatible shape K={in_features} for "
                f"group_size={group_size}, convrot_group_size={convrot_group_size}): {key}"
            )
            new_tensors[key] = original_tensor.to(device="cpu")
            loader.mark_processed(key)
            skipped_count += 1
            continue

        calib_data = calibration_data_cache.get(in_features)

        info(f"({i + 1}/{total_weights}) Processing (W6A8 w6a8_int8): {key}")
        q_tensor, s_rel, s_channel, dequant_w, extra_tensors = converter.convert(
            original_tensor,
            key=key,
            depth=-1,
            calibration_data=calib_data,
            has_bias=has_bias,
        )

        new_tensors[key] = q_tensor.to(device="cpu")
        new_tensors[f"{base_name}.weight_s_rel"] = s_rel.to(device="cpu")
        new_tensors[f"{base_name}.weight_s_channel"] = s_channel.to(device="cpu", dtype=SCALE_DTYPE)
        quantized_weight_keys.add(key)
        processed_count += 1
        quantized_orig_dtype = str(resolved_output_dtype)
        used_convrot_gs = converter.last_group_size or resolved_convrot_gs
        used_quant_gs = converter.last_quant_group_size or group_size

        if comfy_quant:
            comfy_meta = create_comfy_quant_tensor(
                "w6a8_int8",
                block_size=used_quant_gs,
                convrot_groupsize=used_convrot_gs,
                full_precision_matrix_mult=full_precision_matrix_mult if full_precision_matrix_mult else None,
                orig_dtype=quantized_orig_dtype,
            )
            new_tensors[f"{base_name}.comfy_quant"] = comfy_meta

        if save_quant_metadata and quant_metadata_layers is not None:
            meta_entry = {
                "format": "w6a8_int8",
                "orig_dtype": quantized_orig_dtype,
                "group_size": used_quant_gs,
                "convrot_groupsize": used_convrot_gs,
            }
            if full_precision_matrix_mult:
                meta_entry["full_precision_matrix_mult"] = True
            quant_metadata_layers[base_name] = meta_entry

        # Final shape outputs
        info(f"    - Final Weight shape        : {list(q_tensor.shape)}")
        info(f"    - Final W6A8 s_rel shape    : {list(s_rel.shape)} ({s_rel.dtype})")
        info(f"    - Final W6A8 s_channel shape: {list(s_channel.shape)}")
        info("-" * 60)

        # Apply residual bias correction if generated
        if has_bias and "bias_correction" in extra_tensors:
            with torch.no_grad():
                original_bias = loader.get_tensor(bias_key)
                bias_adj = extra_tensors["bias_correction"].to(dtype=COMPUTE_DTYPE)
                corrected_bias = (original_bias.to(dtype=COMPUTE_DTYPE) + bias_adj).to(dtype=original_bias.dtype)
                new_tensors[bias_key] = corrected_bias.cpu()
                loader.mark_processed(bias_key)
                info(f"  - Applied residual bias correction to: {bias_key}")

        loader.mark_processed(key)
        del original_tensor, q_tensor, s_rel, s_channel, dequant_w
        if target_device == "cuda":
            torch.cuda.empty_cache()

    # Copy remaining tensors (bias, norms, embeddings, etc.)
    for key in all_keys:
        if key not in new_tensors:
            new_tensors[key] = loader.get_tensor(key)

    cast_count = cast_unquantized_weights(
        new_tensors,
        quantized_weight_keys,
        resolved_output_dtype,
        preserve_compiled,
        filter_flags,
    )
    if cast_count > 0:
        info(f"Cast {cast_count} unquantized tensors to {resolved_output_dtype}.")

    new_tensors, _ = normalize_tensorwise_scales(new_tensors, NORMALIZE_SCALES_ENABLED)

    # Metadata assembly matching ComfyUI format_version 1.0 schema
    final_meta = dict(existing_meta) if existing_meta else {}
    if save_quant_metadata and quant_metadata_layers:
        full_metadata = {
            "format_version": "1.0",
            "layers": quant_metadata_layers,
        }
        final_meta["_quantization_metadata"] = json.dumps(full_metadata)

    loader.close()
    info(f"Saving {len(new_tensors)} tensors to {output_file}...")
    save_file(new_tensors, output_file, metadata=final_meta)
    info(f"Successfully converted model to W6A8 (w6a8_int8): {output_file}")
    info("-" * 60)
    info("Summary:")
    info(f"  - Original tensor count : {len(all_keys)}")
    info(f"  - Weights processed     : {processed_count}")
    info(f"  - Weights skipped       : {skipped_count}")
    info(f"  - Final tensor count    : {len(new_tensors)}")
    info("-" * 60)
    info("Conversion complete!")
