"""
INT4 ConvRot W4A4 conversion functions for convert_to_quant.

Converts safetensors models to INT4 ConvRot W4A4 format with
group-wise Hadamard rotation and row-wise symmetric INT4 scaling
for ComfyUI convrot_w4a4 inference.
"""

import gc
import os
import re
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from ..constants import (
    AVOID_KEY_NAMES,
    COMPUTE_DTYPE,
    SCALE_DTYPE,
    resolve_model_filter_patterns,
)
from ..converters.int4_converter import INT4Converter
from ..utils.comfy_quant import create_comfy_quant_tensor, should_skip_layer_for_performance
from ..utils.logging import error, info, log_debug, minimal, verbose, warning
from ..utils.memory_efficient_loader import UnifiedSafetensorsLoader
from ..utils.output_dtype import cast_unquantized_weights, compile_preserve_layers, resolve_output_dtype
from ..utils.tensor_utils import dict_to_tensor


@log_debug
def convert_to_int4(
    input_file: str,
    output_file: str,
    # Filter flags
    filter_flags: Optional[Dict[str, bool]] = None,
    exclude_layers: Optional[str] = None,
    # Quantization options
    simple: bool = True,
    convrot: bool = True,
    convrot_group_size: int = 64,
    dynamic_convrot: bool = True,
    calib_scale: float = 1.0,
    device: Optional[str] = None,
    # Calibration options for bias correction
    calib_samples: int = 256,
    seed: int = 42,
    # Metadata & preservation
    comfy_quant: bool = True,
    save_quant_metadata: bool = True,
    full_precision_mm: bool = False,
    hp_filter: bool = False,
    low_memory: bool = False,
    output_dtype: str = "bfloat16",
    preserve_layers: Optional[str] = None,
    # Layer config overrides
    layer_config: Optional[dict] = None,
    layer_config_fullmatch: bool = False,
    **kwargs,
):
    """
    Convert a safetensors checkpoint to INT4 ConvRot W4A4 format.
    """
    if not os.path.exists(input_file):
        error(f"ERROR: Input file does not exist: {input_file}")
        return

    out_dir = os.path.dirname(output_file)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    info(f"Converting to INT4 ConvRot W4A4 on device: {target_device}")

    # Build exclusion filters
    filter_flags = filter_flags or {}
    resolved_filter_patterns = resolve_model_filter_patterns(filter_flags)
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
    existing_meta = loader.metadata()

    # Pre-generate simulated calibration data by in_features if needed
    calibration_data_cache: Dict[int, torch.Tensor] = {}
    torch.manual_seed(seed)
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
                    device="cpu",
                )
                calibration_data_cache[in_features] = calib_tensor

    converter = INT4Converter(
        convrot=convrot,
        convrot_group_size=convrot_group_size,
        dynamic_convrot=dynamic_convrot,
        device=target_device,
        no_learned_rounding=simple,
        calib_scale=calib_scale,
        has_bias=True,
    )

    weight_keys = sorted([k for k in all_keys if k.endswith(".weight") and loader.get_ndim(k) == 2])
    total_weights = len(weight_keys)
    info(f"Found {total_weights} 2D weight tensors to process.")

    new_tensors: Dict[str, torch.Tensor] = {}
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
        calib_data = calibration_data_cache.get(in_features)

        info(f"({i + 1}/{total_weights}) Processing (INT4 ConvRot W4A4): {key}")
        q_tensor, scale, dequant_w, extra_tensors = converter.convert(
            original_tensor,
            key=key,
            depth=-1,
            calibration_data=calib_data,
            has_bias=has_bias,
        )

        new_tensors[key] = q_tensor.to(device="cpu")
        new_tensors[f"{base_name}.weight_scale"] = scale.to(device="cpu", dtype=SCALE_DTYPE)
        quantized_weight_keys.add(key)
        processed_count += 1

        if comfy_quant:
            used_group_size = converter.last_group_size or convrot_group_size
            comfy_meta = create_comfy_quant_tensor(
                "convrot_w4a4",
                convrot=True,
                convrot_groupsize=used_group_size,
                full_precision_matrix_mult=full_precision_mm if full_precision_mm else None,
            )
            new_tensors[f"{base_name}.comfy_quant"] = comfy_meta

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
        del original_tensor, q_tensor, scale, dequant_w
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

    # Metadata assembly
    final_meta = dict(existing_meta) if existing_meta else {}
    if save_quant_metadata:
        import json
        quant_info = {
            "format": "int4_convrot_w4a4",
            "comfy_format": "convrot_w4a4",
            "simple": simple,
            "convrot": convrot,
            "convrot_group_size": convrot_group_size,
            "dynamic_convrot": dynamic_convrot,
            "quantized_layers": processed_count,
            "skipped_layers": skipped_count,
        }
        final_meta["_quantization_metadata"] = json.dumps(quant_info)

    loader.close()
    info(f"Saving {len(new_tensors)} tensors to {output_file}...")
    save_file(new_tensors, output_file, metadata=final_meta)
    info(f"Successfully converted model to INT4 ConvRot W4A4: {output_file}")
