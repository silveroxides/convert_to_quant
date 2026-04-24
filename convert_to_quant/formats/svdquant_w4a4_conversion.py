# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SVDQuant W4A4 conversion for convert_to_quant.

Converts float safetensors weights to the five-tensor layout that
TensorCoreSVDQuantW4A4Layout (comfy-kitchen) expects at inference time.

Output tensors per layer (base_key = key without ".weight"):

    {base_key}.weight                  (N, K//2) int8   qweight (nibble-packed)
    {base_key}.weight_scale            (K//64, N) bf16  per-group weight scales
    {base_key}.weight_proj_down        (K, R) bf16      SVD down projection
    {base_key}.weight_proj_up          (N, R) bf16      SVD up projection
    {base_key}.weight_smooth_factor    (K,) bf16        per-channel smooth divisor
    {base_key}.comfy_quant             uint8            JSON metadata tensor

Suffix names come directly from TensorCoreSVDQuantW4A4Layout.state_dict_tensors().

act_unsigned flag
-----------------
Post-GELU fc2 layers in some architectures (Qwen-based models with nunchaku
calibration) use unsigned int4 [0,15] quantization. convert_to_quant cannot
determine this from weight data alone. The --act-unsigned-layers regex lets
callers identify these layers. The flag is stored in .comfy_quant metadata
so the inference loader can read it back and pass act_unsigned=True to the
kernel. Layers not matching the regex default to act_unsigned=False (signed).

Smooth mode quality note
------------------------
Without real activation statistics, the "weight_only" smooth heuristic is a
known approximation. Expect lower quality than nunchaku's calibration-derived
smooth for models with strong activation outliers (e.g. Qwen-Image-Edit at
full precision). Use --smooth-mode external with a calibrated smooth file
for production-quality checkpoints.
"""
from __future__ import annotations

import gc
import json
import os
import re
from typing import Dict, Optional

import torch
from safetensors.torch import save_file

from ..constants import AVOID_KEY_NAMES, MODEL_FILTERS, COMPUTE_DTYPE
from ..converters.svdquant_w4a4_converter import SVDQuantW4A4Converter
from ..converters.learned_svdquant_w4a4 import LearnedSVDQuantW4A4Converter
from ..utils.tensor_utils import dict_to_tensor, compute_bias_correction
from ..utils.comfy_quant import should_skip_layer_for_performance
from ..utils.memory_efficient_loader import UnifiedSafetensorsLoader
from ..utils.logging import info, verbose, debug, minimal, warning, error, log_debug

# Format string written to .comfy_quant metadata.
# Must be stable -- it's the key the kitchen loader uses to dispatch.
_FORMAT_STRING = "svdquant_w4a4"

# Group size that matches the hardcoded constant in the kitchen CUDA kernel.
_GROUP_SIZE = 64


@log_debug
def convert_to_svdquant_w4a4(
    input_file: str,
    output_file: str,
    # Layer selection
    filter_flags: Dict[str, bool] = None,
    exclude_layers: Optional[str] = None,
    # SVDQuant options
    rank: int = 32,
    smooth_mode: str = "weight_only",
    smooth_alpha: float = 0.5,
    smooth_file: Optional[str] = None,
    act_unsigned_layers: Optional[str] = None,
    # Simple vs learned
    simple: bool = False,
    # Learned rounding optimizer options (ignored when simple=True)
    num_iter: int = 2000,
    optimizer: str = "prodigy",
    lr: float = 1.0,
    lr_schedule: str = "plateau",
    top_p: float = 0.2,
    min_k: int = 128,
    max_k: int = 1280,
    full_matrix: bool = False,
    lr_gamma: float = 0.99,
    lr_patience: int = 1,
    lr_factor: float = 0.95,
    lr_min: float = 1e-8,
    lr_cooldown: int = 0,
    lr_threshold: float = 0.0,
    lr_adaptive_mode: str = "simple-reset",
    lr_shape_influence: float = 1.0,
    lr_threshold_mode: str = "rel",
    early_stop_loss: float = 5e-9,
    early_stop_lr: float = 1.01e-8,
    early_stop_stall: int = 2000,
    use_speed: bool = False,
    # Bias correction
    calib_samples: int = 3072,
    seed: int = 42,
    # Misc
    heur: bool = False,
    low_memory: bool = False,
) -> None:
    """Convert a float safetensors model to SVDQuant W4A4 format.

    Args:
        input_file: Path to input float .safetensors file.
        output_file: Path to write the converted .safetensors file.
        filter_flags: Model-specific layer exclusion dict from CLI.
        exclude_layers: Regex pattern for additional layers to skip.
        rank: SVD rank for low-rank LoRA correction (default 32).
            0 disables the correction.
        smooth_mode: "weight_only" (default), "ones", or "external".
        smooth_alpha: Power for "weight_only" mode (default 0.5).
        smooth_file: Path to pre-computed smooth .safetensors (external mode).
        act_unsigned_layers: Regex identifying post-GELU fc2 layers for
            unsigned int4 [0,15] quantization.
        simple: If True, skip learned rounding (round-to-nearest only).
        num_iter / optimizer / lr / ...: Learned rounding parameters.
            Ignored when simple=True.
        calib_samples: Random calibration samples for bias correction.
        seed: Random seed.
        heur: Skip layers unlikely to benefit from int4.
        low_memory: Load tensors one at a time to reduce peak RAM.
    """
    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    info("Target format: SVDQuant W4A4 (int4 weight + int4 activation)")
    info(f"SVD rank    : {rank}")
    info(f"Group size  : {_GROUP_SIZE} (fixed)")
    info(f"Smooth mode : {smooth_mode}" + (f" (alpha={smooth_alpha})" if smooth_mode == "weight_only" else ""))
    info(f"Mode        : {'simple (no learned rounding)' if simple else f'learned ({optimizer})'}")
    if act_unsigned_layers:
        info(f"Unsigned act: {act_unsigned_layers}")
    info("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Build exclusion list -----------------------------------------------
    exclude_patterns = list(AVOID_KEY_NAMES)
    active_filters = filter_flags or {}
    for filter_name, is_active in active_filters.items():
        if not is_active:
            continue
        cfg = MODEL_FILTERS.get(filter_name, {})
        exclude_patterns.extend(cfg.get("exclude", []))
        exclude_patterns.extend(cfg.get("highprec", []))

    exclude_regex = None
    if exclude_layers:
        try:
            exclude_regex = re.compile(exclude_layers)
            info(f"Layer exclusion regex: '{exclude_layers}'")
        except re.error as e:
            error(f"Invalid regex pattern '{exclude_layers}': {e}")
            return

    unsigned_regex = None
    if act_unsigned_layers:
        try:
            unsigned_regex = re.compile(act_unsigned_layers)
        except re.error as e:
            error(f"Invalid --act-unsigned-layers regex '{act_unsigned_layers}': {e}")
            return

    # ---- Load external smooth factors if requested --------------------------
    external_smooth: Dict[str, torch.Tensor] = {}
    if smooth_mode == "external":
        if not smooth_file:
            error("--smooth-mode external requires --smooth-file.")
            return
        if not os.path.exists(smooth_file):
            error(f"Smooth file not found: {smooth_file}")
            return
        from safetensors import safe_open as _safe_open
        info(f"Loading smooth factors from: {smooth_file}")
        with _safe_open(smooth_file, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(".smooth_factor"):
                    base = k[: -len(".smooth_factor")]
                    external_smooth[base] = f.get_tensor(k)
        info(f"  Loaded {len(external_smooth)} smooth tensors.")

    # ---- Open input file ----------------------------------------------------
    try:
        loader = UnifiedSafetensorsLoader(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    all_keys = loader.keys()
    original_metadata = loader.metadata()

    weight_keys = sorted([
        k for k in all_keys
        if k.endswith(".weight") and loader.get_ndim(k) == 2
    ])
    total_weights = len(weight_keys)

    # ---- Build calibration data cache for bias correction -------------------
    calibration_cache: Dict[int, torch.Tensor] = {}
    seed_gen = torch.Generator(device=device)
    seed_gen.manual_seed(seed)
    minimal("Scanning model and generating calibration data...")
    for key in weight_keys:
        shape = loader.get_shape(key)
        if len(shape) == 2:
            in_features = shape[1]
            if in_features not in calibration_cache:
                calibration_cache[in_features] = torch.randn(
                    calib_samples, in_features,
                    dtype=COMPUTE_DTYPE, generator=seed_gen, device=device,
                )

    # ---- Converter ----------------------------------------------------------
    if simple:
        converter = SVDQuantW4A4Converter(
            rank=rank,
            smooth_mode=smooth_mode,
            smooth_alpha=smooth_alpha,
            device=device,
        )
    else:
        converter = LearnedSVDQuantW4A4Converter(
            rank=rank,
            smooth_mode=smooth_mode,
            smooth_alpha=smooth_alpha,
            # Optimizer
            optimizer=optimizer,
            num_iter=num_iter,
            lr=lr,
            lr_schedule=lr_schedule,
            top_p=top_p,
            min_k=min_k,
            max_k=max_k,
            full_matrix=full_matrix,
            # LR schedule
            lr_gamma=lr_gamma,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_min=lr_min,
            lr_cooldown=lr_cooldown,
            lr_threshold=lr_threshold,
            lr_adaptive_mode=lr_adaptive_mode,
            lr_shape_influence=lr_shape_influence,
            lr_threshold_mode=lr_threshold_mode,
            # Early stopping
            early_stop_loss=early_stop_loss,
            early_stop_lr=early_stop_lr,
            early_stop_stall=early_stop_stall,
            use_speed=use_speed,
            device=device,
        )

    output_tensors: Dict[str, torch.Tensor] = {}
    quant_metadata: Dict[str, dict] = {}
    quantized_count = 0
    skipped_count = 0

    for i, key in enumerate(weight_keys):
        tensor = loader.get_tensor(key)
        base_key = key.rsplit(".weight", 1)[0]
        exclusion_reason = ""

        if any(p in key for p in exclude_patterns):
            exclusion_reason = "exclusion pattern"
        elif exclude_regex and exclude_regex.search(key):
            exclusion_reason = "regex exclusion"
        elif tensor.dim() != 2:
            exclusion_reason = "non-2D tensor"
        elif tensor.shape[1] % _GROUP_SIZE != 0:
            exclusion_reason = f"K={tensor.shape[1]} not divisible by group_size={_GROUP_SIZE}"

        if not exclusion_reason and heur:
            should_skip, skip_reason = should_skip_layer_for_performance(tensor, _GROUP_SIZE)
            if should_skip:
                exclusion_reason = skip_reason

        if exclusion_reason:
            info(f"({i+1}/{total_weights}) Skipping: {key} ({exclusion_reason})")
            output_tensors[key] = tensor
            skipped_count += 1
            continue

        info(f"({i+1}/{total_weights}) Processing: {key}")

        N, K = tensor.shape
        is_unsigned = bool(unsigned_regex and unsigned_regex.search(key))

        # Extract block depth from key for learned converter heuristics.
        depth = -1
        depth_match = re.search(r"\.(\d+)\.", key)
        if depth_match:
            depth = int(depth_match.group(1))

        # External smooth: look up by base_key
        ext_smooth = external_smooth.get(base_key) if smooth_mode == "external" else None

        try:
            qweight, wscales, proj_down, proj_up, smooth_factor, dequant_w = (
                converter.convert(tensor, key=key, depth=depth, smooth=ext_smooth)
            )
        except Exception as e:
            warning(f"  Conversion failed for {key}: {e}. Keeping original weight.")
            output_tensors[key] = tensor
            skipped_count += 1
            continue

        actual_rank = proj_down.shape[1]

        # ---- Store per-layer tensors ----------------------------------------
        # Suffixes from TensorCoreSVDQuantW4A4Layout.state_dict_tensors():
        #   ""              -> qdata   (the .weight key itself)
        #   "_scale"        -> wscales
        #   "_proj_down"    -> proj_down
        #   "_proj_up"      -> proj_up
        #   "_smooth_factor"-> smooth_factor
        output_tensors[key] = qweight                                 # .weight
        output_tensors[f"{base_key}.weight_scale"]        = wscales
        output_tensors[f"{base_key}.weight_proj_down"]    = proj_down
        output_tensors[f"{base_key}.weight_proj_up"]      = proj_up
        output_tensors[f"{base_key}.weight_smooth_factor"] = smooth_factor

        # ---- Bias correction ------------------------------------------------
        bias_key = f"{base_key}.bias"
        if bias_key in all_keys:
            original_bias = loader.get_tensor(bias_key)
            in_features = tensor.shape[1]
            if in_features in calibration_cache:
                verbose(f"  Correcting bias: {bias_key}")
                corrected_bias, ok = compute_bias_correction(
                    original_weight=tensor,
                    dequantized_weight=dequant_w,
                    original_bias=original_bias,
                    calibration_data=calibration_cache[in_features],
                    device=device,
                )
                if ok:
                    output_tensors[bias_key] = corrected_bias
                else:
                    output_tensors[bias_key] = original_bias
            else:
                output_tensors[bias_key] = original_bias

        # ---- .comfy_quant metadata ------------------------------------------
        layer_meta = {
            "format":        _FORMAT_STRING,
            "group_size":    _GROUP_SIZE,
            "rank":          actual_rank,
            "orig_dtype":    str(tensor.dtype),
            "orig_shape":    list(tensor.shape),
            "act_unsigned":  is_unsigned,
            "smooth_mode":   smooth_mode,
            "learned_round": not simple,
        }
        output_tensors[f"{base_key}.comfy_quant"] = dict_to_tensor(layer_meta)
        quant_metadata[base_key] = layer_meta

        # ---- Logging --------------------------------------------------------
        info(f"    weight  : {list(qweight.shape)} int8")
        info(f"    wscales : {list(wscales.shape)} {wscales.dtype}")
        info(f"    proj_down: {list(proj_down.shape)} | proj_up: {list(proj_up.shape)}")
        if is_unsigned:
            info("    act_unsigned: True")
        info("-" * 60)

        quantized_count += 1

        del tensor, dequant_w
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- Copy remaining non-weight tensors ----------------------------------
    for key in all_keys:
        if key not in output_tensors:
            output_tensors[key] = loader.get_tensor(key)

    loader.close()

    # ---- Save ---------------------------------------------------------------
    output_metadata = dict(original_metadata)
    if quant_metadata:
        full_metadata = {
            "format_version": "1.0",
            "layers": quant_metadata,
        }
        output_metadata["_quantization_metadata"] = json.dumps(full_metadata)

    info(f"\nSaving {len(output_tensors)} tensors to {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(output_tensors, output_file, metadata=output_metadata or None)

    info("-" * 60)
    info("Summary:")
    info(f"  Original tensor count : {len(all_keys)}")
    info(f"  Weights quantized     : {quantized_count}")
    info(f"  Weights skipped       : {skipped_count}")
    info(f"  Final tensor count    : {len(output_tensors)}")
    info("-" * 60)
    info("Conversion complete!")
