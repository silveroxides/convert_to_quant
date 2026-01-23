"""
NVFP4 conversion functions for convert_to_quant.

Converts safetensors models to NVFP4 (FP4 E2M1) quantized format with
per-tensor + per-block scaling for Blackwell GPU inference.

Uses LearnedNVFP4Converter (SVD optimization) by default.
Use --simple to switch to raw NVFP4Converter.
"""
import gc
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Optional

from ..constants import (
    AVOID_KEY_NAMES,
    MODEL_FILTERS,
    FP4_BLOCK_SIZE,
    NORMALIZE_SCALES_ENABLED,
    COMPUTE_DTYPE,
)
from ..converters.nvfp4_converter import NVFP4Converter
from ..converters.learned_nvfp4 import LearnedNVFP4Converter
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales
from ..utils.comfy_quant import should_skip_layer_for_performance
from ..utils.memory_efficient_loader import UnifiedSafetensorsLoader
from ..utils.logging import info, verbose, debug, minimal, warning, error, log_debug

@log_debug
def convert_to_nvfp4(
    input_file: str,
    output_file: str,
    # Filter flags (validated dict from CLI)
    filter_flags: Dict[str, bool] = None,
    exclude_layers: Optional[str] = None,
    # Quantization options
    simple: bool = False,
    num_iter: int = 500,
    heur: bool = False,
    verbose_output: bool = True,  # kept for API compat, actual logging uses logging.py
    # Calibration options (for bias correction)
    calib_samples: int = 3072,
    seed: int = 42,
    # Optimizer/LR options (passed to LearnedNVFP4Converter)
    optimizer: str = "original",
    lr: float = 8.077300000003e-3,
    lr_schedule: str = "adaptive",
    top_p: float = 0.01,
    min_k: int = 1,
    max_k: int = 16,
    full_matrix: bool = False,
    # LR schedule tuning
    lr_gamma: float = 0.99,
    lr_patience: int = 50,
    lr_factor: float = 0.5,
    lr_min: float = 1e-8,
    lr_cooldown: int = 0,
    lr_threshold: float = 0.0,
    lr_adaptive_mode: str = "simple-reset",
    lr_shape_influence: float = 1.0,
    lr_threshold_mode: str = "rel",
    # Early stopping
    early_stop_loss: float = 1e-8,
    early_stop_lr: float = 1e-10,
    early_stop_stall: int = 1000,
    # Scale optimization
    scale_refinement_rounds: int = 1,
    scale_optimization: str = "fixed",
    # Input scales (optional, from calibration or another NVFP4 model)
    input_scales: Optional[dict] = None,
    # Memory mode
    low_memory: bool = False,
) -> None:
    """
    Convert safetensors model to NVFP4 (FP4 E2M1) quantized format.

    Uses LearnedNVFP4Converter with SVD optimization by default.
    Pass simple=True for raw quantization without optimization.

    Always creates .comfy_quant metadata tensors and _quantization_metadata header.
    """
    info(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    info("-" * 60)
    info("Target format: NVFP4 (FP4 E2M1 block quantization)")
    info(f"Block size: {FP4_BLOCK_SIZE}")
    info("-" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    # Build exclusion list from filter flags using MODEL_FILTERS registry
    exclude_patterns = list(AVOID_KEY_NAMES)  # Base exclusions

    # Use filter_flags dict passed from CLI (or empty if not provided)
    active_filters = filter_flags or {}

    # Add patterns from active filters
    for filter_name, is_active in active_filters.items():
        if not is_active:
            continue
        cfg = MODEL_FILTERS[filter_name]
        exclude_patterns.extend(cfg.get("exclude", []))
        exclude_patterns.extend(cfg.get("highprec", []))

    # Compile --exclude-layers regex pattern
    exclude_regex_pattern = None
    if exclude_layers:
        import re
        try:
            exclude_regex_pattern = re.compile(exclude_layers)
            info(f"Layer exclusion enabled: pattern '{exclude_layers}'")
        except re.error as e:
            error(f"ERROR: Invalid regex pattern '{exclude_layers}': {e}")
            return

    # Select converter based on --simple flag
    if simple:
        converter = NVFP4Converter(
            block_size=FP4_BLOCK_SIZE,
            pad_to_16x=True,
        )
        use_learned = False
        info("NVFP4 Simple mode (no learned rounding optimization)")
    else:
        converter = LearnedNVFP4Converter(
            optimizer=optimizer,
            num_iter=num_iter,
            top_p=top_p,
            min_k=min_k,
            max_k=max_k,
            block_size=FP4_BLOCK_SIZE,
            pad_to_16x=True,
            full_matrix=full_matrix,
            no_learned_rounding=False,
            lr_schedule=lr_schedule,
            lr_gamma=lr_gamma,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            lr_min=lr_min,
            lr_cooldown=lr_cooldown,
            lr_threshold=lr_threshold,
            lr_adaptive_mode=lr_adaptive_mode,
            lr_shape_influence=lr_shape_influence,
            lr_threshold_mode=lr_threshold_mode,
            early_stop_loss=early_stop_loss,
            early_stop_lr=early_stop_lr,
            early_stop_stall=early_stop_stall,
            scale_refinement_rounds=scale_refinement_rounds,
            scale_optimization=scale_optimization,
            lr=lr,
        )
        use_learned = True

    output_tensors: Dict[str, torch.Tensor] = {}
    quant_metadata = {}
    quantized_count = 0
    skipped_count = 0

    # Load tensors using unified loader (handles both standard and low-memory modes)
    try:
        loader = UnifiedSafetensorsLoader(input_file, low_memory=low_memory)
    except Exception as e:
        error(f"FATAL: Error loading '{input_file}': {e}")
        return

    all_keys = loader.keys()

    # Read original file metadata to preserve during conversion
    original_metadata = loader.metadata()

    # Filter to only weight tensors for quantization
    weight_keys = sorted([
        k for k in all_keys
        if k.endswith(".weight") and loader.get_ndim(k) == 2
    ])
    total_weights = len(weight_keys)

    # Generate calibration data for bias correction (always, even in simple mode)
    calibration_data_cache = {}
    minimal("Scanning model and generating simulated calibration data...")
    for key in weight_keys:
        shape = loader.get_shape(key)
        if len(shape) == 2:
            in_features = shape[1]
            if in_features not in calibration_data_cache:
                verbose(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(
                    calib_samples,
                    in_features,
                    dtype=COMPUTE_DTYPE,
                    generator=seed_generator,
                    device=seed_device,
                )
    info("Simulated calibration data generated.\n")

    info(f"Found {total_weights} weight tensors to potentially process.")
    info("-" * 60)

    for i, key in enumerate(weight_keys):
        tensor = loader.get_tensor(key)
        base_key = key.rsplit(".weight", 1)[0]
        exclusion_reason = ""

        # Check exclusion patterns (substring match)
        if any(pattern in key for pattern in exclude_patterns):
            exclusion_reason = "Exclusion pattern match"

        # Check --exclude-layers regex pattern
        if not exclusion_reason and exclude_regex_pattern and exclude_regex_pattern.search(key):
            exclusion_reason = "regex exclusion (--exclude-layers)"

        # Skip non-2D tensors (NVFP4 requires 2D)
        if tensor.dim() != 2:
            info(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: non-2D tensor)")
            output_tensors[key] = tensor
            skipped_count += 1
            continue

        # Skip if exclusion pattern matched
        if exclusion_reason:
            info(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {exclusion_reason})")
            output_tensors[key] = tensor
            skipped_count += 1
            continue

        # Skip if heuristics say layer is poor for quantization
        if heur:
            should_skip, skip_reason = should_skip_layer_for_performance(tensor, FP4_BLOCK_SIZE)
            if should_skip:
                info(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                output_tensors[key] = tensor
                skipped_count += 1
                continue

        info(f"({i+1}/{total_weights}) Processing tensor: {key}")

        # Quantize to NVFP4
        if use_learned:
            # LearnedNVFP4Converter returns (qdata, block_scales, per_tensor_scale, dequantized)
            qdata, block_scales, per_tensor_scale, dequant_w = converter.convert(tensor)
            # Crop dequant_w back to original shape if it was padded
            if dequant_w.shape != tensor.shape:
                dequant_w = dequant_w[:tensor.shape[0], :tensor.shape[1]]
        else:
            # Transfer to GPU for simple quantization
            tensor_gpu = tensor.to(device=device, dtype=torch.float32)
            qdata, block_scales, per_tensor_scale = converter.quantize(tensor_gpu)
            # For simple mode, we need to dequantize for bias correction
            dequant_w = converter.dequantize(qdata, per_tensor_scale, block_scales, output_dtype=torch.float32)
            # Crop dequant_w back to original shape if it was padded
            if dequant_w.shape != tensor.shape:
                dequant_w = dequant_w[:tensor.shape[0], :tensor.shape[1]]
            del tensor_gpu

        # Store quantized data and scales (move to CPU for saving)
        output_tensors[key] = qdata.cpu()  # Packed uint8

        # per_tensor_scale -> weight_scale_2 (scalar, matching NVIDIA format)
        output_tensors[f"{base_key}.weight_scale_2"] = per_tensor_scale.cpu().to(torch.float32)

        # block_scales -> weight_scale (float8_e4m3fn, matching NVIDIA format)
        output_tensors[f"{base_key}.weight_scale"] = block_scales.cpu()

        # Optional: input_scale from calibration (scalar float32)
        if input_scales and base_key in input_scales:
            output_tensors[f"{base_key}.input_scale"] = torch.tensor(
                input_scales[base_key], dtype=torch.float32
            )

        # Bias correction (matching FP8 logic)
        bias_key = f"{base_key}.bias"
        if bias_key in all_keys:
            # Apply bias correction even in simple mode for better accuracy
            verbose(f"  - Adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                    original_bias = loader.get_tensor(bias_key)
                    in_features = tensor.shape[1]
                    if in_features not in calibration_data_cache:
                        warning("  - WARNING: No calibration data for bias correction.")
                        output_tensors[bias_key] = original_bias
                    else:
                        X_calib_dev = calibration_data_cache[in_features].to(device=device)
                        W_orig_dev = tensor.to(device=device, dtype=COMPUTE_DTYPE)
                        W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                        b_orig_dev = original_bias.to(device=device, dtype=COMPUTE_DTYPE)
                        weight_error = W_orig_dev - W_dequant_dev
                        output_error = X_calib_dev @ weight_error.T
                        bias_correction = output_error.mean(dim=0)
                        b_new = b_orig_dev - bias_correction
                        output_tensors[bias_key] = b_new.to(device="cpu", dtype=original_bias.dtype)
                        verbose(
                            f"    - Original bias mean : {original_bias.mean().item():.6f}\n"
                            f"    - Corrected bias mean: {output_tensors[bias_key].mean().item():.6f}"
                        )
                        del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                        if device == "cuda":
                            torch.cuda.empty_cache()

        # Always create .comfy_quant metadata tensor (required for NVFP4)
        metadata = {
            "format": "nvfp4",
            "group_size": FP4_BLOCK_SIZE,
            "orig_dtype": str(tensor.dtype),
            "orig_shape": list(tensor.shape),
        }
        output_tensors[f"{base_key}.comfy_quant"] = dict_to_tensor(metadata)
        quant_metadata[base_key] = metadata

        # Final shape outputs
        info(f"    - Final Weight shape      : {list(qdata.shape)}")
        info(f"    - Final Block Scale shape : {list(block_scales.shape)}")
        info("-" * 60)

        quantized_count += 1

        # Cleanup
        del tensor, dequant_w
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Copy non-weight tensors (bias handled above, copy others)
    for key in all_keys:
        if key not in output_tensors:
            output_tensors[key] = loader.get_tensor(key)

    # Close loader
    loader.close()

    # Normalize scales if enabled
    if NORMALIZE_SCALES_ENABLED:
        output_tensors, normalized_count = normalize_tensorwise_scales(output_tensors)
        if normalized_count > 0:
            print(f"Normalized {normalized_count} scale tensors to scalars")

    # Save output - preserve original metadata and include quantization metadata
    output_metadata = dict(original_metadata)
    if quant_metadata:
        import json
        # Wrap in proper structure with format_version and layers (matching FP8)
        full_metadata = {"format_version": "1.0", "layers": quant_metadata}
        output_metadata["_quantization_metadata"] = json.dumps(full_metadata)

    info(f"\nSaving {len(output_tensors)} tensors to {output_file}")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(output_tensors, output_file, metadata=output_metadata if output_metadata else None)

    info("-" * 60)
    info("Summary:")
    info(f"  - Original tensor count : {len(all_keys)}")
    info(f"  - Weights processed     : {quantized_count}")
    info(f"  - Weights skipped       : {skipped_count}")
    info(f"  - Final tensor count    : {len(output_tensors)}")
    info("-" * 60)
    info("Conversion complete!")
