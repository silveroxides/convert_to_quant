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
from typing import Optional
from tqdm import tqdm

from ..constants import (
    AVOID_KEY_NAMES,
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
    ZIMAGE_LAYER_KEYNAMES,
    ZIMAGE_REFINER_LAYER_KEYNAMES,
    FP4_BLOCK_SIZE,
    NORMALIZE_SCALES_ENABLED,
)
from ..converters.nvfp4_converter import NVFP4Converter
from ..converters.learned_nvfp4 import LearnedNVFP4Converter
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales
from ..utils.comfy_quant import should_skip_layer_for_performance


def convert_to_nvfp4(
    input_file: str,
    output_file: str,
    # Filter flags (matching FP8 convention)
    t5xxl: bool = False,
    mistral: bool = False,
    visual: bool = False,
    flux2: bool = False,
    distillation_large: bool = False,
    distillation_small: bool = False,
    nerf_large: bool = False,
    nerf_small: bool = False,
    radiance: bool = False,
    wan: bool = False,
    qwen: bool = False,
    hunyuan: bool = False,
    zimage: bool = False,
    zimage_refiner: bool = False,
    # Quantization options
    simple: bool = False,
    num_iter: int = 500,
    heur: bool = False,
    verbose: bool = True,
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
) -> None:
    """
    Convert safetensors model to NVFP4 (FP4 E2M1) quantized format.
    
    Uses LearnedNVFP4Converter with SVD optimization by default.
    Pass simple=True for raw quantization without optimization.
    
    Always creates .comfy_quant metadata tensors and _quantization_metadata header.
    """
    # Build exclusion list from filter flags (matching FP8 convention)
    exclude_patterns = list(AVOID_KEY_NAMES)  # Base exclusions
    
    if visual:
        exclude_patterns.extend(VISUAL_AVOID_KEY_NAMES)
    if qwen:
        exclude_patterns.extend(QWEN_AVOID_KEY_NAMES)
    if hunyuan:
        exclude_patterns.extend(HUNYUAN_AVOID_KEY_NAMES)
    if zimage or zimage_refiner:
        exclude_patterns.extend(ZIMAGE_AVOID_KEY_NAMES)
    if flux2:
        exclude_patterns.extend(FLUX2_LAYER_KEYNAMES)
    if distillation_large:
        exclude_patterns.extend(DISTILL_LAYER_KEYNAMES_LARGE)
    if distillation_small:
        exclude_patterns.extend(DISTILL_LAYER_KEYNAMES_SMALL)
    if nerf_large:
        exclude_patterns.extend(NERF_LAYER_KEYNAMES_LARGE)
    if nerf_small:
        exclude_patterns.extend(NERF_LAYER_KEYNAMES_SMALL)
    if radiance:
        exclude_patterns.extend(RADIANCE_LAYER_KEYNAMES)
    if wan:
        exclude_patterns.extend(WAN_LAYER_KEYNAMES)
    if zimage:
        exclude_patterns.extend(ZIMAGE_LAYER_KEYNAMES)
    if zimage_refiner:
        exclude_patterns.extend(ZIMAGE_REFINER_LAYER_KEYNAMES)
    
    # Select converter based on --simple flag
    if simple:
        converter = NVFP4Converter(
            block_size=FP4_BLOCK_SIZE,
            pad_to_16x=True,
        )
        use_learned = False
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
            lr=lr,
        )
        use_learned = True
    
    output_tensors = {}
    quant_metadata = {}
    quantized_count = 0
    
    with safe_open(input_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # Filter to only weight tensors
        weight_keys = [
            k for k in keys 
            if k.endswith(".weight") and not any(pattern in k for pattern in exclude_patterns)
        ]
        
        if verbose:
            mode = "Simple" if simple else "Learned Rounding"
            print(f"NVFP4 Quantization ({mode}): {input_file}")
            print(f"  Total tensors: {len(keys)}")
            print(f"  Weight tensors to quantize: {len(weight_keys)}")
        
        for key in tqdm(keys, desc="Processing", disable=not verbose):
            tensor = f.get_tensor(key)
            
            # Skip non-weight tensors (copy as-is)
            if not key.endswith(".weight"):
                output_tensors[key] = tensor
                continue
            
            # Skip excluded layers (based on filter flags)
            if any(pattern in key for pattern in exclude_patterns):
                output_tensors[key] = tensor
                continue
            
            # Skip non-2D tensors (NVFP4 requires 2D)
            if tensor.dim() != 2:
                output_tensors[key] = tensor
                continue
            
            # Skip if heuristics say layer is poor for quantization
            if heur and should_skip_layer_for_performance(tensor.shape, FP4_BLOCK_SIZE):
                output_tensors[key] = tensor
                continue
            
            # Quantize to NVFP4
            base_key = key.rsplit(".weight", 1)[0]
            
            if verbose:
                print(f"\n  Quantizing: {key} {list(tensor.shape)}")
            
            if use_learned:
                # LearnedNVFP4Converter returns (qdata, block_scales, per_tensor_scale, dequantized)
                qdata, block_scales, per_tensor_scale, _ = converter.convert(tensor)
            else:
                # Transfer to GPU for simple quantization
                device = "cuda" if torch.cuda.is_available() else "cpu"
                tensor_gpu = tensor.to(device=device, dtype=torch.float32)
                qdata, block_scales, per_tensor_scale = converter.quantize(tensor_gpu)
                del tensor_gpu
            
            # Store quantized data and scales (move to CPU for saving)
            output_tensors[key] = qdata.cpu()  # Packed uint8
            output_tensors[f"{base_key}.weight_scale"] = per_tensor_scale.cpu().to(torch.float32)
            output_tensors[f"{base_key}.block_scale"] = block_scales.cpu()  # FP8 in cuBLAS layout
            
            # Always create .comfy_quant metadata tensor (required for NVFP4)
            metadata = {
                "format": "nvfp4",
                "group_size": FP4_BLOCK_SIZE,
                "orig_dtype": str(tensor.dtype),
                "orig_shape": list(tensor.shape),
            }
            output_tensors[f"{base_key}.comfy_quant"] = dict_to_tensor(metadata)
            quant_metadata[base_key] = metadata
            
            quantized_count += 1
            
            # Cleanup
            del tensor
            gc.collect()
    
    # Normalize scales if enabled
    if NORMALIZE_SCALES_ENABLED:
        output_tensors, _ = normalize_tensorwise_scales(output_tensors)
    
    # Save output - always include quantization metadata for NVFP4
    metadata_dict = {}
    if quant_metadata:
        import json
        metadata_dict["_quantization_metadata"] = json.dumps(quant_metadata)
    
    if verbose:
        print(f"\nSaving to: {output_file}")
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(output_tensors, output_file, metadata=metadata_dict if metadata_dict else None)
    
    if verbose:
        print(f"Done! Quantized {quantized_count} layers to NVFP4.")
