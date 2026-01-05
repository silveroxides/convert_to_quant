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
    simple: bool = False,
    num_iter: int = 500,
    avoid_key_names: Optional[list] = None,
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
    
    Args:
        input_file: Path to input safetensors file
        output_file: Path to output safetensors file
        comfy_quant: If True, add .comfy_quant metadata tensors
        simple: If True, skip learned rounding optimization (use raw converter)
        num_iter: Optimization iterations (if not simple)
        avoid_key_names: Layer name patterns to skip quantization
        heur: If True, skip layers with poor quantization characteristics
        verbose: Print progress
        optimizer: Optimization algorithm ("original", "adamw", "radam")
        lr: Initial learning rate
        lr_schedule: LR schedule ("adaptive", "exponential", "plateau")
        top_p: Proportion of SVD components to use
        min_k: Minimum SVD components
        max_k: Maximum SVD components
        full_matrix: Use full SVD instead of lowrank
        lr_gamma: Decay factor for exponential schedule
        lr_patience: Steps before decay for plateau
        lr_factor: LR reduction factor for plateau
        lr_min: Minimum learning rate floor
        lr_cooldown: Cooldown steps after reduction
        lr_threshold: Minimum improvement threshold
        lr_adaptive_mode: Counter reset mode ("simple-reset", "no-reset")
        lr_shape_influence: Shape-aware LR scaling (0.0-1.0)
        lr_threshold_mode: Threshold mode ("rel", "abs")
        early_stop_loss: Stop when loss drops below this
        early_stop_lr: Stop when LR drops below this
        early_stop_stall: Stop after this many steps without improvement
    """
    if avoid_key_names is None:
        avoid_key_names = AVOID_KEY_NAMES
    
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
            if k.endswith(".weight") and not any(avoid in k for avoid in avoid_key_names)
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
            
            # Skip avoided layers
            if any(avoid in key for avoid in avoid_key_names):
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
