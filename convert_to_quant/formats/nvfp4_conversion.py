"""
NVFP4 conversion functions for convert_to_quant.

Converts safetensors models to NVFP4 (FP4 E2M1) quantized format with
per-tensor + per-block scaling for Blackwell GPU inference.
"""
import gc
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Any, Optional
from tqdm import tqdm

from ..constants import (
    AVOID_KEY_NAMES,
    FP4_E2M1_MAX,
    FP4_BLOCK_SIZE,
    NORMALIZE_SCALES_ENABLED,
)
from ..converters.nvfp4_converter import NVFP4Converter, quantize_nvfp4
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales
from ..utils.comfy_quant import should_skip_layer_for_performance


def convert_to_nvfp4(
    input_file: str,
    output_file: str,
    comfy_quant: bool = True,
    simple: bool = False,
    num_iter: int = 100,
    avoid_key_names: Optional[list] = None,
    heur: bool = False,
    save_quant_metadata: bool = False,
    verbose: bool = True,
) -> None:
    """
    Convert safetensors model to NVFP4 (FP4 E2M1) quantized format.
    
    Args:
        input_file: Path to input safetensors file
        output_file: Path to output safetensors file
        comfy_quant: If True, add .comfy_quant metadata tensors
        simple: If True, skip learned rounding optimization
        num_iter: Optimization iterations (if not simple)
        avoid_key_names: Layer name patterns to skip quantization
        heur: If True, skip layers with poor quantization characteristics
        save_quant_metadata: If True, save _quantization_metadata header
        verbose: Print progress
    """
    if avoid_key_names is None:
        avoid_key_names = AVOID_KEY_NAMES
    
    converter = NVFP4Converter(
        block_size=FP4_BLOCK_SIZE,
        pad_to_16x=True,
        optimize=not simple,
        num_iter=num_iter,
    )
    
    output_tensors = {}
    quant_metadata = {}  # For header metadata
    
    with safe_open(input_file, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        
        # Filter to only weight tensors
        weight_keys = [k for k in keys if k.endswith(".weight") and not any(avoid in k for avoid in avoid_key_names)]
        
        if verbose:
            print(f"NVFP4 Quantization: {input_file}")
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
            
            qdata, block_scales, per_tensor_scale = converter.quantize(tensor.float())
            
            # Store quantized data and scales
            output_tensors[key] = qdata  # Packed uint8
            output_tensors[f"{base_key}.weight_scale"] = per_tensor_scale.to(torch.float32)
            output_tensors[f"{base_key}.block_scale"] = block_scales  # FP8 in cuBLAS layout
            
            if comfy_quant:
                # Create .comfy_quant metadata tensor
                metadata = {
                    "format": "nvfp4",
                    "group_size": FP4_BLOCK_SIZE,
                    "orig_dtype": str(tensor.dtype),
                    "orig_shape": list(tensor.shape),
                }
                output_tensors[f"{base_key}.comfy_quant"] = dict_to_tensor(metadata)
                quant_metadata[base_key] = metadata
            
            # Cleanup
            del tensor
            gc.collect()
    
    # Normalize scales if enabled
    if NORMALIZE_SCALES_ENABLED:
        output_tensors = normalize_tensorwise_scales(output_tensors)
    
    # Save output
    metadata_dict = {}
    if save_quant_metadata and quant_metadata:
        import json
        metadata_dict["_quantization_metadata"] = json.dumps(quant_metadata)
    
    if verbose:
        print(f"\nSaving to: {output_file}")
    
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    save_file(output_tensors, output_file, metadata=metadata_dict if metadata_dict else None)
    
    if verbose:
        print(f"Done! Quantized {len(weight_keys)} layers to NVFP4.")
