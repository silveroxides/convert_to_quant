import torch
import re
from typing import Dict, Any, Union, Optional
from tqdm import tqdm
from .sdnq_math import sdnq_quantize_layer_weight
from .metadata import pack_metadata

def is_matched(name: str, patterns: list) -> bool:
    """
    Check if the name matches any of the patterns (exact string or regex).
    """
    if not patterns:
        return False
    for pattern in patterns:
        if pattern == name:
            return True
        if re.search(pattern, name):
            return True
    return False

def get_layer_type(weight: torch.Tensor) -> str:
    """
    Infer layer type based on weight dimensions.
    """
    ndim = weight.ndim
    if ndim == 2:
        return "Linear"
    elif ndim == 4:
        return "Conv2d"
    elif ndim == 3:
        return "Conv1d"
    elif ndim == 5:
        return "Conv3d"
    return "Unknown"

def convert_state_dict(
    state_dict: Dict[str, torch.Tensor],
    config: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    """
    Iterate over the state_dict and apply quantization to eligible layers.
    
    Args:
        state_dict: The original model state dictionary.
        config: Configuration dictionary containing:
            - modules_to_not_convert: List of patterns to skip.
            - modules_dtype_dict: Dict mapping patterns to specific dtypes.
            - weights_dtype: Default weight dtype (e.g., "int8").
            - ... other settings for sdnq_quantize_layer_weight.
            
    Returns:
        A new state dictionary with quantized weights and auxiliary tensors.
    """
    new_state_dict = {}
    
    modules_to_not_convert = config.get("modules_to_not_convert", [])
    modules_to_remove = config.get("modules_to_remove", [])
    modules_dtype_dict = config.get("modules_dtype_dict", {})
    default_weights_dtype = config.get("weights_dtype", "int8")
    verbose = config.get("verbose", False)

    # Handle dict or iterator
    if hasattr(state_dict, "items"):
        items = state_dict.items()
        total = len(state_dict)
    else:
        items = state_dict
        total = None # Unknown length for iterator

    # Use tqdm if verbose, otherwise simple iteration
    iterator = tqdm(items, desc="Quantizing", total=total) if verbose else items

    for name, tensor in iterator:
        # Check removal
        if is_matched(name, modules_to_remove):
            if verbose:
                print(f"Removing {name} due to removal pattern.")
            continue

        # Check if it's a weight tensor
        if not name.endswith(".weight"):
            new_state_dict[name] = tensor
            continue
            
        # Check exclusion
        if is_matched(name, modules_to_not_convert):
            if verbose:
                print(f"Skipping {name} due to exclusion pattern.")
            new_state_dict[name] = tensor
            continue

        # Infer layer type
        layer_class_name = get_layer_type(tensor)
        
        # We only quantize if we identified it as a Linear or Conv layer
        # (sdnq_math handles these, or generic >1 dim, but we want to be safe)
        if layer_class_name == "Unknown" and tensor.ndim <= 1:
             # Skip 1D tensors (biases, layer norms) if they weren't caught by exclusion
             # Although usually only .weight on Linear/Conv is what we want.
             # LN/BN also have .weight but they are 1D.
             new_state_dict[name] = tensor
             continue
        
        # Determine specific settings for this layer
        layer_settings = config.copy()
        
        # Check for dtype override
        layer_dtype = default_weights_dtype
        for pattern, dtype in modules_dtype_dict.items():
            if re.search(pattern, name) or pattern == name:
                layer_dtype = dtype
                # If multiple match, last one wins? Or first? 
                # Usually specific overrides are rare, assume order doesn't matter much or user is careful.
                # Let's break on first match or keep last. dict order is insertion order.
                # Let's keep it simple.
        
        layer_settings["weights_dtype"] = layer_dtype
        
        # Quantize
        try:
            quantized_weight, scale, zero_point, svd_up, svd_down, info = sdnq_quantize_layer_weight(
                tensor,
                layer_class_name=layer_class_name,
                settings=layer_settings
            )
            
            # Store results
            new_state_dict[name] = quantized_weight
            new_state_dict[f"{name}_scale"] = scale
            
            if zero_point is not None:
                new_state_dict[f"{name}_zp"] = zero_point
                
            if svd_up is not None:
                new_state_dict[f"{name}_svd_up"] = svd_up
                
            if svd_down is not None:
                new_state_dict[f"{name}_svd_down"] = svd_down
                
            # Store metadata
            # We use .weight_metadata suffix to associate it with the weight tensor
            # Some conventions use .metadata or embed in header, but separate tensor is safer for safetensors
            new_state_dict[f"{name}_metadata"] = pack_metadata(info)

        except Exception as e:
            print(f"Error quantizing {name}: {e}")
            # Fallback: store original
            new_state_dict[name] = tensor

    return new_state_dict
