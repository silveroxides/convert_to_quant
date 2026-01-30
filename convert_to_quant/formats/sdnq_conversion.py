"""
SDNQ (Stochastic Differentiable Neural Quantization) Conversion Workflow.

Handles the high-level process of loading a model, applying SDNQ quantization
to eligible layers with filtering, and saving the results with ComfyUI-compatible
metadata.
"""
import os
import torch
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from safetensors.torch import save_file

from ..constants import MODEL_FILTERS, build_exclusion_patterns
from ..converters.sdnq_converter import SDNQConverter
from ..utils.memory_efficient_loader import UnifiedSafetensorsLoader
from ..utils.tensor_utils import dict_to_tensor, normalize_tensorwise_scales
from ..utils.logging import info, verbose, minimal

def convert_to_sdnq(
    input_path: str,
    output_path: str,
    weights_dtype: str = "int8",
    group_size: int = 0,
    use_svd: bool = False,
    svd_rank: int = 32,
    svd_steps: int = 8,
    use_quantized_matmul: bool = False,
    use_stochastic_rounding: bool = False,
    active_filters: Optional[Dict[str, bool]] = None,
    save_comfy_quant: bool = True,
    **kwargs
):
    """
    Apply SDNQ quantization to a model.
    """
    if active_filters is None:
        active_filters = {}

    # 1. Build exclusion patterns from active filters
    exclude_patterns, highprec_patterns, remove_patterns = build_exclusion_patterns(active_filters)
    
    # 2. Initialize Converter
    converter = SDNQConverter(
        weights_dtype=weights_dtype,
        group_size=group_size,
        use_svd=use_svd,
        svd_rank=svd_rank,
        svd_steps=svd_steps,
        use_quantized_matmul=use_quantized_matmul,
        use_stochastic_rounding=use_stochastic_rounding,
        **kwargs
    )

    new_tensors = {}
    
    # 3. Load and Process Model
    with UnifiedSafetensorsLoader(input_path, low_memory=True) as loader:
        metadata = loader.metadata()
        tensor_keys = loader.keys()
        
        pbar = tqdm(tensor_keys, desc="Quantizing with SDNQ", dynamic_ncols=True)
        
        for key in pbar:
            tensor = loader.get_tensor(key)
            base_name = key.replace(".weight", "")
            
            # Handle removal
            should_remove = any(p in key for p in remove_patterns)
            if should_remove:
                verbose(f"  - Removing layer: {key}")
                loader.mark_processed(key)
                continue

            # Determine if we should quantize
            is_weight = key.endswith(".weight")
            is_excluded = any(p in key for p in exclude_patterns)
            is_highprec = any(p in key for p in highprec_patterns)
            
            if not is_weight or is_excluded or is_highprec:
                if is_excluded:
                    verbose(f"  - Skipping (excluded): {key}")
                elif is_highprec:
                    verbose(f"  - Skipping (high-precision): {key}")
                
                new_tensors[key] = tensor
                loader.mark_processed(key)
                continue

            # Apply SDNQ Quantization
            try:
                qdata, scale, zero_point, svd_up, svd_down, q_info = converter.quantize(tensor)
                
                # Store core tensors
                new_tensors[key] = qdata
                new_tensors[f"{base_name}.weight_scale"] = scale
                
                if zero_point is not None:
                    new_tensors[f"{base_name}.weight_zp"] = zero_point
                
                if svd_up is not None:
                    new_tensors[f"{base_name}.svd_up"] = svd_up
                    new_tensors[f"{base_name}.svd_down"] = svd_down
                
                # Add ComfyUI Metadata
                if save_comfy_quant:
                    comfy_quant_key = f"{base_name}.comfy_quant"
                    # SDNQ needs more metadata than standard formats
                    comfy_dict = {
                        "format": "sdnq",
                        "weights_dtype": weights_dtype,
                        "group_size": q_info["group_size"],
                        "use_svd": use_svd,
                        "original_shape": list(tensor.shape),
                        "orig_dtype": str(tensor.dtype).split('.')[-1]
                    }
                    new_tensors[comfy_quant_key] = dict_to_tensor(comfy_dict)
                
            except Exception as e:
                minimal(f"Error quantizing {key}: {e}")
                new_tensors[key] = tensor
            
            loader.mark_processed(key)
            
    # 4. Finalize and Save
    info(f"Saving quantized model to: {output_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Normalize scales before saving
    new_tensors, _ = normalize_tensorwise_scales(new_tensors)
    
    # Ensure all tensors are contiguous before saving (required by safetensors)
    for key in list(new_tensors.keys()):
        if isinstance(new_tensors[key], torch.Tensor) and not new_tensors[key].is_contiguous():
            new_tensors[key] = new_tensors[key].contiguous()
            
    save_file(new_tensors, output_path, metadata=metadata)
    info("Conversion complete!")
