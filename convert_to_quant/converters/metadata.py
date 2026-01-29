import torch
import json
from typing import Dict, Any

def pack_metadata(settings: Dict[str, Any]) -> torch.Tensor:
    """
    Packs quantization settings into a JSON-encoded uint8 tensor.
    
    Args:
        settings: Dictionary containing quantization parameters.
        
    Returns:
        A 1D torch.uint8 tensor containing the JSON bytes.
    """
    # Standardized keys that are relevant for model loading/inference
    allowed_keys = {
        "weights_dtype",
        "quantized_matmul_dtype",
        "group_size",
        "use_svd",
        "svd_rank",
        "use_quantized_matmul",
        "use_stochastic_rounding",
        "dequantize_fp32",
        "use_tensorwise_fp8_matmul",
        "use_contiguous_mm",
        "use_svd_quant", # Sometimes used as synonym or related?
        # Add keys for SVD shapes if dynamic? No, tensor shapes handle that.
    }
    
    # Filter settings to keep only relevant metadata
    # We allow keys that start with "custom_" for future extensibility if needed,
    # but primarily restrict to the known set to keep metadata clean.
    filtered_settings = {
        k: v for k, v in settings.items() 
        if k in allowed_keys
    }
    
    # Sort keys for deterministic output
    json_str = json.dumps(filtered_settings, sort_keys=True)
    json_bytes = json_str.encode('utf-8')
    
    # Convert to standard python list of integers first to avoid numpy dependency if possible
    # (though torch probably handles bytes directly)
    # torch.tensor(list(bytes)) is safe.
    return torch.tensor(list(json_bytes), dtype=torch.uint8)

def unpack_metadata(metadata_tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Unpacks a metadata tensor back into a dictionary.
    
    Args:
        metadata_tensor: A 1D torch.uint8 tensor.
        
    Returns:
        The settings dictionary.
    """
    if metadata_tensor.dtype != torch.uint8:
        raise ValueError(f"Metadata tensor must be uint8, got {metadata_tensor.dtype}")
    
    # Move to CPU and convert to list/bytes
    byte_data = bytes(metadata_tensor.cpu().tolist())
    json_str = byte_data.decode('utf-8')
    return json.loads(json_str)
