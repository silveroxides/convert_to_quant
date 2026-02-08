"""
Synthetic Quantization Error Measurement Example

This script demonstrates the quantization error measurement workflow by:
1. Creating a small synthetic model in BF16
2. Manually quantizing it to FP8
3. Measuring the quantization error
4. Generating a detailed report

Run this to test the measurement framework without needing large model files.
"""

import torch
import json
import os
from pathlib import Path
from safetensors.torch import save_file
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor
from convert_to_quant.utils.logging import setup_logging, info, verbose
from measure_quantization_error import QuantizationErrorMeasurer


def create_synthetic_bf16_model(output_path: str, num_layers: int = 5):
    """Create a synthetic BF16 model for testing."""
    
    info(f"Creating synthetic BF16 model with {num_layers} layers...")
    
    tensors = {}
    
    # Create synthetic layers with various sizes
    for i in range(num_layers):
        # Vary layer sizes
        m = 1024 * (i + 1)  # Input features
        n = 2048 * (i + 1)  # Output features
        
        # Create random tensor in BF16
        tensor = torch.randn(m, n, dtype=torch.bfloat16)
        
        # Scale by a factor to simulate realistic weight distributions
        tensor = tensor * (1.0 / (i + 1))
        
        layer_name = f"model.layer.{i}.weight"
        tensors[layer_name] = tensor
        
        verbose(f"  Layer {i}: {tensor.shape} ({tensor.dtype})")
    
    # Create bias tensors
    for i in range(num_layers):
        n = 2048 * (i + 1)
        bias = torch.randn(n, dtype=torch.bfloat16) * 0.01
        layer_name = f"model.layer.{i}.bias"
        tensors[layer_name] = bias
    
    # Save model
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_file(tensors, output_path)
    
    info(f"Synthetic BF16 model saved to: {output_path}")
    return tensors


def create_synthetic_fp8_quantized_model(
    bf16_tensors: dict,
    output_path: str,
    per_tensor_scale: bool = True
):
    """Create a quantized FP8 version of the BF16 model."""
    
    info(f"Creating synthetic FP8 quantized model...")
    
    tensors = {}
    quant_metadata_layers = {}
    
    for key, original in bf16_tensors.items():
        if key.endswith(".bias"):
            # Keep biases in BF16
            tensors[key] = original
            continue
        
        # Convert to float32 for quantization computation
        original_f32 = original.to(torch.float32)
        
        # Compute scale (per-tensor or per-channel)
        if per_tensor_scale:
            amax = original_f32.abs().max()
            scale = (amax / 127.0).clamp(min=1e-8)
        else:
            # Per-channel scaling (along last dimension)
            amax = original_f32.abs().max(dim=0, keepdim=True)[0]
            scale = (amax / 127.0).clamp(min=1e-8)
        
        # Quantize to FP8
        quantized_f32 = (original_f32 / scale).clamp(-127, 127)
        quantized = quantized_f32.to(torch.float8_e4m3fn)
        
        # Store quantized tensor and scale
        layer_name = key[:-7]  # Remove ".weight"
        tensors[f"{layer_name}.weight"] = quantized
        tensors[f"{layer_name}.weight_scale"] = scale.squeeze() if scale.ndim > 1 else scale
        
        # Create .comfy_quant metadata tensor
        comfy_quant_config = {
            "format": "float8_e4m3fn",
        }
        tensors[f"{layer_name}.comfy_quant"] = create_comfy_quant_tensor(
            "float8_e4m3fn"
        )
        
        # Add to quantization metadata
        quant_metadata_layers[f"{layer_name}"] = {
            "format": "float8_e4m3fn",
            "dtype": "float8_e4m3fn",
        }
        
        verbose(f"  Quantized {key}: scale={scale.item():.6e}")
    
    # Create quantization metadata for header
    quant_metadata = {
        "format_version": "1.0",
        "layers": quant_metadata_layers,
    }
    
    # Save model with metadata
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    save_file(
        tensors,
        output_path,
        metadata={"_quantization_metadata": json.dumps(quant_metadata)}
    )
    
    info(f"Synthetic FP8 quantized model saved to: {output_path}")
    return tensors


def main():
    """Run synthetic quantization error measurement."""
    
    # Setup logging
    setup_logging("VERBOSE")
    
    # Create temporary directory for test files
    test_dir = Path(__file__).parent / "synthetic_test_models"
    test_dir.mkdir(exist_ok=True)
    
    bf16_path = test_dir / "synthetic_bf16.safetensors"
    fp8_path = test_dir / "synthetic_fp8.safetensors"
    report_path = test_dir / "error_report.json"
    
    try:
        # Step 1: Create synthetic BF16 model
        bf16_tensors = create_synthetic_bf16_model(str(bf16_path), num_layers=5)
        
        # Step 2: Create quantized FP8 version
        fp8_tensors = create_synthetic_fp8_quantized_model(bf16_tensors, str(fp8_path))
        
        # Step 3: Measure quantization error
        measurer = QuantizationErrorMeasurer(device="cpu", low_memory=False)  # Use CPU for demo
        
        # Load models
        original_tensors, original_metadata = measurer.load_model(str(bf16_path))
        quantized_tensors, quantized_metadata = measurer.load_model(str(fp8_path))
        
        # Compare
        measurer.compare_models(
            original_tensors,
            quantized_tensors,
            original_metadata,
            quantized_metadata
        )
        
        # Compute metrics
        aggregate = measurer.compute_aggregate_metrics()
        
        if aggregate:
            # Print report
            measurer.print_report(aggregate, top_n=5)
            
            # Save detailed report
            measurer.save_report(str(report_path), aggregate)
            
            info(f"\n✓ Test completed successfully!")
            info(f"  BF16 model:    {bf16_path}")
            info(f"  FP8 model:     {fp8_path}")
            info(f"  Report:        {report_path}")
            
            return 0
        else:
            return 1
    
    except Exception as e:
        from convert_to_quant.utils.logging import error
        error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
