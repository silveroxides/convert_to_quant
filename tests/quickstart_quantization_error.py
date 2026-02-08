#!/usr/bin/env python3
"""
Quick-Start Guide for Quantization Error Measurement

This script demonstrates the complete workflow for measuring quantization error
using your models. Follow the steps below to get started.

STEP 1: Prepare Your Models
===========================
You need two models in safetensors format:
  - model_bf16.safetensors  (Original BF16/FP32 model)
  - model_fp8.safetensors   (FP8 quantized version)

To create a quantized model from your BF16 model:
  python convert_to_quant.py -i model_bf16.safetensors --comfy_quant -o model_fp8.safetensors

STEP 2: Run Quantization Error Measurement
===========================================
Basic measurement:
  python tests/measure_quantization_error.py \\
      --original model_bf16.safetensors \\
      --quantized model_fp8.safetensors

With report output:
  python tests/measure_quantization_error.py \\
      --original model_bf16.safetensors \\
      --quantized model_fp8.safetensors \\
      --output-report error_report.json

STEP 3: Interpret Results
==========================
Check the console output and JSON report:
  - SNR > 40 dB: Excellent quality
  - SNR 30-40 dB: Good quality
  - SNR < 30 dB: Quality may be affected

See QUANTIZATION_ERROR_MEASUREMENT.md for detailed interpretation.

STEP 4: Try Different Quantization Settings
=============================================
If quality is poor, try different quantization parameters:

  # Try INT8 with larger blocks
  python convert_to_quant.py -i model_bf16.safetensors --int8 --block_size 128 -o model_int8.safetensors
  python tests/measure_quantization_error.py --original model_bf16.safetensors --quantized model_int8.safetensors

  # Try different optimizer (slower but sometimes better)
  python convert_to_quant.py -i model_bf16.safetensors --optimizer adamw -o model_adamw.safetensors
  python tests/measure_quantization_error.py --original model_bf16.safetensors --quantized model_adamw.safetensors

===================================================================

This script can also be run directly to test with synthetic models:
  python tests/example_quantization_error.py
"""

def print_welcome():
    """Print welcome message with quick guide."""
    print(__doc__)


def print_examples():
    """Print practical usage examples."""
    print("\n" + "=" * 70)
    print("QUICK COMMAND REFERENCE")
    print("=" * 70)
    
    examples = [
        (
            "Test with Synthetic Models",
            "python tests/example_quantization_error.py"
        ),
        (
            "Measure FP8 Quantization Error",
            "python tests/measure_quantization_error.py \\\n"
            "    --original model_bf16.safetensors \\\n"
            "    --quantized model_fp8.safetensors"
        ),
        (
            "Save Detailed Report",
            "python tests/measure_quantization_error.py \\\n"
            "    --original model_bf16.safetensors \\\n"
            "    --quantized model_fp8.safetensors \\\n"
            "    --output-report error_report.json"
        ),
        (
            "Compare Only Attention Layers",
            "python tests/measure_quantization_error.py \\\n"
            "    --original model_bf16.safetensors \\\n"
            "    --quantized model_fp8.safetensors \\\n"
            "    --layers-to-compare \"attention\""
        ),
        (
            "Use CPU Instead of GPU",
            "python tests/measure_quantization_error.py \\\n"
            "    --original model_bf16.safetensors \\\n"
            "    --quantized model_fp8.safetensors \\\n"
            "    --device cpu"
        ),
    ]
    
    for title, cmd in examples:
        print(f"\n{title}:")
        print(f"  {cmd}")
    
    print("\n" + "=" * 70)


def print_interpretation_guide():
    """Print guide for interpreting results."""
    print("\n" + "=" * 70)
    print("INTERPRETING RESULTS")
    print("=" * 70)
    
    guide = """
Key Metrics Explained:

1. SNR (Signal-to-Noise Ratio) - in Decibels
   ─────────────────────────────────────────
   • SNR > 40 dB  → EXCELLENT: Almost no perceptible difference
   • SNR 35-40 dB → VERY GOOD: High-quality quantization
   • SNR 30-35 dB → GOOD: Acceptable for most applications
   • SNR 25-30 dB → ACCEPTABLE: May have visible degradation
   • SNR < 25 dB  → POOR: Significant quality loss

2. Mean Absolute Error (MAE)
   ────────────────────────
   • Absolute magnitude of error in weight units
   • Compare to weight magnitudes in your model
   • Smaller is better

3. Relative Error
   ──────────────
   • Error as percentage of typical weight value
   • Direct comparison: 0.01 = 1% error, 0.5 = 50% error
   • More interpretable than MAE

4. Max Relative Error
   ──────────────────
   • Identifies worst-case quantization errors
   • Some outliers are acceptable if isolated
   • Very high max relative error suggests scaling issues

5. Relative Error Percentiles
   ──────────────────────────
   • Shows distribution of errors
   • P50 (median) + P99 tells you spread
   • High spread suggests some layers quantize much worse

Quality Diagnosis:

SCENARIO: High SNR (>40 dB) across all layers
→ Excellent! Quantization is working well.

SCENARIO: Good average SNR (35 dB) but a few layers with SNR <30 dB
→ Normal. Some layers (norms, embeddings) are harder to quantize.
  Consider: Exclude these from quantization, use higher precision.

SCENARIO: Consistent low SNR (<30 dB) across all layers
→ Quantization parameters may be wrong.
  Try: Different quantization format, optimizer, or block size.

SCENARIO: High relative error but acceptable SNR
→ Check if outlier values are expected (e.g., biases, scales).
  May still produce good results if dense error (not sparse).

For detailed interpretation, see:
  tests/QUANTIZATION_ERROR_MEASUREMENT.md
"""
    print(guide)


def print_workflow():
    """Print recommended workflow."""
    print("\n" + "=" * 70)
    print("RECOMMENDED WORKFLOW")
    print("=" * 70)
    
    workflow = """
Step 1: Start with Synthetic Models (Fast)
───────────────────────────────────────────
   python tests/example_quantization_error.py
   • Confirms measurement tool works
   • Takes < 1 minute
   • No model files needed

Step 2: Measure Your Base Model
────────────────────────────────
   python tests/measure_quantization_error.py \\
       --original model_bf16.safetensors \\
       --quantized model_fp8_basic.safetensors \\
       --output-report baseline.json

   ✓ establishes baseline quality
   ✓ identifies problem layers

Step 3: Try Different Quantization Settings
───────────────────────────────────────────
   # Option A: Different optimizer
   python convert_to_quant.py \\
       -i model_bf16.safetensors \\
       --optimizer adamw \\
       -o model_fp8_adamw.safetensors

   python tests/measure_quantization_error.py \\
       --original model_bf16.safetensors \\
       --quantized model_fp8_adamw.safetensors \\
       --output-report adamw.json

   # Option B: Different format
   python convert_to_quant.py \\
       -i model_bf16.safetensors \\
       --int8 --block_size 128 \\
       -o model_int8_128.safetensors

   python tests/measure_quantization_error.py \\
       --original model_bf16.safetensors \\
       --quantized model_int8_128.safetensors \\
       --output-report int8_128.json

Step 4: Compare Results
──────────────────────
   Compare baseline.json, adamw.json, int8_128.json
   Choose option with best SNR / acceptable trade-offs

Step 5: Validate in Production
──────────────────────────────
   • Load quantized model in your inference system
   • Run inference on test data
   • Compare output quality with baseline
   • Measure inference speed and memory usage

   ✓ Quantization error ≠ inference error
   • Some quantization error is expected
   • But inference output may still be high quality
"""
    print(workflow)


def main():
    """Print all quick-start information."""
    import sys
    
    print_welcome()
    print_examples()
    print_interpretation_guide()
    print_workflow()
    
    print("\n" + "=" * 70)
    print("For more information, see:")
    print("  • tests/QUANTIZATION_ERROR_MEASUREMENT.md (comprehensive guide)")
    print("  • tests/measure_quantization_error.py --help (tool options)")
    print("  • tests/example_quantization_error.py (synthetic example)")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
