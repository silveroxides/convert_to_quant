# Quantization Error Measurement Tools

This directory contains tools for measuring the quantization error between full-precision (BF16/FP32) and quantized (FP8/INT8) models.

## Overview

Quantization introduces error when converting high-precision weights to lower precision formats. These tools help:

1. **Quantify the error** with detailed per-layer and aggregate statistics
2. **Identify problem layers** that have high quantization error
3. **Validate quantization quality** before deployment
4. **Compare different quantization strategies** (formats, parameters, optimizers)

## Scripts

### 1. `measure_quantization_error.py` - Main Error Measurement Tool

Loads two safetensors models and measures the quantization error between them.

#### Usage

**Basic usage:**
```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors
```

**With filtered layers:**
```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --layers-to-compare "attention,mlp"
```

**Save detailed JSON report:**
```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report error_report.json
```

**With all options:**
```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --layers-to-compare "attention,mlp" \
    --output-report report.json \
    --top-layers 20 \
    --device cuda
```

#### Command-line Arguments

- `--original` (required): Path to original BF16/FP32 model
- `--quantized` (required): Path to quantized FP8/INT8 model
- `--layers-to-compare`: Comma-separated layer names to include (optional)
- `--output-report`: Path to save detailed JSON report (optional)
- `--top-layers`: Number of worst layers to display (default: 10)
- `--device`: Computation device `cuda` or `cpu` (default: auto-detect)

### 2. `example_quantization_error.py` - Synthetic Example

Demonstrates the error measurement workflow using synthetic models without requiring large model files.

#### Usage

```bash
python tests/example_quantization_error.py
```

This script:
1. Creates a small synthetic BF16 model with 5 layers
2. Quantizes it to FP8
3. Measures the quantization error
4. Generates a detailed report

Perfect for testing and understanding the measurement framework.

## Error Metrics

The tools compute the following error metrics for each layer and in aggregate:

### Per-Layer Metrics

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **MAE** (Mean Absolute Error) | Average `\|original - quantized\|` | Absolute magnitude of error (in same units as weights) |
| **Relative Error** | `MAE / mean(\|original\|)` | Error as fraction of typical weight magnitude (0-1) |
| **Max Absolute Error** | Maximum `\|original - quantized\|` | Worst-case error magnitude |
| **Max Relative Error** | Maximum relative error per element | Worst-case relative error |
| **SNR (dB)** | `20 * log₁₀(signal_norm / error_norm)` | Signal-to-noise ratio in decibels (higher is better) |

### Aggregate Metrics

- **Mean/Median/Max Absolute Error**: Across all layers
- **Mean/Median/Max Relative Error**: Across all layers
- **Relative Error Percentiles**: P50, P75, P90, P95, P99 (understanding error distribution)
- **Overall SNR**: Combined across all layers
- **Total Elements**: Total number of parameters compared

## Report Format

When using `--output-report`, the JSON output contains:

```json
{
  "aggregate": {
    "total_layers": 48,
    "total_elements": 45000000,
    "mean_absolute_error": 0.000123,
    "median_absolute_error": 0.000089,
    "max_absolute_error": 0.00456,
    "mean_relative_error": 0.00234,
    "median_relative_error": 0.00156,
    "max_relative_error": 0.0789,
    "overall_snr_db": 42.5,
    "relative_error_percentiles": {
      "p50": 0.00156,
      "p75": 0.00234,
      "p90": 0.00456,
      "p95": 0.00678,
      "p99": 0.0123
    }
  },
  "layers": [
    {
      "layer_name": "model.layer.0.weight",
      "original_dtype": "torch.bfloat16",
      "quantized_dtype": "torch.float8_e4m3fn",
      "tensor_shape": [1024, 2048],
      "tensor_elements": 2097152,
      "mean_absolute_error": 0.000123,
      "relative_error": 0.00234,
      "max_absolute_error": 0.00456,
      "max_relative_error": 0.0789,
      "snr_db": 42.5,
      "scale_factor": 0.028,
      "comfy_format": "float8_e4m3fn"
    }
  ]
}
```

## Interpretation Guide

### Good Quantization Quality
- **SNR > 40 dB**: Excellent - minimal perceptual difference
- **Mean Relative Error < 0.01**: Very good - less than 1% error
- **Max Relative Error < 0.1**: Good - isolated problematic values still acceptable

### Acceptable Quantization Quality
- **SNR 30-40 dB**: Good for most applications
- **Mean Relative Error 0.01-0.05**: Acceptable for many use cases
- **Max Relative Error 0.1-0.5**: May affect quality in edge cases

### Poor Quantization Quality
- **SNR < 30 dB**: Likely noticeable quality degradation
- **Mean Relative Error > 0.05**: Substantial error
- **Max Relative Error > 0.5**: High error on some elements

### Identifying Problem Layers

1. **Sort by MAE (Mean Absolute Error)**: Largest absolute errors
2. **Sort by SNR**: Layers with worst signal-to-noise ratio
3. **Look at outliers**: Layers with very high max relative error
4. **Check percentiles**: Many layers with high error across the board?

## Example Workflow

Here's a typical workflow for measuring and analyzing quantization error:

```bash
# 1. Quantize a model to FP8
python convert_to_quant.py \
    -i model_bf16.safetensors \
    --comfy_quant \
    -o model_fp8.safetensors

# 2. Measure quantization error
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report error_analysis.json

# 3. Analyze the report (examine error_analysis.json)
# 4. If quality is poor, try different quantization options:

# Try INT8 instead
python convert_to_quant.py \
    -i model_bf16.safetensors \
    --int8 --block_size 128 \
    --comfy_quant \
    -o model_int8.safetensors

# Measure new error
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_int8.safetensors \
    --output-report error_int8.json

# Compare results to decide which format is better
```

## Advanced Usage

### Filtering Specific Layers

Measure error only for attention and MLP layers:

```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --layers-to-compare "attention,mlp" \
    --output-report attention_mlp_error.json
```

### Using Different Devices

By default, the tool auto-detects whether to use CUDA or CPU. Force a specific device:

```bash
# Force CPU (for debugging or low-memory systems)
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --device cpu

# Force CUDA
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --device cuda
```

### Processing Reports Programmatically

The JSON report can be post-processed with standard tools:

```bash
# Find worst 5 layers by SNR
cat error_analysis.json | jq '.layers | sort_by(.snr_db) | .[0:5]'

# Get mean absolute error per layer
cat error_analysis.json | jq '.layers | map({name: .layer_name, mae: .mean_absolute_error})'

# Calculate statistics
cat error_analysis.json | jq '.aggregate | {mae: .mean_absolute_error, snr: .overall_snr_db}'
```

## Implementation Notes

### How Quantization Error is Computed

For each layer:

1. **Load tensors**: Original (BF16) and quantized (FP8) weights
2. **Extract scales**: From `.weight_scale` tensors or metadata
3. **Dequantize**: `dequant = quant * scale` (FP8 only)
4. **Compute error**: `error = |original - dequantized|`
5. **Calculate metrics**: MAE, relative error, SNR, etc.

### Supported Formats

- **FP8** (`torch.float8_e4m3fn`): Standard 8-bit floating point
- **BF16** (`torch.bfloat16`): Low-precision reference
- **INT8** (`torch.int8`): 8-bit integer quantization

Biases and non-weight tensors are typically skipped (kept in original precision).

### Memory Efficiency

For large models:
- Tensors are loaded on CPU by default
- Computation is moved to GPU only when needed
- Intermediate results are freed promptly
- Suitable for models > 100GB in size

## Troubleshooting

### No layers found
**Symptom**: "No matching layers found between original and quantized models"

**Solution**:
- Verify both models have the same structure
- Check that weight names match between files
- Use `--layers-to-compare` to debug with a subset

### Shape mismatch
**Symptom**: "Shape mismatch for layer_name"

**Solution**:
- Ensure you're comparing the same model
- Verify quantization didn't reshape weights
- Check for transpose operations in quantization script

### Out of memory
**Symptom**: CUDA out of memory error

**Solution**:
```bash
# Use CPU instead
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --device cpu
```

## References

- **Quantization Theory**: See [FORMATS.md](../docs/FORMATS.md)
- **Conversion Examples**: See [quantization.examples.md](../quantization.examples.md)
- **Development Guide**: See [DEVELOPMENT.md](../DEVELOPMENT.md)

## Related Tools

- `convert_to_quant.py`: Main quantization script
- `convert_to_quant/formats/fp8_conversion.py`: FP8 quantization logic
- `convert_to_quant/utils/comfy_quant.py`: Metadata handling
