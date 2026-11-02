# Quantization Error Measurement Tools

Scripts and tools for measuring quantization error between full-precision and quantized models.

## Quick Start

### 1. Test with Synthetic Models (Fastest - No Model Files Needed)

```bash
python example_quantization_error.py
```

This creates small test models and runs error measurement in ~1 minute.

### 2. Measure Your Own Models

```bash
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report error_report.json
```

### 3. See Quick-Start Guide

```bash
python quickstart_quantization_error.py
```

Prints interactive guide with examples and interpretation help.

## Available Files

| File | Purpose |
|------|---------|
| **measure_quantization_error.py** | Main tool - measures quantization error between two models |
| **example_quantization_error.py** | Synthetic example - demonstrates workflow with test data |
| **quickstart_quantization_error.py** | Interactive guide - prints quick-start instructions |
| **QUANTIZATION_ERROR_MEASUREMENT.md** | Complete documentation - detailed guide and reference |
| **QUANTIZATION_ERROR_TOOLS_SUMMARY.md** | Summary - overview of all tools and metrics |

## What Gets Measured?

For each layer and in aggregate:

- **Mean Absolute Error (MAE)** - Average error magnitude
- **Relative Error** - Error as percentage of weights
- **Signal-to-Noise Ratio (SNR)** - Quality in decibels (dB)
- **Max/Min Error** - Worst-case and best-case errors
- **Error Distribution** - Percentiles showing spread

## Quality Interpretation

| SNR | Quality | Recommendation |
|-----|---------|-----------------|
| > 40 dB | Excellent | Use for any application |
| 35-40 dB | Very Good | Production-ready |
| 30-35 dB | Good | Acceptable for most uses |
| 25-30 dB | Acceptable | Non-critical use only |
| < 25 dB | Poor | Needs optimization |

## Common Workflows

### Basic Measurement
```bash
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors
```

### With Report
```bash
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report report.json --top-layers 20
```

### Filter Specific Layers
```bash
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --layers-to-compare "attention,mlp"
```

### Use CPU (for low-memory systems)
```bash
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --device cpu
```

## Error Metrics Explained

### SNR (Signal-to-Noise Ratio)
- **Formula**: 20 × log₁₀(original_norm / error_norm)
- **Unit**: Decibels (dB), higher is better
- **Interpretation**: How much quantization noise vs. signal strength

### MAE (Mean Absolute Error)
- **Formula**: Average |original - quantized|
- **Unit**: Same as weights (usually bfloat16/float32)
- **Interpretation**: Typical error magnitude per weight

### Relative Error
- **Formula**: MAE / mean(|original|)
- **Unit**: Unitless (0-1 range best)
- **Interpretation**: Error as percentage of typical weight

### Max Relative Error
- **Formula**: Largest element-wise error / element value
- **Unit**: Unitless
- **Interpretation**: Worst-case quantization accuracy

## Example Output

```
================================================================================
QUANTIZATION ERROR REPORT
================================================================================

AGGREGATE STATISTICS:
  Total layers compared:       48
  Total elements:              45,000,000

ERROR METRICS:
  Mean Absolute Error (MAE):   1.234567e-04
  Median Absolute Error:       8.901234e-05
  Max Absolute Error:          4.567890e-03

SIGNAL QUALITY:
  Overall SNR (dB):            42.50

RELATIVE ERROR PERCENTILES:
  p50:  1.567890e-03
  p75:  2.345678e-03
  p90:  4.567890e-03
  p95:  6.789012e-03
  p99:  1.234567e-02
```

## Reports

Reports are JSON format for easy post-processing:

```bash
# Get mean absolute error per layer
jq '.layers | map({name: .layer_name, mae: .mean_absolute_error})'

# Find worst layers by SNR
jq '.layers | sort_by(.snr_db) | .[0:5]'

# Get aggregate statistics
jq '.aggregate'
```

## Troubleshooting

### "No layers found"
- Check both models have compatible structure
- Use `--layers-to-compare` to debug

### "Shape mismatch"
- Verify same model quantized both ways
- Check no reshaping in quantization

### Out of Memory
- Use `--device cpu` instead of CUDA
- Quantize and measure with smaller models first

## Workflow Example

```bash
# 1. Create FP8 quantized model
python ../convert_to_quant.py -i model_bf16.safetensors --comfy_quant -o model_fp8.safetensors

# 2. Measure quantization error
python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report fp8_error.json

# 3. Try INT8 for comparison
python ../convert_to_quant.py -i model_bf16.safetensors --int8 --block_size 128 --comfy_quant -o model_int8.safetensors

python measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_int8.safetensors \
    --output-report int8_error.json

# 4. Compare results (check fp8_error.json vs int8_error.json)
```

## Documentation

- **QUANTIZATION_ERROR_MEASUREMENT.md** - Full documentation and API reference
- **QUANTIZATION_ERROR_TOOLS_SUMMARY.md** - Tool overview and summary
- Run `python measure_quantization_error.py --help` for CLI help

## Key Features

✅ **Comprehensive Metrics** - Multiple error measurements for thorough analysis  
✅ **Per-Layer Analysis** - Debug which layers have issues  
✅ **Easy to Use** - Simple command-line interface  
✅ **Fast** - Efficient tensor processing  
✅ **Flexible** - Filter layers, change devices, save reports  
✅ **Well Documented** - Guides and examples included  

## Integration

These tools work with:
- `convert_to_quant.py` - Quantization script
- FP8, INT8, MXFP8, NVFP4 formats
- Learned rounding optimization
- ComfyUI quantization metadata

## Next Steps

1. **Try the example**: `python example_quantization_error.py`
2. **Print quick guide**: `python quickstart_quantization_error.py`  
3. **Measure your models**: `python measure_quantization_error.py --original X --quantized Y`
4. **Read full docs**: See `QUANTIZATION_ERROR_MEASUREMENT.md`
