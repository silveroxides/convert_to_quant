# Quantization Error Measurement Tools - Summary

This document summarizes the scripts created for measuring quantization error between BF16 and FP8 models.

## Files Created

### 1. **`tests/measure_quantization_error.py`** (Main Tool)
The core quantization error measurement script.

**Purpose**: Load two safetensors models (original BF16 and quantized FP8) and measure detailed quantization error metrics.

**Key Features**:
- Per-layer error metrics (MAE, relative error, SNR, etc.)
- Aggregate statistics across all layers
- Automatic scale factor detection
- Memory-efficient computation
- JSON report generation
- Detailed console output

**Main Class**: `QuantizationErrorMeasurer`
- `load_model()`: Load safetensors file
- `compare_models()`: Compare original vs quantized tensors
- `compute_error_metrics()`: Calculate error for single layer
- `compute_aggregate_metrics()`: Summary statistics
- `print_report()`: Human-readable output
- `save_report()`: JSON report generation

**Usage**:
```bash
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report error_report.json
```

---

### 2. **`tests/example_quantization_error.py`** (Synthetic Example)
Demonstrates the measurement workflow with synthetic models.

**Purpose**: Show how to use the measurement tools without needing large model files, perfect for testing and understanding the framework.

**What It Does**:
1. Creates a synthetic 5-layer BF16 model
2. Quantizes it to FP8 with proper scaling
3. Runs error measurement
4. Generates report

**Usage**:
```bash
python tests/example_quantization_error.py
```

**Output**:
- Creates `tests/synthetic_test_models/` directory with:
  - `synthetic_bf16.safetensors` (synthetic BF16 model)
  - `synthetic_fp8.safetensors` (quantized version)
  - `error_report.json` (detailed metrics)

---

### 3. **`tests/QUANTIZATION_ERROR_MEASUREMENT.md`** (Comprehensive Guide)
Complete documentation for using and understanding the measurement tools.

**Contents**:
- Overview of error metrics
- Command-line argument reference
- Metric definitions and interpretations
- Report format documentation
- Interpretation guide (what is "good" quantization?)
- Example workflows
- Advanced usage patterns
- Troubleshooting guide
- Mathematical definitions

**Key Sections**:
- **Error Metrics**: MAE, relative error, SNR, percentiles
- **Interpretation**: Quality thresholds and diagnosis
- **Workflow**: Step-by-step examples
- **Advanced Usage**: Filtering, device selection, post-processing

---

### 4. **`tests/quickstart_quantization_error.py`** (Quick-Start Tutorial)
Interactive guide that prints quick-start instructions, examples, and workflows.

**Purpose**: Get users started quickly without reading long documentation.

**Usage**:
```bash
python tests/quickstart_quantization_error.py
```

**Prints**:
- Welcome message with overview
- Quick command reference (5 common examples)
- Interpretation guide for results
- Recommended workflow (4-step process)

---

## Error Metrics Computed

For each layer, the tools compute:

| Metric | Formula | Meaning |
|--------|---------|---------|
| **MAE** | Average \|original - quantized\| | Absolute error magnitude |
| **Relative Error** | MAE / mean(\|original\|) | Error as % of typical weight |
| **Max Absolute Error** | max(\|original - quantized\|) | Worst single error |
| **Max Relative Error** | max(relative error per element) | Worst relative error |
| **SNR (dB)** | 20 × log₁₀(signal_norm / error_norm) | Signal quality in dB |
| **Error Norm** | L2 norm of error vector | Total error magnitude |

Aggregate metrics include:
- Mean, median, max across all layers
- Relative error percentiles (P50, P75, P90, P95, P99)
- Overall SNR combining all layers

## Typical Workflow

```bash
# 1. Create FP8 quantized model
python convert_to_quant.py \
    -i model_bf16.safetensors \
    --comfy_quant \
    -o model_fp8.safetensors

# 2. Measure quantization error
python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_fp8.safetensors \
    --output-report error_report.json

# 3. Analyze results
cat error_report.json | python -m json.tool | less

# 4. If quality is poor, try different settings
python convert_to_quant.py \
    -i model_bf16.safetensors \
    --int8 --block_size 128 \
    --comfy_quant \
    -o model_int8_128.safetensors

python tests/measure_quantization_error.py \
    --original model_bf16.safetensors \
    --quantized model_int8_128.safetensors \
    --output-report error_int8_128.json

# 5. Compare both options
# (SNR and other metrics are in the JSON files)
```

## Quality Indicators

| SNR Range | Quality | Use Case |
|-----------|---------|----------|
| > 40 dB | Excellent | Any production use |
| 35-40 dB | Very Good | General inference |
| 30-35 dB | Good | Most applications |
| 25-30 dB | Acceptable | Non-critical use |
| < 25 dB | Poor | Not recommended |

## Key Features

✅ **Comprehensive Metrics**
- Multiple error metrics for thorough analysis
- Per-layer statistics for debugging
- Aggregate metrics for quick assessment

✅ **Easy to Use**
- Simple command line interface
- Automatic scale detection from metadata
- Clear, formatted output

✅ **Flexible**
- Filter specific layers for analysis
- Save detailed JSON reports for post-processing
- Support for different computation devices (CUDA/CPU)

✅ **Well Documented**
- Detailed usage guide
- Interpretation guidelines
- Synthetic example for testing

✅ **Production Ready**
- Handles large models efficiently
- Proper error handling
- Memory-efficient tensor processing

## Integration with Convert-to-Quant

These measurements tools work seamlessly with:
- `convert_to_quant.py`: Main quantization script
- `.comfy_quant` metadata: Automatic format detection
- Various quantization formats: FP8, INT8, MXFP8, NVFP4
- Multiple quantization optimizers: Original, AdamW, RAdam

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

RELATIVE ERROR:
  Mean Relative Error:         2.345678e-03
  Median Relative Error:       1.567890e-03
  Max Relative Error:          7.890123e-02

SIGNAL QUALITY:
  Overall SNR (dB):            42.50
  Original Norm:               1.234567e+03
  Quantized Norm:              1.234560e+03
  Error Norm:                  5.678901e+00

RELATIVE ERROR PERCENTILES:
  p50:  1.567890e-03
  p75:  2.345678e-03
  p90:  4.567890e-03
  p95:  6.789012e-03
  p99:  1.234567e-02

TOP 10 WORST LAYERS (by MAE):
─────────────────────────────
1. model.layer.0.weight
   Shape:                (1024, 2048)
   Dtype:                torch.bfloat16 -> torch.float8_e4m3fn
   MAE:                  1.234567e-03
   Rel. Error:           3.456789e-02
   Max Rel. Error:       7.890123e-01
   SNR (dB):             35.50
   Format:               float8_e4m3fn
```

## Next Steps

1. **Try the synthetic example**: `python tests/example_quantization_error.py`
2. **Print quick-start guide**: `python tests/quickstart_quantization_error.py`
3. **Read full documentation**: See `tests/QUANTIZATION_ERROR_MEASUREMENT.md`
4. **Measure your models**: `python tests/measure_quantization_error.py --original X --quantized Y --output-report report.json`

## Questions?

- See `QUANTIZATION_ERROR_MEASUREMENT.md` for detailed documentation
- Run `python tests/measure_quantization_error.py --help` for command options
- Run `python tests/example_quantization_error.py` to test the framework
- See `DEVELOPMENT.md` for codebase architecture
