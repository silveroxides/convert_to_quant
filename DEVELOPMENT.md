
## 2026-01-24: Formats Documentation

### Session Summary
Created comprehensive technical documentation for all supported quantization formats in `docs/FORMATS.md`. This guide details the internal mechanics of FP8, INT8, NVFP4, and MXFP8 formats, including scaling strategies, SVD-based learned rounding optimization, bias correction techniques, and hardware requirements, strictly based on the implementation in the codebase.

### Files Created
- `docs/FORMATS.md`: Comprehensive guide to quantization formats.

---

## 2026-01-23: Tensor-wise INT8 Quantization Support

### Session Summary
Implemented tensor-wise INT8 quantization alongside the existing block-wise implementation. The new mode uses global scaling (1 scale per tensor) and leverages `torch._scaled_mm` for efficient inference in ComfyUI. Learned rounding optimization is supported for this mode.

### Changes Made

1. **New Quantization Format**: Added `int8_tensorwise` support to the entire pipeline.
2. **Efficient Inference**: Implemented `TensorWiseINT8Layout` and corresponding operation handlers using `torch._scaled_mm` for maximum performance.
3. **CLI Enhancements**: Added support for `--scaling_mode tensor` with `--int8`, and relaxed the `block_size` requirement for this mode.

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/constants.py` | Registered `int8_tensorwise` format. |
| `convert_to_quant/comfy/quant_ops.py` | Added `TensorWiseINT8Layout` and operation handlers. |
| `convert_to_quant/converters/learned_rounding.py` | Added tensor-wise INT8 quantization and optimization logic. |
| `convert_to_quant/formats/fp8_conversion.py` | Updated metadata generation and conditional `input_scale` addition. |
| `convert_to_quant/cli/main.py` | Updated argument validation and filename generation for tensor-wise INT8. |

---

## 2026-01-23: Restored Original Adaptive Mode in Learned Rounding

### Session Summary
Restored the original inline tier-based adaptive LR schedule from `learned_rounding_before_refactor.py` to all learned rounding optimizers. The previous refactored abstraction (`adaptive_lr_update()`) was causing issues. Also implemented new dimension-aware `small_mult` formula.

### Changes Made

1. **Restored Original Adaptive Mode Logic**: Replaced calls to centralized `_adaptive_lr_update()` with original inline tier-based logic in all 4 `_optimize_original` methods.

2. **New Dimension-Aware `small_mult` Formula**:
   - Square: `math.gamma((M ** (1/3) / M) + 1)`
   - Tall (M > N): `math.pow(100, M / N^2)`
   - Wide (M < N): `math.pow(10, N / M^2)`

### Files Modified

| File | Changes |
|------|---------|
| `converters/learned_rounding.py` | Restored inline adaptive mode in `_optimize_original()` and `_optimize_int8_original()` |
| `converters/learned_mxfp8.py` | Restored inline adaptive mode in `_optimize_original()` |
| `converters/learned_nvfp4.py` | Restored inline adaptive mode in `_optimize_original()` |

---

## 2026-01-22: Hybrid MXFP8 Support & Edit Quant Enhancements

### Session Summary
Added Hybrid MXFP8 conversion support (`--make-hybrid-mxfp8`), which adds a tensorwise scale fallback to standard MXFP8 models. This enables compatibility with Ada Lovelace (SM 8.9) GPUs which lack native MXFP8 hardware support but can use the tensorwise scale for standard FP8 operations. Also clarified `--edit-quant` functionality for updating existing keys.

Additionally, updated `ComfyUI-QuantOps` to support loading and inference of Hybrid MXFP8 models using `comfy-kitchen` integration.

---

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--make-hybrid-mxfp8` | Convert existing MXFP8 model to Hybrid MXFP8 (adds tensorwise fallback) |
| `--tensor-scales PATH` | Path to tensorwise FP8 model to import scales from (optional, otherwise computed) |

### Features Added

1. **Hybrid MXFP8 Conversion**:
   - Takes standard MXFP8 model (block scales only)
   - Adds `.weight_scalar` (tensorwise scale)
   - Computes scalar from block scales (max) OR imports from external model
   - Updates `.comfy_quant` format to `hybrid_mxfp8`
   - Updates `_quantization_metadata` header

2. **Edit Quant Updates**:
   - Verified `--edit-quant --add-keys` correctly updates existing keys in both layer config and header metadata.
   - Updated help text to clarify "add or update" behavior.

3. **ComfyUI-QuantOps Integration**:
   - Updated `fp8_ops.py` to handle `HybridMXFP8Layout`, reading `.weight_scalar` from state dict.
   - Updated `loader_nodes.py` to add `hybrid_mxfp8` to format options and enable `HybridFP8Ops`.

### Files Modified

| File | Changes |
|------|---------|
| `formats/hybrid_mxfp8_conversion.py` | **NEW** - Implementation of conversion logic and scale computation |
| `cli/main.py` | Added CLI args and dispatch logic for hybrid conversion |
| `cli/argument_parser.py` | Added new args to help sections |
| `constants.py` | Added `hybrid_mxfp8` to `VALID_QUANT_FORMATS` |
| `ComfyUI-QuantOps/fp8_ops.py` | Added Hybrid MXFP8 layout handling and scalar loading |
| `ComfyUI-QuantOps/nodes/loader_nodes.py` | Added format option to loaders |

### Usage

```bash
# Convert MXFP8 to Hybrid (compute scales)
convert_to_quant -i model_mxfp8.safetensors --make-hybrid-mxfp8 -o model_hybrid.safetensors

# Convert using scales from another model
convert_to_quant -i model_mxfp8.safetensors --make-hybrid-mxfp8 --tensor-scales model_fp8.safetensors
```

### Verification

- Created `tests/test_hybrid_mxfp8.py` covering computation, external scales, and metadata updates.
- All tests passed.
