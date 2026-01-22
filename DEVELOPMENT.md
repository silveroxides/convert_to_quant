
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
