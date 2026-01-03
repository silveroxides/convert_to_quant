# FP8 Activation Scale Calibration - Handoff Document

> **STATUS: REVIEWED - Ready for Testing**  
> **Branch:** `feature/calibrated-input-scale`  
> **Date:** 2026-01-02 (Review completed)

---

## Review Completed

Systematic review completed. Fixed 5 issues:
1. Block-wise FP8 dequantization (was using wrong broadcast)
2. Added block_size inference from scale shape
3. LoRA key matching (explicit key_map, no fuzzy matching)
4. CLI seed bug (added `--actcal-seed`)
5. Metadata sync for comfy_quant models

---

## Problem Statement

FP8 quantized diffusion models have `input_scale = 1.0` hardcoded, which causes activation overflow/underflow. ComfyUI's [QUANTIZATION.md](file:///f:/convert_to_quant/ComfyUI/QUANTIZATION.md) specifies PTQ calibration to compute proper `input_scale` values.

---

## What Exists

### Files Created

| File | Status | Description |
|------|--------|-------------|
| `convert_to_quant/calibrate_activation_scales.py` | **NEEDS REVIEW** | Calibration module |
| `convert_to_quant/convert_to_quant.py` | Modified | CLI integration |
| `DEVELOPMENT.md` | Updated | Session log |

### CLI Arguments Added

```
--actcal                  Enable calibration mode
--actcal-samples N        Number of samples (default: 64)
--actcal-percentile F     Percentile for absmax (default: 99.9)
--actcal-lora PATH        LoRA file for informed calibration
```

### Current Implementation Flow

```
1. Load FP8 model (safetensors)
2. For each FP8 layer:
   a. Find weight and weight_scale
   b. Dequantize: w_float = w_fp8 * weight_scale
   c. Find matching LoRA_A if --actcal-lora specified
   d. Generate calibration inputs (random or LoRA-informed)
   e. Simulate forward: activations = X @ W.T
   f. Compute input_scale = absmax / 448.0
3. Patch model with computed input_scale values
4. Save output
```

---

## KNOWN ISSUES & GAPS

### Issue 1: Weight Dequantization (Partially Fixed)

**Status:** Fixed but needs verification

The original code did `w_float = weight.to(float32)` without applying `weight_scale`. This was fixed to:
```python
w_float = weight.to(float32) * weight_scale
```

**What needs review:**
- Is blockwise scale broadcasting correct?
- Are all scale shapes handled (scalar, per-row, 2D block)?

### Issue 2: FP8 State Dict Structure

**Status:** NOT VERIFIED

A quantized FP8 safetensor may contain:
```
layer.weight           # FP8 tensor
layer.weight_scale     # OR layer.scale_weight (legacy)
layer.input_scale      # OR layer.scale_input (legacy)
layer.bias             # May or may not exist
layer.comfy_quant      # Metadata tensor (if comfy format)
```

**What needs review:**
- Does the code handle ALL these key patterns?
- What about blockwise with `block_size` in comfy_quant?
- What about rowwise scales?

### Issue 3: LoRA Key Matching

**Status:** NOT VERIFIED

LoRA layer names may not exactly match model layer names. Current matching:
```python
if base_name.endswith(lora_base) or lora_base.endswith(base_name) or base_name == lora_base
```

**What needs review:**
- Is this sufficient for Flux/SD3/etc model naming conventions?
- Should we normalize prefixes (e.g., strip `transformer.`, `model.diffusion_model.`)?

### Issue 4: LoRA Down/Up Naming

**Status:** Fixed but needs verification

Handles: `.lora_A`, `.lora_B`, `.lora_down`, `.lora_up`, `.down`, `.up`, `.alpha`

**What needs review:**
- Are there other naming conventions?
- Is alpha actually used for anything?

### Issue 5: Activation Scale Formula

**Status:** NOT VERIFIED

Current: `input_scale = absmax / FP8_MAX`

**What needs review:**
- Is this the correct formula per ComfyUI's expectations?
- Should it be `absmax * safety_margin / FP8_MAX`?
- Does the scale need to be inverted for inference?

### Issue 6: Block-wise FP8 Models

**Status:** NOT REVIEWED

Block-wise FP8 has 2D weight_scale tensors. 

**What needs review:**
- Does dequantization handle 2D scales correctly?
- Does the output input_scale need to be per-block too?

---

## Files to Review

### Primary: `calibrate_activation_scales.py`

[View file](file:///f:/convert_to_quant/convert_to_quant/calibrate_activation_scales.py)

**Functions to review:**

| Function | Lines | Check For |
|----------|-------|-----------|
| `compute_activation_scale()` | 31-117 | Weight dequantization, scale broadcasting |
| `load_lora_tensors()` | 119-181 | All LoRA key patterns |
| `calibrate_model()` | 183-277 | Layer detection, scale lookup |
| `patch_model_with_scales()` | 280-310 | Output key format |

### Secondary: `convert_to_quant.py`

[View lines 4843-4900](file:///f:/convert_to_quant/convert_to_quant/convert_to_quant.py)

The CLI handler that calls the calibration module.

---

## Reference: Quantized Model Structure

From `convert_to_quant.py`, an FP8 quantized layer contains:

```python
# Comfy format (new)
f"{base_name}.weight"       # FP8 tensor
f"{base_name}.weight_scale" # Float32 scale
f"{base_name}.input_scale"  # Float32 scale (what we're computing)
f"{base_name}.comfy_quant"  # Metadata tensor

# Legacy format
f"{base_name}.weight"       # FP8 tensor
f"{base_name}.scale_weight" # Float32 scale
f"{base_name}.scale_input"  # Float32 scale
```

Weight scale shapes:
- Tensor-wise: `()` or `(1,)` - scalar
- Row-wise: `(out_features,)` 
- Block-wise: `(out_features // block_size, in_features // block_size)`

---

## Systematic Review Checklist

For each function in `calibrate_activation_scales.py`:

- [ ] Read every line
- [ ] Trace data flow for tensor-wise FP8
- [ ] Trace data flow for row-wise FP8
- [ ] Trace data flow for block-wise FP8
- [ ] Check error handling
- [ ] Check edge cases (empty tensors, missing keys)
- [ ] Verify output format matches what convert_to_quant expects

---

## Test Cases to Write

After systematic review, create tests for:

1. **Tensor-wise FP8 model** - scalar weight_scale
2. **Row-wise FP8 model** - 1D weight_scale
3. **Block-wise FP8 model** - 2D weight_scale
4. **With LoRA** - various naming conventions
5. **Legacy format** - scale_weight/scale_input keys
6. **Mixed format** - some layers quantized, some not

---

## How to Continue

1. Read this document completely
2. View `calibrate_activation_scales.py` line by line
3. Create a list of ALL issues found
4. Fix all issues systematically (not one-by-one reactively)
5. Write tests
6. Then test with user's models

---

## References

- [ComfyUI QUANTIZATION.md](file:///f:/convert_to_quant/ComfyUI/QUANTIZATION.md) - Official guidance on PTQ calibration
- [convert_to_quant.py](file:///f:/convert_to_quant/convert_to_quant/convert_to_quant.py) - How models are quantized
- [DEVELOPMENT.md](file:///f:/convert_to_quant/DEVELOPMENT.md) - Session log with CLI usage
