# FP8 Activation Scale Calibration - Complete Research Summary

> **Status**: ON ICE (Researched, Not Yet Implemented)  
> **Date**: 2026-01-02  
> **Context Source**: Conversation with user about FP8 `input_scale` problems

---

## Problem Statement

**Question**: Why does `input_scale = weight_scale` work for T5XXL/Mistral but `input_scale = 1.0` produces noise for diffusion models?

**Root Cause**: Diffusion model activations vary 10-1000x across timesteps. Using `input_scale = 1.0` means no activation scaling → FP8 overflow/underflow → noise.

---

## Current Code Behavior

Location: `convert_to_quant/convert_to_quant/convert_to_quant.py` lines 3421-3432

```python
# T5XXL/Mistral: input_scale = weight_scale (works)
if t5xxl or mistral or visual:
    new_tensors[f"{base_name}.input_scale"] = dequant_s.clone()
# Diffusion: input_scale = 1.0 (causes noise)  
else:
    new_tensors[f"{base_name}.input_scale"] = torch.tensor(1.0)
```

---

## ComfyUI Official Guidance

From [ComfyUI/QUANTIZATION.md](file:///f:/convert_to_quant/ComfyUI/QUANTIZATION.md) lines 159-168:

> **Activation quantization requires `input_scale` parameters that cannot be determined from static weights alone.** Since activation values depend on actual inputs, we use **post-training calibration (PTQ)**:
> 
> 1. **Collect statistics**: Run inference on N representative samples
> 2. **Track activations**: Record the absolute maximum (`amax`) of inputs to each quantized layer
> 3. **Compute scales**: Derive `input_scale` from collected statistics
> 4. **Store in checkpoint**: Save `input_scale` parameters alongside weights

---

## Proposed Implementation Approaches

### Approach 1: Full PTQ Calibration (Most Accurate)

Run inference with real data, collect per-layer activation statistics:

```python
def calibrate_activation_scales(model, calibration_data):
    """Hook-based activation statistics collection."""
    fp8_max = 448.0  # float8_e4m3fn max
    
    for batch in calibration_data:
        # Varied timesteps: 0, 250, 500, 750, 999
        model.forward(batch, timestep=t)
        # Collect amax per layer
    
    input_scale = amax_99th_percentile / fp8_max
```

**Pros**: Most accurate  
**Cons**: Requires running inference, slow

### Approach 2: Simulated Calibration (Fast)

Use existing `calibration_data_cache` (random data) to estimate:

```python
# Already exists at lines 3136-3148
X_calib = calibration_data_cache[in_features]
activation_out = X_calib @ W_dequant.T
absmax = activation_out.abs().amax()
input_scale = absmax / fp8_max
```

**Pros**: Uses existing infrastructure, fast  
**Cons**: Less accurate than real data

### Approach 3: Heuristic (Simplest)

```python
input_scale = weight_scale * 0.5  # or other factor
```

**Pros**: Zero cost  
**Cons**: Not data-driven, may be wrong for some layers

---

## Implementation Location

**Correct**: `convert_to_quant/convert_to_quant/convert_to_quant.py`  
**Wrong**: ComfyUI-QuantOps (custom node is for loading, not creating models)

---

## Key Learnings from Session

1. **Bias calibration already uses simulated data** (lines 3485-3507) - same pattern can apply to `input_scale`
2. **The issue is quantization-time**, not inference-time
3. **ComfyUI documents PTQ as the official approach** when `input_scale` is needed

---

## Next Steps (When Resumed)

1. Create feature branch in `convert_to_quant` repo
2. Add CLI flag `--calibrate-activation-scale` or similar
3. Implement simulated calibration using existing `calibration_data_cache`
4. Test on small layer subset first
5. Full validation on diffusion model

---

## References

- [ComfyUI QUANTIZATION.md](file:///f:/convert_to_quant/ComfyUI/QUANTIZATION.md)
- [convert_to_quant.py](file:///f:/convert_to_quant/convert_to_quant/convert_to_quant.py) lines 3421-3432, 3136-3148
- [ACTIVATIONS.md](file:///f:/convert_to_quant/ACTIVATIONS.md) - original research notes
