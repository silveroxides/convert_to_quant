# FP8 Activation Scale Calibration - Implementation Summary

> **Status**: ✅ IMPLEMENTED  
> **Date**: 2026-01-24  
> **Context**: Resolved FP8 `input_scale` issues for diffusion models using simulated PTQ.

---

## Problem Statement

Diffusion model activations vary significantly across timesteps. Using a static `input_scale = 1.0` often leads to activation overflow or underflow in FP8, resulting in noisy image outputs.

---

## Implementation

The solution uses **Simulated Calibration** (formerly Approach 2 in this research).

### Core Mechanics
- **File**: [`convert_to_quant/calibrate_activation_scales.py`](convert_to_quant/calibrate_activation_scales.py)
- **Process**:
    1. Generates synthetic input activations (random or LoRA-informed).
    2. Runs a simulated forward pass through each quantized layer.
    3. Computes the absolute maximum (`amax`) of the output activations.
    4. Calculates `input_scale = amax / FP8_MAX`.
    5. Injects the computed scale as a `.input_scale` tensor.

### CLI Integration
```bash
# Calibrate with 64 samples
convert_to_quant -i model.safetensors --actcal --actcal-samples 64

# Informed calibration using LoRA directions
convert_to_quant -i model.safetensors --actcal --actcal-lora lora_weights.safetensors
```

---

## Results
- **Quality**: Significantly reduces quantization-induced noise in Flux and other diffusion models.
- **Compatibility**: Produces checkpoints fully compatible with ComfyUI's standard FP8 loading path.

For more details on data layouts, see **[FORMATS.md](docs/FORMATS.md)**.
