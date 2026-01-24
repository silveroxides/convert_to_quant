# Activation Scale Calibration

> **STATUS: IMPLEMENTED**  
> **Date:** 2026-01-24

---

## Overview

Quantized models often suffer from activation overflow/underflow if `input_scale` is not properly calibrated. This module computes optimal `input_scale` values using Post-Training Quantization (PTQ) simulation.

---

## Implementation Details

### Core Module
- **File**: [`convert_to_quant/calibrate_activation_scales.py`](convert_to_quant/calibrate_activation_scales.py)
- **Features**:
    - **Simulated Forward Pass**: Runs $Y = XW$ using synthetic or LoRA-informed inputs.
    - **Percentile-based Calibration**: Computes scales based on the 99.9th percentile (default) of absolute activations.
    - **LoRA-Informed Calibration**: Uses LoRA weights to determine the most important activation directions.
    - **Bias Correction**: Integrated with the calibration pass to neutralize mean shift introduced by quantization.

### CLI Usage
```bash
# Basic calibration
convert_to_quant -i model.safetensors --actcal --actcal-samples 64

# LoRA-informed calibration
convert_to_quant -i model.safetensors --actcal --actcal-lora my_lora.safetensors
```

---

## Quantized Model Structure

A calibrated model contains the following tensors per layer:

```python
# Comfy format
f"{base_name}.weight"       # Quantized weight (FP8/INT8/NVFP4)
f"{base_name}.weight_scale" # Weight scaling factor
f"{base_name}.input_scale"  # Calibrated activation scaling factor
f"{base_name}.comfy_quant"  # Metadata tensor
```

For more details on formats and scaling, see **[FORMATS.md](docs/FORMATS.md)**.
