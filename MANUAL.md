# Learned Rounding Quantization Converter - Manual

A powerful tool for converting safetensors model weights to FP8, INT8, NVFP4, or MXFP8 quantized formats with SVD-based learned rounding optimization.

---

## 🚀 Quick Start

### Standard FP8 Conversion (ComfyUI compatible)
```bash
convert_to_quant -i model.safetensors --comfy_quant
```

### INT8 Block-wise Conversion
```bash
convert_to_quant -i model.safetensors --int8 --block_size 128 --comfy_quant
```

### Blackwell NVFP4 (4-bit) Conversion
```bash
convert_to_quant -i model.safetensors --nvfp4 --comfy_quant
```

---

## 🛠️ CLI Reference

### Required Arguments
- `-i`, `--input`: Path to input safetensors file.

### Primary Format Selection
- `--int8`: Use INT8 quantization (per-block or per-tensor).
- `--nvfp4`: Use Blackwell 4-bit (FP4 E2M1) format.
- `--mxfp8`: Use Blackwell Microscaling FP8 format.
- `--scaling_mode [tensor|row|block]`: Strategy for computing scales (default: `tensor`).

### Optimization & Speed
- `--simple`: Disable SVD-based learned rounding for fast conversion.
- `--num_iter N`: Number of optimization iterations (default: 1000).
- `--optimizer [original|adamw|radam]`: Optimization algorithm.
- `--heur`: Skip layers with poor quantization characteristics.

### Model-Specific Presets
Apply architecture-aware exclusions to maintain quality:
- `--flux2`, `--distillation_large`, `--nerf_large`
- `--wan`, `--hunyuan`
- `--qwen`, `--zimage`, `--t5xxl`, `--mistral`

### Advanced Options
- `--actcal`: Calibrate activation scales using simulated PTQ.
- `--low-memory`: Enable streaming mode for models exceeding system RAM.
- `--layer-config`: Use a JSON file for per-layer format overrides.

---

## 📚 Technical Documentation

For detailed information on data layouts, scaling mathematics, and how the SVD optimization pipeline works, please refer to:

👉 **[FORMATS.md](docs/FORMATS.md)**

---

## 📖 How It Works

1. **Analysis**: The tool scans the model and identifies weight tensors, applying model-specific exclusion filters.
2. **SVD Optimization**: It performs Singular Value Decomposition on the weight matrix and optimizes the quantized values to minimize error along the principal directions.
3. **Bias Correction**: Synthetic inputs are used to simulate layer activations and adjust biases to compensate for the mean shift introduced by quantization.
4. **Metadata Generation**: If `--comfy_quant` is enabled, it injects layout metadata so loaders can automatically handle the format.
