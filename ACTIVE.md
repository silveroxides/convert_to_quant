# Active Development & Current Implementations

This document tracks current work in progress and actively maintained implementations in the quantization workspace.

---

## Current Focus

### Quantization Method Development
- **SVD-based learned rounding optimization** - Core algorithm for minimizing quantization error.
  - File: [`convert_to_quant/converters/learned_rounding.py`](convert_to_quant/converters/learned_rounding.py)
  - Status: ✅ Stable
  - Optimizers: `original`, `adamw`, `radam`

- **Blackwell Formats (NVFP4 / MXFP8)** - Next-gen 4-bit and 8-bit formats.
  - File: [`convert_to_quant/formats/nvfp4_conversion.py`](convert_to_quant/formats/nvfp4_conversion.py)
  - Status: ✅ Fully Implemented

- **Hybrid Compatibility** - Blackwell/Ada dual-mode checkpoints.
  - File: [`convert_to_quant/formats/hybrid_mxfp8_conversion.py`](convert_to_quant/formats/hybrid_mxfp8_conversion.py)
  - Status: ✅ Stable

---

## Model Support

| Category | Models | Presets | Status |
|----------|--------|---------|--------|
| **Diffusion** | Flux.1/2, SD3, SDXL | `--flux2`, `--distillation_large`, `--nerf_large` | ✅ Tested |
| **Video** | Hunyuan 1.5, WAN 2.1 | `--hunyuan`, `--wan` | ✅ Tested |
| **Image** | Qwen-VL, Z-Image | `--qwen`, `--zimage`, `--zimage_refiner` | ✅ Implemented |
| **Text** | T5-XXL, Mistral | `--t5xxl`, `--mistral` | ✅ Tested |

---

## ComfyUI Integration

### Quantized Model Format
- **Metadata**: `.comfy_quant` JSON-encoded tensor.
- **Dispatch**: `__torch_dispatch__` mechanism in [`convert_to_quant/comfy/quant_ops.py`](convert_to_quant/comfy/quant_ops.py).
- **Layouts**: `TensorCoreFP8Layout`, `RowWiseFP8Layout`, `BlockWiseFP8Layout`, `BlockWiseINT8Layout`, `TensorWiseINT8Layout`.

---

## Documentation Roadmap
- 📚 **[FORMATS.md](docs/FORMATS.md)** - Technical guide for formats and SVD optimization (Newly added).
- 📖 **[MANUAL.md](MANUAL.md)** - End-user CLI reference.
- 🧪 **[DEVELOPMENT.md](DEVELOPMENT.md)** - Chronological development log.

---

_Last updated: 2026-01-24_
