# Inference Documentation

This document explains how quantized models from `convert_to_quant` are used at inference time.

---

## 1. FP8 & INT8 (Tensor-wise)
These formats leverage `torch._scaled_mm` for maximum performance on modern GPUs.
- **Hardware Requirement**: NVIDIA Ada Lovelace (SM 8.9) or newer.
- **PyTorch Support**: Native integration via `torch._scaled_mm`.

## 2. Block-wise Formats (INT8 / FP8)
Uses tiled scaling factors stored in a 2D grid.
- **Implementation**: Handled by custom layouts in [`convert_to_quant/comfy/quant_ops.py`](convert_to_quant/comfy/quant_ops.py).
- **Kernels**: Uses Triton-based kernels for optimized matmul if `triton` is installed; otherwise falls back to a high-performance PyTorch implementation.

## 3. Blackwell Formats (NVFP4 / MXFP8)
Requires NVIDIA Blackwell (SM 10.0+) hardware.
- **NVFP4**: 4-bit packed format with dual-scale weights.
- **MXFP8**: Microscaling FP8 with power-of-2 exponents.
- **Compatibility**: Models converted with `--make-hybrid-mxfp8` can fall back to standard FP8 on older GPUs (Ada/Hopper).

---

## Backend Reference: comfy-kitchen
For optimized inference kernels with multi-backend support, see the [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) library. `convert_to_quant` is designed to produce checkpoints fully compatible with the `comfy-kitchen` ecosystem.

For a deep dive into data layouts and scaling math, see **[FORMATS.md](docs/FORMATS.md)**.
