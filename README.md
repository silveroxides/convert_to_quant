# convert_to_quant

**Convert safetensors weights to quantized formats (FP8, INT8, NVFP4, MXFP8) with learned rounding optimization for ComfyUI inference.**

[![PyPI version](https://badge.fury.io/py/convert-to-quant.svg)](https://badge.fury.io/py/convert-to-quant)
[![GitHub release](https://img.shields.io/github/v/release/silveroxides/convert_to_quant)](https://github.com/silveroxides/convert_to_quant/releases)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Installation

```bash
pip install convert-to-quant
```

**Or install from source:**

```bash
git clone https://github.com/silveroxides/convert_to_quant.git
cd convert_to_quant
pip install -e .
```

---

## Requirements Summary

| Feature | Requirement |
|---------|-------------|
| **Minimum (FP8/INT8)** | Python 3.10+, PyTorch 2.8+, CUDA 12.8+ |
| **Full (NVFP4/MXFP8)** | Python 3.12+, PyTorch 2.10+, CUDA 13.0+, **[comfy-kitchen](https://github.com/silveroxides/comfy-kitchen)** |
| **Bundled legacy INT8 runtime kernels** | Optional Triton (Linux native, Windows via `triton-windows`) |
| **ConvRot** | SciPy (installed automatically) |

> [!IMPORTANT]
> **PyTorch must be installed manually** with the correct CUDA version for your GPU.
> This package does not install PyTorch automatically to prevent environment conflicts.

---

## Detailed Installation (GPU-Specific)

### 1. Install PyTorch
Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct install command.

**Examples:**

```bash
# CUDA 13.0 (Required for Blackwell NVFP4/MXFP8)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8 (Stable)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. Optional: Triton (bundled legacy/custom runtime kernels)

Weight conversion and current ComfyUI tensor/row INT8 layouts do not require Triton. Install it only when using the bundled Triton runtime kernels.

```bash
# Linux for torch 2.12+
pip install -U "triton<3.8"

# Windows for torch 2.10 and 2.11
pip install -U "triton-windows<3.7"
# Windows for torch 2.12+
pip install -U "triton-windows<3.8"
```

### 3. Optional: Blackwell formats (NVFP4/MXFP8)

```bash
pip install -U "comfy-kitchen>=0.2.18,<0.3"
```

---

## Quick Start
### Use the command 'ctq -hf' to view arguments for layer exclusion presets for various models

```bash
# All examples include metadata and comfy_quant layers for ComfyUI compatible quantization.
# Examples utilize low memory overhead argument to reduce peak RAM/VRAM usage.

# Basic FP8 Tensorcore quantization without learned rounding
ctq -i model.safetensors -o model-fp8mixed.safetensors --comfy_quant --save-quant-metadata --simple --low-memory

# INT8 Row-Wise quantization without learned rounding
ctq -i model.safetensors -o model-int8mixedrow.safetensors --int8 --scaling_mode row --comfy_quant --save-quant-metadata --simple --low-memory

# Blackwell MXFP8 quantization without learned rounding
ctq -i model.safetensors -o model-mxfp8mixed.safetensors --mxfp8 --comfy_quant --save-quant-metadata --simple --low-memory
```

## Use In Code As Module

```bash
# Example modular usage of INT8 Row-Wise quantization of Flux2 Klein 9B
from convert_to_quant import quantize

quantize(
    input="./flux-2-klein-9b.safetensors",
    output="./flux-2-klein-9b-int8mixedrow.safetensors",
    comfy_quant=True,
    save_quant_metadata=True,
    verbose="VERBOSE",
    low_memory=True,
    int8=True,
    scaling_mode="row",
    flux2=True,
    simple=True,
    calib_samples=8192
)
```

Load the output `.safetensors` file in ComfyUI like any other model.

---

## Supported Quantization Formats

| Format | CLI Flag | Hardware | Optimization |
|--------|----------|----------|--------------|
| **FP8 (E4M3)** | *(default)* | Ada/Hopper+ | Learned Rounding (SVD) |
| **INT8 Block-wise**| `--int8` | QuantOps runtime | Learned Rounding (SVD) |
| **INT8 Tensor-wise**| `--int8 --scaling_mode tensor` | Turing+ runtime | Native `_scaled_mm` path |
| **INT8 ConvRot**| `--int8 --scaling_mode row --convrot --convrot-group-size 256` | Turing+ runtime | Native row-wise layout with Hadamard rotation |
| **NVFP4 (4-bit)** | `--nvfp4` | Blackwell | Dual-scale optimization |
| **MXFP8** | `--mxfp8` | Blackwell | Microscaling (E8M0) |

For a deep dive into how these formats work, see **[FORMATS.md](docs/FORMATS.md)**.

---

## Model-Specific Presets

| Model | Flag | Notes |
|-------|------|-------|
| Flux.1 | `--flux1` | Keep modulation/guidance/time/final/input layers high-precision |
| Flux.2 | `--flux2` | Keep modulation/guidance/time/final high-precision |
| FLUX.2 Klein | `--flux_klein` | Flux-sensitive layers without guidance input |
| ERNIE Image | `--ernie_image` | Protect author-recommended time/AdaLN/boundary/projection layers |
| Gemma 4 | `--gemma4` | Protect embeddings, K/V, audio, vision, and multimodal projectors |
| Qwen3.5 | `--qwen35` | Protect size-specific boundaries, embeddings, and complete visual stack |
| Boogu | `--boogu` | Protect embeddings and norm projections |
| LTX 2 / 2.3 | `--ltxv2`, `--ltx2`, `--ltx2_3` | Protect boundary blocks, connectors, VAE, and vocoder |
| T5-XXL | `--t5xxl` | Decoder removed |
| Hunyuan Video| `--hunyuan`| Attention norms excluded |
| WAN Video | `--wan` | Time embeddings excluded |

*(See `--help-filters` for a full list of presets)*

---

## Key Features

- **Learned Rounding**: SVD-based optimization minimizes quantization error.
- **Bias Correction**: Automatic bias adjustment using synthetic calibration data.
- **Lower INT8 Peak Memory**: In-place learned-rounding finalization avoids a second full-size float working tensor.
- **Model-Specific Support**: Exclusion lists for sensitive layers (norms, embeddings).
- **Three-Tier Quantization**: Mix different formats per layer using `--custom-layers`.

---

## Advanced Usage

### Exclude Layer Option
Define specific excluded layers with regex patterns for models with no exclusion preset(This is just example):
```bash
ctq -i model.safetensors --exclude-layers "(double_blocks.[01]|final_layer|txt_attn.proj)" --comfy_quant
```

### Scaling Modes
```bash
# Block-wise scaling for better accuracy
ctq -i model.safetensors --scaling-mode block --block_size 64 --comfy_quant
```

---

## Acknowledgements

Special thanks to:
- [Clybius](https://github.com/Clybius) – For [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) inspiration.
- [lyogavin](https://github.com/lyogavin) – For ComfyUI `int8_blockwise` support.

---

## License

MIT License
