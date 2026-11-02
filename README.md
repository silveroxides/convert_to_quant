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
| **INT8 Kernels** | Triton (Linux native, Windows via `triton-windows`) |

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

### 2. Optional: Triton (needed for INT8)

```bash
# Linux
pip install -U triton

# Windows for torch 2.10 and 2.11
pip install -U "triton-windows<3.7"
# Windows for torch 2.12
pip install -U "triton-windows<3.8"
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
| **FP8 (E4M3)** | *(default)* | Ada/Hopper+ |
| **INT8 Block-wise**| `--int8` | Datacenter GPU |
| **INT8 ConvRot**| `--int8 --scaling_mode row --convrot --convrot-group-size 256` | Any GPU |
| **INT8 Row-wise**| `--int8 --scaling_mode row` | Any GPU |
| **INT8 Tensor-wise**| `--int8 --scaling_mode tensor` | Any GPU |
| **NVFP4 (4-bit)** | `--nvfp4` | Blackwell |
| **MXFP8** | `--mxfp8` | Blackwell |

### INT8 ConvRot

Learned/AdaRound row-wise INT8 ConvRot:

```bash
ctq -i model.safetensors -o model-int8-convrot.safetensors --comfy_quant --save-quant-metadata --int8 --scaling-mode row --convrot --convrot-group-size 256
```

Add `--simple` to use direct row-wise INT8 quantization without learned/AdaRound optimization:

```bash
ctq -i model.safetensors -o model-int8-convrot-simple.safetensors --comfy_quant --save-quant-metadata --int8 --scaling-mode row --convrot --convrot-group-size 256 --simple
```

Group 256 remains the default when it divides the layer input width. Primary
INT8 ConvRot layers otherwise use regular group 64 when compatible, including
widths such as 2688.


---

## Model-Specific Presets

| Model | Flag | Notes |
|-------|------|-------|
| Flux.2 | `--flux2` | Keep modulation/guidance/time/final high-precision |
| T5-XXL | `--t5xxl` | Decoder removed |
| Hunyuan Video| `--hunyuan`| Attention norms excluded |
| WAN Video | `--wan` | Time embeddings excluded |
| MiniMax H3 | `--minimaxh3` | Patch, conditioning, final, time, and token-refiner layers kept high-precision |

*(See `--help-filters` for a full list of presets)*

---

## Key Features

- **Learned Rounding**: SVD-based optimization minimizes quantization error.
- **Bias Correction**: Automatic bias adjustment using synthetic calibration data.
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
