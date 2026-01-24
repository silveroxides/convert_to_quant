# convert_to_quant

**Convert safetensors weights to quantized formats (FP8, INT8) with learned rounding optimization for ComfyUI inference.**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Installation

> [!IMPORTANT]
> **PyTorch must be installed first** with the correct CUDA version for your GPU.
> This package does not install PyTorch automatically to avoid conflicts with your existing setup.

### Step 1: Install PyTorch (GPU-specific)

Visit [pytorch.org](https://pytorch.org/get-started/locally/) to get the correct install command for your system.

**Examples:**

```bash
# CUDA 13.0 (newest)
pip install torch --index-url https://download.pytorch.org/whl/cu130

# CUDA 12.8 (stable)
pip install torch --index-url https://download.pytorch.org/whl/cu128

# CUDA 12.6
pip install torch --index-url https://download.pytorch.org/whl/cu126

# CPU only (no GPU acceleration)
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install convert_to_quant

```bash
# Install from PyPI (when available)
pip install convert_to_quant

# Or install from source
git clone https://github.com/silveroxides/convert_to_quant.git
cd convert_to_quant
pip install -e .
```

### Optional: Triton (needed for INT8)

```bash
On Linux
pip install -U triton

On Windows
for torch>=2.9
pip install -U "triton-windows<3.6"
for torch>=2.8
pip install -U "triton-windows<3.5"
for torch>=2.7
pip install -U "triton-windows<3.4"
for torch>=2.6
pip install -U "triton-windows<3.3"
```

---

## Quick Start

```bash
# Basic FP8 quantization
convert_to_quant -i model.safetensors

# FP8 with ComfyUI metadata (recommended)
convert_to_quant -i model.safetensors --comfy_quant

# With custom learning rate (adaptive schedule by default)
convert_to_quant -i model.safetensors --comfy_quant --lr 0.01

# With plateau LR schedule for better convergence
convert_to_quant -i model.safetensors --comfy_quant --lr_schedule plateau --lr_patience 9 --lr_factor 0.92
```

Load the output `.safetensors` file in ComfyUI like any other model.

---

## Supported Quantization Formats

| Format | CLI Flag | Hardware | Optimization |
|--------|----------|----------|--------------|
| **FP8 (E4M3)** | *(default)* | Ada/Hopper+ | Learned Rounding (SVD) |
| **INT8 Block-wise**| `--int8` | Any GPU | Learned Rounding (SVD) |
| **INT8 Tensor-wise**| `--int8 --scaling_mode tensor` | Any GPU | High-perf `_scaled_mm` |
| **NVFP4 (4-bit)** | `--nvfp4` | Blackwell | Dual-scale optimization |
| **MXFP8** | `--mxfp8` | Blackwell | Microscaling (E8M0) |

For a deep dive into how these formats work and their technical implementation, see **[FORMATS.md](docs/FORMATS.md)**.

---

## Model-Specific Presets

| Model | Flag | Notes |
|-------|------|-------|
| Flux.2 | `--flux2` | Keep modulation/guidance/time/final high-precision |
| Chroma / Radiance | `--distillation_large` / `--nerf_large` | Distillation layers excluded |
| T5-XXL Text Encoder | `--t5xxl` | Decoder removed |
| Mistral Text Encoder | `--mistral` | Norms/biases excluded |
| Visual Encoder | `--visual` | MLP layers excluded |
| Hunyuan Video | `--hunyuan` | Attention norms excluded |
| WAN Video | `--wan` | Time embeddings excluded |
| Qwen Image | `--qwen` | Image layers excluded |
| Z-Image | `--zimage` / `--zimage_refiner` | Refiner excludes context/noise refiner |

---

## Documentation

- 📖 **[MANUAL.md](MANUAL.md)** - Complete usage guide with examples and troubleshooting
- 📚 **[FORMATS.md](docs/FORMATS.md)** - Technical reference for quantization formats and SVD optimization
- 📋 **[AGENTS.md](AGENTS.md)** - Developer guide & registry architecture
- ✨ **[ACTIVE.md](ACTIVE.md)** - Current status and active implementations
- 🧪 **[DEVELOPMENT.md](DEVELOPMENT.md)** - Changelog and research notes
- 🔗 **[quantization.examples.md](quantization.examples.md)** - ComfyUI integration patterns

---

## Project Structure

```
convert_to_quant/
├── convert_to_quant/            # Main package
│   ├── cli/                     # CLI entry point & argument parsing
│   ├── converters/              # Core quantization logic (FP8, INT8, NVFP4)
│   ├── formats/                 # Format-specific conversion flows
│   ├── comfy/                   # ComfyUI integration components
│   ├── config/                  # Layer configuration & templates
│   ├── utils/                   # Shared utilities (tensor, memory)
│   ├── constants.py             # Model Filter Registry & constants
│   └── convert_to_quant.py      # Backward-compatibility wrapper
├── pyproject.toml               # Package configuration
├── MANUAL.md                    # User documentation
└── ...
```

---

## Key Features

- **Learned Rounding**: SVD-based optimization minimizes quantization error in weight's principal directions
- **Multiple Optimizers**: Original (adaptive LR), AdamW, RAdam
- **Bias Correction**: Automatic bias adjustment using synthetic calibration data
- **Model-Specific Support**: Exclusion lists for sensitive layers (norms, embeddings, distillation)
- **Triton Kernels**: GPU-accelerated quantization/dequantization with fallback to PyTorch
- **Three-Tier Quantization**: Mix different formats per layer using `--custom-layers` and `--fallback`
- **Layer Config JSON**: Fine-grained per-layer control with regex pattern matching
- **LR Schedules**: Adaptive, exponential, and plateau learning rate scheduling

---

## Advanced Usage

### Layer Config JSON

Define per-layer quantization settings with regex patterns:

```bash
# Generate a template from your model
convert_to_quant -i model.safetensors --dry-run --layer-config-template layers.json

# Apply custom layer config
convert_to_quant -i model.safetensors --layer-config layers.json --comfy_quant
```

### Scaling Modes

```bash
# Tensor-wise scaling (default)
convert_to_quant -i model.safetensors --scaling-mode tensor --comfy_quant

# Block-wise scaling for better accuracy
convert_to_quant -i model.safetensors --scaling-mode block --block_size 64 --comfy_quant
```

### Additional Help

```bash
# View experimental features
convert_to_quant --help-experimental

# View model-specific filter presets
convert_to_quant --help-filters
```

---

## Usage Examples

### INT8 with performance heuristics
```bash
convert_to_quant -i model.safetensors --int8 --block_size 128 --comfy_quant --heur
```

### Blackwell NVFP4 (4-bit)
```bash
convert_to_quant -i model.safetensors --nvfp4 --comfy_quant
```

## Requirements

- Python 3.9+
- PyTorch 2.1+ (with CUDA for GPU acceleration)
- safetensors >= 0.4.2
- tqdm
- (Optional) triton >= 2.1.0 for INT8 kernels

---

## Acknowledgements

Special thanks to:
- [Clybius](https://github.com/Clybius) – For inspiring me to take on quantization and his [Learned-Rounding](https://github.com/Clybius/Learned-Rounding) repository.
- [lyogavin](https://github.com/lyogavin) – For ComfyUI PR [#10864](https://github.com/comfyanonymous/ComfyUI/pull/10864) adding `int8_blockwise` format support and int8 kernels.

---

## References

- DeepSeek scaled FP8 matmul: https://github.com/deepseek-ai/DeepSeek-V3
- JetFire paper: https://arxiv.org/abs/2403.12422

---

## License

MIT License
