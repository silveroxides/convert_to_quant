# Learned Rounding Quantization Converter - Manual

A tool for converting safetensors model weights to FP8 or INT8 quantized formats with optional SVD-based learned rounding optimization.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Command-Line Arguments](#command-line-arguments)
  - [Required Arguments](#required-arguments)
  - [Output Format Options](#output-format-options)
  - [Quantization Options](#quantization-options)
  - [Model-Specific Filters](#model-specific-filters)
  - [Optimization Options](#optimization-options)
- [Usage Examples](#usage-examples)
- [How It Works](#how-it-works)
- [Model-Specific Guidance](#model-specific-guidance)
- [Troubleshooting](#troubleshooting)

---

## Overview

This script converts neural network weights from full precision (FP16/FP32/BF16) to lower precision formats:

| Format | Description | Use Case |
|--------|-------------|----------|
| **FP8** (`float8_e4m3fn`) | 8-bit floating point | Fast inference on modern GPUs with FP8 tensor cores |
| **INT8** (block-wise) | 8-bit integer with per-block scaling | Broader hardware support, good compression |

### Key Features

- **Learned rounding optimization**: Uses SVD-based optimization to minimize quantization error
- **Multiple optimizer choices**: Original, AdamW, RAdam
- **Bias correction**: Automatically adjusts biases to compensate for quantization error
- **Model-specific filters**: Keep sensitive layers in high precision for various architectures
- **ComfyUI compatible**: Generates `.comfy_quant` metadata for seamless integration

---

## Requirements

Install dependencies:

```bash
pip install torch safetensors tqdm
```

For FP8 support, you need:

- PyTorch 2.1+ with FP8 support
- CUDA GPU with FP8 tensor cores (Ada Lovelace / Hopper architecture)

For INT8, any CUDA GPU or CPU works.

---

## Quick Start

### Basic FP8 conversion (ComfyUI compatible)

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --comfy_quant
```

### Basic INT8 conversion

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --int8 \
    --comfy_quant
```

### Fast conversion (no optimization)

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --comfy_quant \
    --simple
```

---

## Command-Line Arguments

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i`, `--input` | Path to input safetensors file |

### Output Format Options

| Argument | Default | Description |
|----------|---------|-------------|
| `-o`, `--output` | Auto-generated | Output file path. If not specified, generates a descriptive filename |
| `--comfy_quant` | False | Enable ComfyUI-compatible quantization format (adds `.comfy_quant` metadata) |
| `--int8` | False | Use INT8 block-wise quantization instead of FP8 |
| `--kernel_backend` | `blockwise` | Kernel backend for INT8 quantization: `blockwise` (2D tile-level scales) or `lodewise` (per-output-lane scales) |
| `--full_precision_matrix_mult` | False | Add `full_precision_matrix_mult=True` to `.comfy_quant` metadata |
| `--nvfp4` | False | Use NVFP4 (FP4 E2M1) block quantization (requires Blackwell GPU for inference) |

### Quantization Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--scaling_mode` | `tensor` | Scale computation mode: `tensor` (per-tensor) or `block` (per-block) |
| `--block_size` | `64` | Block size for block-wise quantization (both FP8 and INT8) |
| `--simple` | False | Skip SVD optimization, use simple round-to-nearest |
| `--heur` | False | Skip layers with poor quantization characteristics |

### Model-Specific Filters

These flags keep certain layers in high precision (not quantized):

#### Text Encoders

| Argument | Description |
|----------|-------------|
| `--t5xxl` | T5-XXL text encoder (removes decoder, keeps norm/bias layers high precision) |
| `--mistral` | Mistral text encoder exclusions |
| `--visual` | Visual encoder: skip MLP layers (down/up/gate proj) |

#### Diffusion Models (Flux-style)

| Argument | Description |
|----------|-------------|
| `--flux2` | Flux.2: keep modulation/guidance/time/final layers high-precision |
| `--distillation_large` | Keep: `distilled_guidance_layer`, `final_layer`, `img_in`, `txt_in` |
| `--distillation_small` | Keep: `distilled_guidance_layer` only |
| `--nerf_large` | Keep: `distilled_guidance_layer`, `nerf_blocks`, `nerf_image_embedder`, `txt_in` |
| `--nerf_small` | Keep: `distilled_guidance_layer`, `nerf_blocks`, `nerf_image_embedder` |

#### Video Models

| Argument | Description |
|----------|-------------|
| `--wan` | WAN video model layers |
| `--hunyuan` | Hunyuan Video 1.5 layers |

#### Image Models

| Argument | Description |
|----------|-------------|
| `--qwen` | Qwen Image model layers |
| `--zimage` | Z-Image model layers |
| `--zimage_refiner` | Z-Image refiner layers (context_refiner, noise_refiner) |
| `--radiance` | Radiance field layers |

### Optimization Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--optimizer` | `original` | Optimization algorithm: `original`, `adamw`, `radam` |
| `--num_iter` | `500` | Maximum optimization iterations per tensor |
| `--lr` | `0.01` | Learning rate for optimizers |
| `--top_p` | `0.01` | Proportion of SVD principal components to use |
| `--min_k` | `1` | Minimum number of SVD components |
| `--max_k` | `16` | Maximum number of SVD components |

| `--full_matrix` | False | Use full SVD instead of low-rank approximation |
| `--scale-refinement` | `1` | [NVFP4] Number of scale refinement rounds (default: 1) |

### Learning Rate Schedule Options

These options control the learning rate schedule for the `--optimizer original` algorithm:

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr_schedule` | `adaptive` | Schedule type: `adaptive`, `exponential`, or `plateau` |
| `--lr_gamma` | `0.99` | [exponential] Multiplicative decay factor per step |
| `--lr_patience` | `50` | [plateau] Steps without improvement before LR reduction |
| `--lr_factor` | `0.5` | [plateau] Factor to multiply LR by when reducing |
| `--lr_min` | `1e-8` | [plateau] Lower bound on learning rate |
| `--lr_cooldown` | `0` | [plateau] Steps to wait after LR reduction before resuming monitoring |
| `--lr_threshold` | `0.0` | [plateau] Minimum improvement to count as "significant" |
| `--lr_adaptive_mode` | `simple-reset` | [adaptive] Counter reset behavior: `simple-reset` or `no-reset` |

#### Schedule Descriptions

**ExponentialLR** (`--lr_schedule exponential`)

Decays learning rate by a constant factor every step:
```
lr_{t+1} = lr_t × gamma
```
- Suitable for smooth, predictable decay
- Lower `gamma` (e.g., 0.95) = faster decay
- Higher `gamma` (e.g., 0.999) = slower decay

**ReduceLROnPlateau** (`--lr_schedule plateau`)

Reduces learning rate when loss stops improving:
```
if no improvement for `patience` steps:
    lr = lr × factor
    wait for `cooldown` steps before monitoring again
```
- Adapts to training progress automatically
- `threshold` sets minimum improvement to reset patience counter
- `min_lr` prevents LR from dropping too low

**Adaptive** (`--lr_schedule adaptive`)

Tier-based schedule with boost/decay behavior:
- LR **increases** (boost) when loss improves after stalling
- LR **decreases** (decay) progressively during stall periods
- Boost multipliers range from 1.25× to 3× depending on stall duration
- Decay multipliers range from 0.95× to 0.995×

**Adaptive Mode** (`--lr_adaptive_mode`):
- `simple-reset` (default): Counter resets to 0 on every improvement. Boosts use tier 0 (1.25× multiplier).
- `no-reset`: Counter preserves stall history. Boosts use the tier matching how long optimization was stuck.

### Other Options

| Argument | Default | Description |
|----------|---------|-------------|
| `--calib_samples` | `3072` | Number of random samples for bias correction |
| `--manual_seed` | `-1` | Random seed (-1 for random) |

---

## Usage Examples

### Example 1: Flux model with distillation layers preserved

```bash
python convert_to_quant.py \
    -i flux1-dev.safetensors \
    --comfy_quant \
    --distillation_large \
    --optimizer original \
    --num_iter 1000
```

### Example 2: T5-XXL text encoder

```bash
python convert_to_quant.py \
    -i t5xxl.safetensors \
    --comfy_quant \
    --t5xxl \
    --scaling_mode block \
    --block_size 64
```

### Example 3: INT8 with block-wise scaling

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --int8 \
    --comfy_quant \
    --block_size 128 \
    --optimizer adamw \
    --num_iter 300
```

### Example 4: Fast conversion (no optimization)

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --comfy_quant \
    --simple
```

### Example 5: WAN video model

```bash
python convert_to_quant.py \
    -i wan_model.safetensors \
    --comfy_quant \
    --wan \
    --num_iter 500
```

### Example 6: Maximum quality (slow)

```bash
python convert_to_quant.py \
    -i model.safetensors \
    --comfy_quant \
    --optimizer original \
    --num_iter 2000 \
    --max_k 32 \
    --top_p 0.02 \
    --full_matrix
```

---

## How It Works

### Quantization Process

1. **Load weights**: Read all tensors from the input safetensors file
2. **Filter layers**: Apply model-specific exclusions (norm, bias, embeddings, etc.)
3. **For each weight tensor**:
   - Compute scaling factors (per-tensor or per-block)
   - Initial quantization (round to nearest)
   - **Learned rounding optimization** (if enabled):
     - Compute low-rank SVD of the weight matrix
     - Optimize quantized values to minimize error in SVD subspace
   - Apply bias correction using synthetic calibration data
4. **Save**: Write quantized tensors with scale factors and metadata

### Learned Rounding Optimization

Standard quantization uses round-to-nearest, which is suboptimal. This script uses SVD-based optimization:

1. Compute truncated SVD: `W ≈ U_k @ S_k @ V_k^T`
2. The top-k singular vectors capture the most important directions
3. Optimize the quantized values to minimize: `||U_k^T @ (W_dequant - W_orig) @ V_k||`

This preserves the most important weight matrix structure while allowing rounding decisions that may not be optimal per-element but are optimal globally.

### Optimizer Comparison

| Optimizer | Speed | Quality | Notes |
|-----------|-------|---------|-------|
| `original` | Medium | Best | Custom adaptive LR, no autograd overhead |
| `adamw` | Fast | Good | Standard PyTorch AdamW |
| `radam` | Fast | Good | Rectified Adam, good for varying LR |

### FP8 vs INT8

| Aspect | FP8 | INT8 |
|--------|-----|------|
| Range | ±448 (e4m3fn) | ±127 |
| Scale shape | Per-tensor or per-row | 2D grid (M/bs × N/bs) |
| Hardware | FP8 tensor cores | Broad support |
| Precision | Better for outliers | Uniform precision |

### INT8 Kernel Backends

When using `--int8`, you can choose between two Triton kernel backends:

| Backend | Description | Block Sizes | Autotuning | Notes |
|---------|-------------|-------------|------------|-------|
| `blockwise` (default) | 2D tile-level weight scales | Fixed 128x128 | No | Stable, well-tested, matches `BlockWiseINT8Layout` |
| `lodewise` | Per-output-lane weight scales | 16-64 | Yes | Smaller tiles, autotuned, may perform better on some hardware |

**When to use `lodewise`:**
- Experimenting with different kernel configurations
- Smaller matrices where autotuning may find better block sizes
- AB testing against the default backend

**Example:**
```bash
python convert_to_quant.py \
    -i model.safetensors \
    --int8 \
    --kernel_backend lodewise \
    --comfy_quant
```

**Note:** If `lodewise` is requested but unavailable (e.g., missing Triton installation), the script will error with a fallback command suggestion.

---

## Mathematical Foundations

### Quantization Basics

Quantization maps continuous values to a discrete set. For both FP8 and INT8, we use **symmetric quantization**:

```
Q = round(W / scale)
W_reconstructed = Q * scale
```

Where:

- `W` is the original weight
- `Q` is the quantized value
- `scale` is the scaling factor

### FP8 Quantization Math

FP8 (`float8_e4m3fn`) has a representable range of approximately `[-448, 448]`.

**Scale computation:**

```
scale = max(|W|) / 448
```

**Quantization:**

```
W_scaled = W / scale
Q = round(clamp(W_scaled, -448, 448))  # Stored as FP8
```

**Dequantization:**

```
W_dequant = Q * scale
```

For **per-tensor scaling**, one scale value applies to the entire tensor.
For **block scaling**, scale has shape `(out_features, num_blocks, 1)` where blocks partition the input dimension.

### INT8 Block-wise Quantization Math

INT8 uses symmetric range `[-127, 127]` with **2D block-wise scaling**.

For a weight matrix `W` with shape `(M, N)` and block size `bs`:

**Block structure:**

```
W_blocked = W.reshape(M//bs, bs, N//bs, bs).permute(0, 2, 1, 3)
# Shape: (M//bs, N//bs, bs, bs)
```

**Per-block scale computation:**

```
amax = max(|W_blocked|, dim=(-2, -1))  # Shape: (M//bs, N//bs)
scale = amax / 127
```

**Quantization:**

```
Q_blocked = round(clamp(W_blocked / scale, -127, 127))
Q = Q_blocked.permute(0, 2, 1, 3).reshape(M, N)
```

**Dequantization:**

```
Q_blocked = Q.reshape(M//bs, bs, N//bs, bs).permute(0, 2, 1, 3)
W_dequant = (Q_blocked * scale).permute(0, 2, 1, 3).reshape(M, N)
```

### Learned Rounding Optimization

Standard round-to-nearest is locally optimal per element but globally suboptimal. The learned rounding approach optimizes in the SVD subspace.

**Objective:**
Minimize the quantization error projected onto the top-k singular vectors:

```
L = ||U_k^T @ (W_dequant - W) @ V_k||_F
```

Where `U_k`, `V_k` are the top-k left and right singular vectors of `W`.

**Why SVD subspace?** The top singular vectors capture the directions of maximum variance in the weight matrix. Errors in these directions have the most impact on model output.

### Gradient Derivation for INT8 Optimization

For the "original" optimizer (manual gradient descent without autograd), we need to derive the gradient of the loss with respect to the quantized values `Q`.

**Forward pass:**

```
dq = Q * scale           # Dequantization (per-block broadcasting)
error = dq - W           # Reconstruction error
proj_error = U_k^T @ error @ V_k    # Project onto SVD subspace
L = ||proj_error||_F     # Frobenius norm
```

**Backward pass (chain rule):**

1. Gradient w.r.t. projected error:

   ```
   ∂L/∂proj_error = proj_error / L
   ```

2. Gradient w.r.t. reconstruction error:

   ```
   ∂L/∂error = U_k @ (∂L/∂proj_error) @ V_k^T
   ```

3. Gradient w.r.t. dequantized weight:

   ```
   ∂L/∂dq = ∂L/∂error    (since error = dq - W)
   ```

4. **Gradient w.r.t. quantized values Q:**

   Since `dq = Q * scale`, by the chain rule:

   ```
   ∂L/∂Q = ∂L/∂dq * ∂dq/∂Q = ∂L/∂dq * scale
   ```

**Key insight:** The gradient must be **multiplied** by scale (not divided). This is because:

- Dequantization multiplies Q by scale
- The derivative of `Q * scale` with respect to Q is `scale`
- Chain rule: multiply the upstream gradient by this derivative

**Block-wise application:**

```python
grad_direction = U_k @ (proj_error / L) @ V_k^T
grad_blocked = grad_direction.reshape(M//bs, bs, N//bs, bs).permute(0, 2, 1, 3)
grad_Q = (grad_blocked * scale).permute(0, 2, 1, 3).reshape(M, N)
Q_new = Q - lr * grad_Q
```

### Numerical Considerations

**Why use 127 instead of 128 for INT8?**

INT8 range is `[-128, 127]`. Using symmetric range `[-127, 127]`:

- Ensures `0` maps exactly to `0` after quantization
- Avoids asymmetry issues with `-128` having no positive counterpart
- Simplifies dequantization (same scale for positive and negative values)

**Scale minimum clamping:**

To avoid division by zero or numerical instability:

```python
scale = max(scale, 1e-8)
```

**Gradient clipping in Q-space:**

After optimization, clamp and round to valid INT8:

```python
Q_final = round(clamp(Q_refined, -127, 127)).to(int8)
```

---

## Model-Specific Guidance

### Flux / Flux-Dev / Flux-Schnell

```bash
# Standard Flux
--distillation_large

# Flux with NeRF layers
--nerf_large
```

### T5-XXL Text Encoder

```bash
--t5xxl --scaling_mode block --block_size 64
```

### Hunyuan Video

```bash
--hunyuan
```

### WAN Video

```bash
--wan
```

### Qwen Image Models

```bash
--qwen
```

---

## Troubleshooting

### "Dimensions not divisible by block_size"

INT8 block-wise quantization requires tensor dimensions to be divisible by the block size.

**Solutions:**

- Use `--heur` to automatically skip incompatible layers
- Try a different `--block_size` (e.g., 32 instead of 64)
- Use FP8 with `--scaling_mode tensor` instead

### "FP8 dtype not supported"

Your PyTorch or GPU doesn't support FP8.

**Solutions:**

- Update PyTorch to 2.1+
- Use `--int8` instead
- Run on a GPU with FP8 support (RTX 40xx, A100, H100)

### Out of memory

Large models may exceed GPU memory during SVD computation.

**Solutions:**

- Use `--simple` for simple quantization (no learned rounding)
- Reduce `--max_k` (e.g., 8 instead of 16)
- Process on CPU (slower but works)

### Optimization not converging

The loss plateaus or doesn't decrease.

**Solutions:**

- Try a different optimizer: `--optimizer adamw`
- Increase iterations: `--num_iter 1000`
- Adjust learning rate: `--lr 0.001` or `--lr 0.1`

### Poor quality results

Output model produces artifacts or degraded results.

**Solutions:**

- Keep sensitive layers in high precision (use model-specific flags)
- Use `--heur` to avoid problematic layer shapes
- Increase optimization effort: `--num_iter 1000 --max_k 32`
- For critical models, consider FP8 over INT8

---

## Output File Naming

When `--output` is not specified, the script generates a descriptive filename:

```raw
{base}_{format}_{scaling}{flags}_k{min}-{max}_p{top_p}_lr{lr}.safetensors
```

Example:

```raw
model_float8_e4m3fn_tensor_nodist_l_k1-16_p0.01_lr0.01.safetensors
```

---

## License

This tool is provided as-is for research and personal use.
