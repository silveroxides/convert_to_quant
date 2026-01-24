# Quantization Formats & Optimization Reference

`convert_to_quant` provides a sophisticated pipeline for converting high-precision model weights (FP16/BF16) into various quantized formats. This document details the technical implementation, scaling strategies, and optimization mechanics of each supported format.

## Overview of Supported Formats

Formats are registered in [`convert_to_quant/constants.py`](../convert_to_quant/constants.py:308) and implemented across the `formats/` and `converters/` directories.

| Format Identifier | Scaling Strategy | Bit Depth | Hardware Focus |
| :--- | :--- | :--- | :--- |
| `float8_e4m3fn` | Per-tensor | 8-bit | Ada Lovelace, Hopper |
| `float8_e4m3fn_rowwise` | Per-row | 8-bit | Accuracy-focused Ada/Hopper |
| `float8_e4m3fn_blockwise`| Per-block (2D) | 8-bit | Tiled GEMM Kernels |
| `int8_blockwise` | Per-block (2D) | 8-bit | General Compatibility |
| `int8_tensorwise` | Per-tensor | 8-bit | High-performance `_scaled_mm` |
| `nvfp4` | Dual-scale Block | 4-bit | Blackwell (SM 10.0+) |
| `mxfp8` | Microscaled Block | 8-bit | Blackwell (SM 10.0+) |

---

## 1. FP8 (Floating Point 8-bit)

All FP8 formats use the `e4m3fn` variant (4 exponent bits, 3 mantissa bits, 1 sign bit). This format provides a wide dynamic range suitable for weight quantization in transformer-based models.

### Per-Tensor FP8 (`float8_e4m3fn`)
Uses a single scalar `weight_scale` for the entire weight matrix.
- **Mechanics:** The weight $W$ is scaled by $S = \text{max}(|W|) / \text{FP8\_MAX}$. During inference, the quantized values are multiplied by $S$ to reconstruct the float representation.
- **CLI:** `convert_to_quant -i model.safetensors`

### Row-wise FP8 (`float8_e4m3fn_rowwise`)
Computes a unique scale factor for every row ($M$ dimension) of the weight matrix.
- **Mechanics:** Provides higher precision for layers where weights exhibit significant variance across different output channels. Stores a scale vector of shape `(M,)`.
- **CLI:** `--scaling_mode row`

### Block-wise FP8 (`float8_e4m3fn_blockwise`)
Divides the weight matrix into 2D tiles (blocks) and computes a scale for each tile.
- **Mechanics:** Uses a `group_size` (default 64) for both dimensions. Stores a scale matrix of shape `(M/G, N/G)`. This format is highly compatible with modern tiled GEMM kernels that process weights in static blocks.
- **CLI:** `--scaling_mode block --block_size 64`

---

## 2. INT8 (Integer 8-bit)

INT8 quantization uses symmetric integer mapping in the range `[-127, 127]`.

### Block-wise INT8 (`int8_blockwise`)
A robust 8-bit format using 2D block scaling.
- **Mechanics:** Weights are partitioned into `block_size` tiles. Unlike FP8, INT8 requires this block-based approach for high accuracy in most diffusion layers. During conversion, the tool ensures dimensions are divisible by the block size (or pads them via `--pad_to_16x` logic in some paths).
- **CLI:** `--int8 --block_size 128`

### Tensor-wise INT8 (`int8_tensorwise`)
Uses a single global scale for the entire tensor.
- **Mechanics:** Optimized for hardware that supports `torch._scaled_mm`. This format is particularly efficient for inference as it avoids the overhead of loading per-block scales during the inner loop of matrix multiplication.
- **CLI:** `--int8 --scaling_mode tensor`

---

## 3. Advanced Hardware Formats (Blackwell)

These formats are designed for NVIDIA's Blackwell architecture and require specialized support in the inference backend.

### NVFP4 (NVIDIA FP4 E2M1)
A 4-bit floating-point format (2 exponent, 1 mantissa) that achieves high compression.
- **Dual-Scaling System:** NVFP4 uses a unique two-level scaling strategy:
    1. **Per-Tensor Scale:** A global float32 scalar (`weight_scale_2`).
    2. **Per-Block Scale:** A per-block float8 (`e4m3fn`) scale (`weight_scale`) for every 16 elements.
- **Refinement:** Since 4-bit precision is highly sensitive, the tool supports `--scale-optimization` (choices: `fixed`, `iterative`, `joint`) to refine these scales alongside the weights.
- **CLI:** `--nvfp4`

### MXFP8 (Microscaling FP8)
Uses FP8 data backed by power-of-2 exponents (E8M0) as scales.
- **Mechanics:** Partitioned into 32-element blocks. Each block has an 8-bit exponent scale. This allows for extremely high dynamic range within a single tensor.
- **CLI:** `--mxfp8`

---

## 4. The Learned Rounding Pipeline

The core "magic" of this tool is the **Learned Rounding** optimization, implemented in [`LearnedRoundingConverter`](../convert_to_quant/converters/learned_rounding.py) and [`LearnedNVFP4Converter`](../convert_to_quant/converters/learned_nvfp4.py).

### SVD-Based Error Projection
Instead of simple rounding, the tool minimizes the error in the output space using Singular Value Decomposition (SVD):
1. **Decomposition:** The weight matrix $W$ is decomposed into $U \Sigma V^T$.
2. **Feature Selection:** We select the top $k$ principal components (controlled by `--top_p`, `--min_k`, `--max_k`).
3. **Loss Function:** The optimizer minimizes the projected error: $\text{Loss} = \| U_k^T (W_{dq} - W) V_{hk}^T \|$, where $W_{dq}$ is the dequantized weight.
4. **Result:** This ensures that the quantization noise is shifted away from the most important "directions" in the weight matrix that the model uses for features.

### Optimizer Controls
- **Algorithms:** Supports `original` (gradient descent), `adamw`, and `radam`.
- **LR Scheduling:** 
    - `adaptive`: A custom cosine-based schedule that boosts learning rates when progress stalls.
    - `plateau`: Reduces learning rate by `--lr_factor` after `--lr_patience` steps without improvement. It is "shape-aware," meaning elongated tensors (high aspect ratio) trigger more aggressive decay via `--lr-shape-influence`.
- **Early Stopping:** Monitors the `worse_loss_counter`. If the loss doesn't improve for `--early-stop-stall` steps, or the learning rate falls below `--early-stop-lr`, optimization terminates to save time.

---

## 5. Advanced Configuration Flags

### Performance Heuristics (`--heur`)
Uses [`should_skip_layer_for_performance`](../convert_to_quant/utils/comfy_quant.py) to analyze tensors. It skips quantization for:
- Very small layers (where quantization overhead exceeds gain).
- Layers with "bad" shapes (dimensions not aligned to hardware-friendly multiples like 8, 16, or 64).

### Custom Layer Targeting (`--layer-config`)
Allows per-layer overrides via a JSON file.
```json
{
  "attn": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true},
  "mlp\\.down_proj": {"skip": true}
}
```
This enables keeping sensitive layers in high precision while quantizing the rest of the model.

### Hybrid Compatibility (`--make-hybrid-mxfp8`)
Creates a "Universal" checkpoint. It stores MXFP8 weights for Blackwell GPUs but computes and attaches a secondary tensor-wise FP8 scale. This allows the same file to run in "Fallback" mode on Ada/Hopper GPUs without needing a separate model file.

### Fast Conversion (`--simple`)
By default, the tool performs heavy SVD-based optimization. The `--simple` flag disables this, enabling high-speed quantization:
- **Mechanics:** Skips SVD decomposition and the iterative optimization loop. It performs direct rounding to the target format after computing initial scales.
- **When to use:** Use for initial testing or when the model is less sensitive to quantization noise. Even in simple mode, **Bias Correction** is still performed (if applicable) to maintain basic accuracy.

### Metadata & Compatibility
- **`--comfy_quant`**: Injects `.comfy_quant` metadata tensors into the safetensors file. This identifies the layout (e.g., `TensorCoreFP8Layout`) and parameters like `group_size` so that ComfyUI and other compatible loaders can handle the weights without manual configuration.
- **`--input_scale`**: Adds a legacy `.input_scale` tensor (FP32, value 1.0). This is required by some older loaders that expect a scaling factor for activations even if the weights are already scaled.
- **`--low-memory`**: Instead of loading the entire model into RAM, this flag streams tensors one by one from disk to GPU and then to the output file. Highly recommended for models that exceed 50% of available system memory.

### Calibration & Bias Correction
Quantization introduces a shift in the mean of the weight distribution. To counter this, the tool performs **Bias Correction**:
1. **Simulation:** It generates random input activations (or uses LoRA-informed directions via `--actcal-lora`).
2. **Comparison:** It compares the output of the original layer ($Y = XW$) and the quantized layer ($Y' = XW_{dq}$).
3. **Adjustment:** It calculates the mean error $\Delta = \mathbb{E}[Y - Y']$ and adjusts the layer's bias vector ($b = b + \Delta$) to neutralize the shift.
This process significantly improves the FID/quality of generated images in diffusion models.
