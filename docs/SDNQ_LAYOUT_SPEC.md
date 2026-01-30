# SDNQ Layout Specification for Optimized Inference

## 1. Overview

We have updated the SDNQ (Stochastic Differentiable Neural Quantization) format to natively support `torch._scaled_mm` on consumer hardware (Ada Lovelace/Blackwell). This ensures compatibility with dynamic activation quantization pipelines and eliminates VRAM spikes caused by accidental broadcasting.

## 2. Weight Layout Changes

### 2.1. Main Weights (`.weight`)

* **Format:** `float8_e4m3fn` (or `int8`)
* **Shape:** `[IC, OC]` (K-major / Transposed)
  * *Previous:* `[OC, IC]` (Standard PyTorch)
  * *New:* Weights are pre-transposed to `[K, N]` layout to match the expected input for `torch._scaled_mm` (which expects `X @ W_t`). This avoids runtime transposition overhead.

### 2.2. Weight Scales (`.weight_scale`)

* **Format:** `float32`
* **Shape:** `[1]` (Scalar)
  * *Previous:* `[1, OC]` (Transposed Vector) — **DEPRECATED**
  * *Rationale:*
        1.  **`scaled_mm` Compatibility:** Consumer Tensor Cores (FP8) utilizing `torch._scaled_mm` typically require a single scalar scaling factor for the weight matrix when combined with dynamic input scales.
        2.  **VRAM Safety:** The previous `[1, OC]` layout often caused "outer product" broadcasting collisions against input tensors (e.g., `[Batch, IC]`), leading to massive `[Batch, IC, OC]` intermediate allocations (the "VRAM Explosion").
  * *Fallback:* If `scaled_mm` is disabled, scales revert to `[OC, 1]` (Column Vector) to support standard per-channel dequantization without broadcasting risks.

### 2.3. SVD Components (Optional Low-Rank Correction)

* **`svd_up`**: `[R, OC]` (Transposed)
* **`svd_down`**: `[IC, R]` (Transposed)
* *Implementation Note:* The residual weight `W_int8` is quantized from `W_original - (svd_up @ svd_down)`. At inference, the full weight is reconstructed as `(W_int8 * scale) + (svd_down @ svd_up)`.

## 3. Inference Implementation Guide (QuantOps)

For diffusion models with highly variable inputs (Dynamic Activation Quantization), use the following flow:

**1. Dynamic Activation Quantization:**
Compute scale $S_x$ (scalar or vector) and quantize input $X$ to $X_{f8}$ on-the-fly.

**2. Scaled Matmul:**
Perform the linear operation using the pre-transposed weights and scalar scale.

```python
# X_f8: [Batch, IC] (Input)
# W_f8: [IC, OC]    (Loaded from .weight)
# S_x:  [1] or [B,1] (Dynamic Input Scale)
# S_w:  [1]          (Loaded from .weight_scale)

Output = torch._scaled_mm(
    X_f8, 
    W_f8, 
    scale_a=S_x, 
    scale_b=S_w,
    out_dtype=torch.bfloat16
)
```

**3. SVD Correction (Add-on):**
If `svd_up`/`svd_down` exist, apply the low-rank correction in high precision.

```python
# Correction = X @ (svd_down @ svd_up)
# Optimized: X @ svd_down_t @ svd_up_t (depending on storage layout)
Output += (X_hp @ svd_down) @ svd_up
```

## 4. Summary of Benefits

* **Zero-Overhead Inference:** No runtime transpositions or shape manipulations required.
* **Memory Efficiency:** Scalar scales prevent unintended broadcasting explosions.
* **Hardware aligned:** Directly maps to `cublasLt` / `cutlass` FP8 kernels via `_scaled_mm`.
