# Local SDNQ Analysis

## Current implementation details:

### 1. Packing (int4)
- **Math**: `bitwise_or(val0, val1 << 4)`
- **Order**: Low nibble first (`val0`), high nibble second (`val1`).

### 2. Transposition logic
- **Quantization**: `sdnq_math.py` calls `quantized_weight.t_()` if `use_quantized_matmul` is true and it's not re-quantized for matmul.
- **Inference**: `quant_ops.py`'s `fp8_linear` calls `weight_t = plain_weight.t()` before `_scaled_mm`.
- **Potential Issue**: Double transposition if the weight is already stored transposed in the safetensors.

### 3. Scaling
- **Math**: `scale = amax / dtype_max`.
- **Shape**: Forced to scalar `[1]` for `_scaled_mm` compatibility in `sdnq_math.py`.

### 4. SVD Correction
- **Implementation**: `correction = torch.mm(svd_up, svd_down)`.
- **Reshape**: Re-views to original shape for Conv layers.

### 5. Potential "Noisy Garbage" causes:
- **Nibble Order Mismatch**: If inference engine expects `(val1, val0 << 4)`.
- **Transposition Mismatch**: `_scaled_mm` expects RHS to be `[K, N]` (transposed weight). If we transpose it twice, we give it `[N, K]`.
- **Signed vs Unsigned**: `sdnq_math.py` uses `torch.uint8` for storage of packed values. If the engine interprets them as signed `int8` before unpacking, bit shifts will be wrong.
- **FP8 Scale Mismatch**: Static/sepia often means the scale is way off or the bias is being added to the wrong dimension.
