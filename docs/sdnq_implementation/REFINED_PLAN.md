# Refined SDNQ Implementation Plan

This plan aligns `convert_to_quant` and its local `quant_ops.py` with the high-performance `ComfyUI-QuantOps` reference.

## 1. Bit-Packing Alignment (Little-Endian)
All packed formats will switch to Little-Endian (low bits for the first element).

### `convert_to_quant/converters/sdnq_math.py`
- **`pack_uint4`**: Change to `(val1 << 4) | val0`.
- **`unpack_uint4`**: Change to `[byte & 0x0F, byte >> 4]`.
- **`pack_uint2`**: First element in bits 0-1, second in 2-3, etc.
- **`pack_uint1`**: First element in bit 0, second in bit 1, etc.
- **`unpack_uint2/1`**: Mirror the packing logic.

## 2. SVD Orientation Alignment
Align storage with inference-optimized sequential correction: `(X @ svd_down) @ svd_up`.

### `convert_to_quant/converters/sdnq_math.py`
- **`apply_svdquant`**: 
    - Return `svd_down` as `[IC, R]`.
    - Return `svd_up` as `[R, OC]`.
- **`sdnq_quantize_layer_weight`**:
    - Remove the redundant `.t_()` on `quantized_weight`.
    - Ensure `svd_up` and `svd_down` are saved in the new orientation.

## 3. SDNQ Layout Refactor
Implement logical transposition and optimized dispatch.

### `convert_to_quant/comfy/quant_ops.py`
- **`SDNQLayout.Params`**: Add `transposed: bool` and `unpack_shape: Tuple`.
- **`SDNQLayout.dequantize`**:
    - Respect `params.transposed`.
    - Correct SVD reconstruction: `W_corr = (svd_down @ svd_up).t()`.
- **Logical Ops**: Register `aten.t.default` and `aten.transpose.int` to flip the `transposed` flag.
- **Optimized Dispatcher**: 
    - Implement `_sdnq_matmul_optimized`.
    - Apply SVD correction as `(X @ svd_down) @ svd_up` to save memory.
    - Support `_scaled_mm` for FP8/INT8 with correct orientation and alignment.

## 4. Expected Results
- **`int4` (Sepia Noise Fixed)**: Correct bit-order and bias application will restore weight fidelity.
- **`float8` (Colorful Artifacts Fixed)**: Standardizing on `(OC, IC)` storage and handling transposition logically in the dispatcher will fix dimension/stride mismatches in `_scaled_mm`.
- **Performance**: Significant reduction in VRAM usage and increase in speed due to logical transposition and sequential SVD.
