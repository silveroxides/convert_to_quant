# Reference Summaries

## 1. ComfyUI (`comfy/float.py` and `comfy/quant_ops.py`)
- **Int4/FP4 Packing**: Uses `(val0 << 4) | val1`. This means the first element of a pair is in the high nibble.
- **FP8 Scaling**: Expects scales that can be multiplied directly with the weight tensor.
- **Layouts**: Uses a class-based layout system (`QuantizedLayout`) where each layout defines its own `quantize` and `dequantize` methods.

## 2. SDNext (`modules/sdnq/`)
- **Int4 Packing**: Uses `(val1 << 4) | val0`. The first element (`val0`) is in the low nibble. This matches our current local implementation but **mismatches ComfyUI**.
- **Dequantization**: Highly optimized for `torch.compile` and handles SVD correction during dequantization.
- **Transposition**: Handles `use_contiguous_mm` by transposing and making contiguous where necessary.

## 3. Remote `convert_to_quant`
- **Scale Transposition**: Explicitly calls `scale.t_()` during quantization if `use_quantized_matmul` is active. Our local implementation has this commented out with a note claiming it's for compatibility.
- **Packing**: Matches the SDNext/Local style (`val1 << 4 | val0`).

## 4. Local Implementation Discrepancies
- **Double Transpose**: `sdnq_math.py` transposes the weight to `[IC, OC]`, then `quant_ops.py` transposes it again to `[OC, IC]` before `_scaled_mm`. This likely results in a shape mismatch or incorrect math during matmul.
- **Nibble Order**: If the user's inference engine is following the ComfyUI `fp4` style packing, our `int4` models will appear as "neon garbage" because the nibbles are swapped.
- **Scale Shape**: We force scales to scalar `[1]` for `_scaled_mm`, but if the engine doesn't use `_scaled_mm` and falls back to standard dequantization, it might expect per-channel scales.
