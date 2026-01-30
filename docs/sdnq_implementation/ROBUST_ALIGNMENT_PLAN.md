# Robust SDNQ Alignment Plan (Origin-to-Target)

This plan ensures `convert_to_quant` aligns with the source origin (**SDNext**) for math and the target platform (**ComfyUI + comfy-kitchen**) for layout integration, resolving the "sand castle" architectural risks.

## 1. Mathematical Alignment (Origin: SDNext)
We must match the bit-level packing and SVD storage format of the origin repo.

### Bit-Packing (Little-Endian)
- **Order**: Revert to `val0 | (val1 << 4)`. The first element in the sequence occupies the low bits.
- **Functions**: Update `pack_uint4/2/1` and `unpack_uint4/2/1` in `sdnq_math.py` to strictly follow this Little-Endian sequence.
- **Impact**: This resolves the "sepia noise" in `int4` which was caused by my previous Big-Endian flip.

### SVD Orientation
- **Storage**: Keep `svd_up` as `[OC, R]` and `svd_down` as `[R, IC]`.
- **Math**: Ensure `W = dequant(Q) + (svd_up @ svd_down)` is the standard reconstruction path.

## 2. Integration Alignment (Target: ComfyUI + Kitchen)
We must adopt the architectural patterns of the target inference platform.

### Layout Refactor (`SDNQLayout`)
- **Params Dataclass**: Switch from `Dict` to a `Params` dataclass matching the `comfy-kitchen` pattern.
- **Logical Transpose**: 
    - Add `transposed: bool` and `unpack_shape: Tuple` to `Params`.
    - Implement `torch.ops.aten.t.default` to toggle the flag without dequantization.
    - Update `dequantize()` to handle the flag robustly: `unpack -> SVD -> transpose`.

### Sequential SVD Correction
- **Ops Handler**: Implement logic in `sdnq_linear` to apply SVD correction as `Y = X @ W_q^T + (X @ svd_down^T) @ svd_up^T`.
- **Efficiency**: This provides the memory savings of `QuantOps` while maintaining mathematical alignment with `SDNext`.

## 3. Fidelity Improvements (Bias Correction)
- **Calibration**: Integrate the project's existing bias correction logic into the `SDNQConverter`.
- **Implementation**: Calculate the mean quantization error over representative inputs (or zero-mean approximation) and subtract it from the layer's bias.
- **Impact**: This centers the error distribution, likely resolving the "colorful" artifacts in `float8` by preventing accumulated offsets.

## 4. Summary of Changes
- **`sdnq_math.py`**: Revert bits, align `uint2/1`, ensure stable SVD orientation.
- **`quant_ops.py`**: Switch to `Params` dataclass, add logical transpose, implement sequential SVD in linear op.
- **`sdnq_converter.py`**: Add optional bias calibration loop.
