# Research Report: SDNQ Robust Alignment and Artifact Fixes

## 1. Architectural Strategy
We have pivoted to a "Robust Alignment" strategy that matches the **SDNext** origin for mathematical correctness while adopting the **ComfyUI + comfy-kitchen** patterns for integration.

## 2. Key Findings & Corrections

### A. Bit-Packing Order (Int4 "Sepia Noise" Fix)
- **Finding**: My previous fix erroneously moved to Big-Endian packing. Both the origin (`SDNext`) and the inference platform (`QuantOps`) use **Little-Endian** (`val0 | (val1 << 4)`).
- **Correction**: Reverted `sdnq_math.py` to Little-Endian. The sequence `[v0, v1]` is packed as `v0` in low bits and `v1` in high bits.
- **Impact**: This fixes the "sepia noise" artifacts which were caused by bit-scrambling and incorrect bias shifts.

### B. Logical Transposition (Fidelity & Performance Fix)
- **Finding**: Previously, any `.t()` or `transpose()` call on a `QuantizedTensor` triggered a physical dequantization. This degraded performance and broke optimized matmul dispatch.
- **Correction**: Implemented logical transposition in `quant_ops.py`. A `transposed` metadata flag is now flipped without dequantizing.
- **Support**: Updated `QuantizedTensor.__new__` to support **logical shapes**, ensuring that packed tensors (like `int4`) report their full decompressed shape to the inference engine.

### C. Sequential SVD (Memory Optimization)
- **Finding**: Merging the low-rank SVD correction into the full weight before matmul causes VRAM spikes.
- **Correction**: Implemented sequential correction in `sdnq_linear`: `Y = X @ W_q^T + (X @ svd_down^T) @ svd_up^T`.
- **Orientation**: Maintained `SDNext` orientation (`svd_up` [OC, R], `svd_down` [R, IC]) while using the optimized inference path.

### D. Float8 "Colorful Artifacts" Fix
- **Finding**: Redundant transpositions and inconsistent metadata were causing dimension/stride mismatches in `_scaled_mm`.
- **Correction**: Standardized on `(OC, IC)` storage in the converter and used the logical `transposed` flag to inform the dispatcher. Added `unpack_shape` to metadata to robustly recover original layout during dequantization.

## 3. Discrepancy Resolution Table

| Feature | Origin (SDNext) | Target (Comfy-Kitchen) | Local CTQ (Fixed) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Bit Order** | Little-Endian | Little-Endian | **Little-Endian** | **Aligned** |
| **SVD Storage** | Weight-centric | Inference-centric | **Weight-centric** | **Aligned (Origin)** |
| **SVD Logic** | Weight-merge | Sequential | **Sequential** | **Aligned (Target)** |
| **Transpose** | Physical | Logical | **Logical** | **Aligned (Target)** |
| **Metadata** | Custom | Dataclass/Params | **Standardized Dict** | **Aligned (Target)** |

## 4. Conclusion
By aligning the converter with the source origin's math and the target platform's architecture, we have eliminated the "sand castle" risk. The implementation is now robust, memory-efficient, and mathematically identical to the original SDNQ specification.
