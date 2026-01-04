# Inference Documentation

This document explains how quantized models from `convert_to_quant` are used at inference time in ComfyUI and other frameworks.

## Hardware-Accelerated FP8 Matmul

For FP8 E4M3 quantized models, PyTorch provides `torch._scaled_mm` for hardware-accelerated matrix multiplication on compatible GPUs.

### Requirements

| Feature | Minimum |
|---------|---------|
| GPU | SM ≥ 8.9 (Ada Lovelace) |
| PyTorch | ≥ 2.5.0 |
| Quantization format | `float8_e4m3fn` |

### Usage

```python
import torch

# FP8 quantized matmul
output = torch._scaled_mm(
    input_fp8.contiguous(),  # FP8 input tensor
    weight_fp8,              # FP8 weight tensor
    bias=bias,               # Optional bias
    scale_a=input_scale,     # Per-tensor scale for input
    scale_b=weight_scale,    # Per-tensor scale for weight
    out_dtype=torch.bfloat16,  # Output dtype
)
```

### ComfyUI Integration

ComfyUI's QuantOps system automatically dispatches FP8 quantized linear layers to `torch._scaled_mm` when:
1. Both input and weight are FP8 tensors
2. Hardware supports FP8 (SM ≥ 8.9)
3. Shapes are compatible

---

## NVFP4 (E2M1) Inference

NVFP4 uses 4-bit E2M1 format with 16-element block quantization and two-level scaling (per-tensor + per-block FP8 scales).

### Requirements

| Feature | Minimum |
|---------|---------|
| GPU | SM ≥ 10.0 (datacenter Blackwell) or SM ≥ 12.0 (consumer RTX 50) |
| PyTorch | ≥ 2.5.0 with `torch.float4_e2m1fn_x2` support |
| Block scales | cuBLAS tiled layout |

### Data Layout

```
Quantized data: (M, K // 2) as uint8 (packed FP4)
Block scales:   cuBLAS tiled layout (swizzled)
Per-tensor scale: scalar float32
```

### Hardware Matmul

```python
result = torch._scaled_mm(
    a.view(torch.float4_e2m1fn_x2),   # FP4 packed as special dtype
    b.view(torch.float4_e2m1fn_x2),
    block_scale_a.view(-1),           # Block scales
    block_scale_b.view(-1),
    out_dtype=torch.bfloat16,
)
result = result * (tensor_scale_a * tensor_scale_b)
```

---

## comfy-kitchen Reference

For optimized inference kernels with multi-backend support (eager/cuda/triton), see the [comfy-kitchen](https://github.com/Comfy-Org/comfy-kitchen) library:

```python
import comfy_kitchen as ck

# FP8 quantization/dequantization
qdata = ck.quantize_per_tensor_fp8(tensor, scale)
dequant = ck.dequantize_per_tensor_fp8(qdata, scale, output_dtype)

# NVFP4 quantization/dequantization
qdata, block_scales = ck.quantize_nvfp4(tensor, per_tensor_scale)
dequant = ck.dequantize_nvfp4(qdata, per_tensor_scale, block_scales, output_dtype)
```

The library provides:
- `QuantizedTensor` class for transparent PyTorch op interception
- Automatic backend selection based on hardware constraints
- CUDA C and Triton kernel implementations
