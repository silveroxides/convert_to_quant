# Planned Implementations & Roadmap

This document outlines future improvements and long-term development direction.

---

## 🚀 High Priority

### Quantization Algorithm Improvements
- **Randomized SVD for Faster Computation**: Use `torch.svd_lowrank()` for very large models to speed up optimization by 2-3× with minimal quality loss.
- **Auto-Mixed Precision Strategy**: Automatically decide between FP8 and INT8 per layer based on quantization error assessments.
- **Dynamic Learning Rate Tuning**: Improve the `adaptive` schedule to react faster to loss plateaus on large tensors.

### Model Architecture Support
- **Upcoming Video Models**: Prepare handling for next-gen video diffusion architectures.
- **SDXL Refiner Optimization**: Specific presets for refiner-style models.

---

## 🛠️ Kernel Optimization
- **Triton V2 Integration**: Stabilize and benchmark the autotuning Triton kernels for INT8.
- **Custom NVFP4 Kernels**: Researching standalone CUDA kernels for Blackwell formats to reduce dependency on `comfy-kitchen`.

---

## 🧪 Future Research
- **Activation Quantization at Inference**: Implement dynamic activation quantization to further reduce peak memory usage during inference.
- **Flexible Dtype Support**: Exploring FP16 and BF16 target formats for specialized "cleanup" tasks.

---

_Last updated: 2026-01-24_
