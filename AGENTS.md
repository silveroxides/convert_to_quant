# AI Agent Instructions

> [!IMPORTANT]
> **Always activate the virtual environment** before running any commands.  
> **After completing ANY task**, add an entry to [DEVELOPMENT.md](DEVELOPMENT.md):
> - Add at the top (reverse chronological order)
> - Include: Session Summary, Files Modified, Usage examples

---

## Critical Directives

1. **Document all work** → Update [DEVELOPMENT.md](DEVELOPMENT.md) after each task
2. **Follow Python style** → See [python.instructions.md](python.instructions.md) (PEP 8, type hints)
3. **Minimal ComfyUI changes** → Only modify `comfy/` files, avoid core infrastructure
4. **Test before completing** → Syntax check, verify with `--help`, manual test if possible

---

## Project Context

**Quantization research workspace** for ComfyUI inference models.

| Component | Location |
|-----------|----------|
| Main script | `convert_to_quant/convert_to_quant.py` |
| Kernels | `convert_to_quant/comfy/` (nf4, int8, fp8) |
| Layout classes | `convert_to_quant/comfy/quant_ops.py` |

**Supported formats**: FP8 (tensor/row/block), INT8 (blockwise/lodewise), NF4, FP4

---

## Quick Reference

| Task | Documentation |
|------|---------------|
| Architecture details | [.github/copilot-instructions.md](.github/copilot-instructions.md) |
| Active work | [ACTIVE.md](ACTIVE.md) |
| Planned features | [PLANNED.md](PLANNED.md) |
| ComfyUI patterns | [quantization.examples.md](quantization.examples.md) |
| End-user guide | [MANUAL.md](MANUAL.md) |
| Past findings | [DEVELOPMENT.md](DEVELOPMENT.md) |

---

## Common Commands

```bash
# Basic FP8
convert_to_quant -i model.safetensors --comfy_quant

# INT8 with heuristics
convert_to_quant -i model.safetensors --int8 --comfy_quant --heur

# NF4 4-bit
convert_to_quant -i model.safetensors --nf4 --comfy_quant

# Three-tier quantization
convert_to_quant -i model.safetensors --fp4 --fallback=fp8 \
    --custom-layers=".*pattern.*" --custom-type=int8 --comfy_quant
```

---

## Key Patterns

### Adding New Model Support
```python
# 1. Add exclusion list
MODEL_AVOID_KEY_NAMES = ["norm", "bias", ...]

# 2. Add CLI flag
parser.add_argument("--my_model", action='store_true')

# 3. Add logic in convert_to_fp8_scaled()
if my_model and any(n in key for n in MODEL_AVOID_KEY_NAMES):
    exclusion_reason = "MY_MODEL exclusion"
```

### ComfyUI Metadata Format
```python
# Per-tensor .comfy_quant metadata (stored as JSON in safetensor)
comfy_quant = {
    "format": "float8_e4m3fn",  # or float8_e4m3fn_rowwise, float8_e4m3fn_blockwise,
                                # int8_blockwise, int8_lodewise, bnb_nf4, bnb_fp4
    "params": {"group_size": 64},  # Fork only: per-layer override (upstream ignores)
}
# Note: params.group_size is read by fork's support_bnb_quant branch.
# Upstream uses QUANT_ALGOS defaults only.
```

---

## Gotchas

- INT8 dimensions must be divisible by `block_size`
- FP8 requires PyTorch 2.1+ and Ada/Hopper GPU
- Scale tensors must be clamped to `1e-8` minimum
- Triton kernels expect `(N, K)` weight format

---

## Constraints

- **No diffusers/transformers** dependencies
- **Safetensors only** for serialization
- **Don't modify** `model_management.py`, `sd.py`, or core loaders
