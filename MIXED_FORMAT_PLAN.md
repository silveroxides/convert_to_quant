# Mixed Format Support Plan

> [!NOTE]
> **REPLACED BY BLACKWELL NVFP4**: The previous plan for BNB 4-bit (bitsandbytes/NF4) has been superseded by native Blackwell NVFP4 support which offers better performance and integration.

---

## Current Status

The tool now supports true **Three-Tier Mixed-Precision Quantization** via the unified path in [`convert_to_quant/formats/fp8_conversion.py`](convert_to_quant/formats/fp8_conversion.py).

### Capabilities
1. **Primary Format**: Global setting (e.g., `--nvfp4` or `--int8`).
2. **Custom Layer Targeting**: Override specific layers with a different format via `--custom-layers` (regex) and `--custom-type`.
3. **Fallback Strategy**: Automatically handles excluded layers (e.g., norms/embeddings) by keeping them in high-precision or using a lighter `--fallback` format.

---

## Verified Mixed-Format Workflow

Instead of the legacy BNB 4-bit approach, use the following:

```bash
# Example: NVFP4 primary with FP8 custom overrides for sensitive layers
convert_to_quant -i model.safetensors \
    --nvfp4 \
    --custom-layers "img_in|txt_in|final_layer" \
    --custom-type fp8 \
    --comfy_quant
```

## Related Documentation
- 📖 **[FORMATS.md](docs/FORMATS.md)** - Technical details on mixed-precision layouts.
- 🛠️ **[MANUAL.md](MANUAL.md)** - Usage guide for custom layer configurations.
