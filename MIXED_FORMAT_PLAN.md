# Mixed Format Support: BNB 4-bit + FP8/INT8

> [!CAUTION]
> **HALLUCINATION NOTE**: Previous agent may have made incorrect assumptions about code structure.
> Next agent: VERIFY all function signatures and existing code before implementing.

## Goal
Add `--custom-layers` support to BNB 4-bit quantization mode with full LearnedRoundingConverter integration.

## Proposed Changes

### convert_to_quant/convert_to_quant.py

#### [MODIFY] convert_to_bnb_4bit function (line ~2634)

Add parameters:
- `custom_layers: Optional[str]` - regex pattern for layers to use custom format
- `custom_type: Optional[str]` - "fp8" or "int8" for matched layers
- `custom_block_size: Optional[int]` - block size for custom layers
- `custom_scaling_mode: Optional[str]` - FP8 scaling mode
- `custom_simple: bool` - use simple quantization (no learned rounding)
- `custom_heur: bool` - apply performance heuristics
- `comfy_quant: bool` - enable comfy_quant metadata format

Logic:
1. Compile `custom_layers` regex
2. Create converter using `create_converter_for_format()` (uses LearnedRoundingConverter unless `--custom-simple`)
3. In quantization loop:
   - If layer matches regex → use FP8/INT8 converter with `.comfy_quant` metadata
   - Else → use BNB 4-bit

#### [MODIFY] main() BNB 4-bit section (line ~5328)

Pass all custom layer args:
```python
convert_to_bnb_4bit(
    args.input, args.output,
    quant_type=args.bnb_quant_type,
    blocksize=args.bnb_blocksize,
    exclude_layers=exclude_list,
    custom_layers=args.custom_layers,
    custom_type=args.custom_type,
    custom_block_size=args.custom_block_size,
    custom_scaling_mode=args.custom_scaling_mode,
    custom_simple=args.custom_simple,
    custom_heur=args.custom_heur,
    comfy_quant=args.comfy_quant,
)
```

---

### ComfyUI-QuantOps/bnb4bit_ops.py

#### [MODIFY] HybridBNB4bitOps.Linear._load_from_state_dict

Add comfy_quant FP8/INT8 detection:
1. Check for `.comfy_quant` key
2. Parse metadata, store scale and FP8/INT8 weight
3. Set `self.is_comfy_quant = True`

#### [MODIFY] forward_comfy_cast_weights

```python
if self.is_bnb_4bit:
    # BNB 4-bit dequantization
elif self.is_comfy_quant:
    # FP8/INT8 dequantization (scale * weight.to(fp32))
else:
    # standard path
```

## Status
- BNB 4-bit loader works (verified)
- Model-specific exclusions added to convert_to_quant (--flux2 etc)
- Mixed format support NOT YET IMPLEMENTED

## Verification
```bash
python convert_to_quant.py model.safetensors --bnb-4bit --flux2 \
    --custom-layers "img_in|txt_in|final_layer" --custom-type fp8 --comfy_quant
```
