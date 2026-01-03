# BNB 4-bit Loader Node Implementation Plan

Implement bitsandbytes-compatible NF4/FP4 model loading in ComfyUI-QuantOps.

## User Review Required

> [!IMPORTANT]
> **Pure PyTorch dequantization**: Unlike the old loader that used `bitsandbytes.matmul_4bit()`, this implementation uses pure PyTorch for dequantization. This avoids the bitsandbytes runtime dependency but runs in dequantized mode (no native 4-bit matmul).

> [!NOTE]  
> **Future optimization**: If bitsandbytes is available at runtime, the forward pass could optionally use `bnb.matmul_4bit()` for native 4-bit inference.

---

## Proposed Changes

### BNB 4-bit Operations Class

#### [NEW] [bnb4bit_ops.py](file:///f:/convert_to_quant/ComfyUI-QuantOps/bnb4bit_ops.py)

**1. Constants (from convert_to_quant):**
```python
NF4_QUANT_MAP = torch.tensor([
    -1.0, -0.6961928, -0.5250730, ..., 0.7229568, 1.0
])
FP4_QUANT_MAP = torch.tensor([...])
```

**2. `tensor_to_dict()` helper** - Decode JSON from uint8 tensor

**3. `dequantize_bnb_4bit()` function:**
- Unpack nibbles from packed uint8
- Look up values in quant_map  
- Scale by absmax per block
- Reshape to original shape

**4. `HybridBNB4bitOps` class extending `comfy.ops.manual_cast`:**

| Component | Purpose |
|-----------|---------|
| `Linear.__init__` | Initialize 4-bit state attributes |
| `Linear._load_from_state_dict` | Parse `.absmax`, `.quant_map`, `.quant_state.bitsandbytes__nf4` keys |
| `Linear._dequantize_weight` | Convert packed 4-bit → float |
| `Linear.forward_comfy_cast_weights` | Dequantize and run F.linear |
| `Linear.forward` | Dispatch to appropriate forward path |

**`_load_from_state_dict` Algorithm:**
1. Pop `{prefix}weight` (packed uint8)
2. Pop `{prefix}weight.absmax` (float32)
3. Pop `{prefix}weight.quant_map` (float32, 16 elements)
4. Pop `{prefix}weight.quant_state.bitsandbytes__nf4` (or `__fp4`)
5. Parse quant_state JSON for shape, blocksize, dtype
6. Store all for dequantization in forward

---

### Loader Node

#### [MODIFY] [loader_nodes.py](file:///f:/convert_to_quant/ComfyUI-QuantOps/nodes/loader_nodes.py)

**Add `BNB4bitUNETLoader` class:**

```python
class BNB4bitUNETLoader:
    """Load BNB 4-bit quantized UNET models (NF4/FP4 format)."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
                "quant_type": (["auto", "nf4", "fp4"],),
            },
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders/quantized"
    DESCRIPTION = "Load BNB 4-bit (NF4/FP4) quantized diffusion models."
    
    def load_unet(self, unet_name, quant_type):
        from ..bnb4bit_ops import HybridBNB4bitOps
        
        unet_path = folder_paths.get_full_path("diffusion_models", unet_name)
        model_options = {"custom_operations": HybridBNB4bitOps}
        model = comfy.sd.load_diffusion_model(unet_path, model_options=model_options)
        return (model,)
```

**Update node registrations:**
```python
NODE_CLASS_MAPPINGS = {
    # ... existing ...
    "BNB4bitUNETLoader": BNB4bitUNETLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    # ... existing ...
    "BNB4bitUNETLoader": "Load Diffusion Model (BNB 4-bit)",
}
```

---

### Module Registration

#### [MODIFY] [__init__.py](file:///f:/convert_to_quant/ComfyUI-QuantOps/__init__.py)

Add bnb4bit_ops import and node registration.

---

## File Structure After Changes

```
ComfyUI-QuantOps/
├── __init__.py              # Updated
├── bnb4bit_ops.py           # NEW - HybridBNB4bitOps class
├── fp8_ops.py               # Unchanged  
├── int8_ops.py              # Unchanged
├── nodes/
│   └── loader_nodes.py      # Updated - add BNB4bitUNETLoader
└── ...
```

---

## Verification Plan

### Manual Verification

1. **Create test 4-bit model:**
   ```bash
   cd f:\convert_to_quant
   python -m convert_to_quant.convert_to_quant -i <model.safetensors> --bnb-4bit -o test_nf4.safetensors
   ```

2. **Test loading in ComfyUI:**
   - Place `test_nf4.safetensors` in `ComfyUI/models/diffusion_models/`
   - Use "Load Diffusion Model (BNB 4-bit)" node
   - Connect to sampler and run inference
   - Verify no errors and output is reasonable

3. **Check dequantization accuracy:**
   - Compare output with original model on same prompt/seed
   - Quality should be similar (4-bit inherently lossy)

---

## Implementation Order

1. Create `bnb4bit_ops.py` with constants and dequantization
2. Implement `HybridBNB4bitOps.Linear` class
3. Add `BNB4bitUNETLoader` to `loader_nodes.py`
4. Update `__init__.py` node registration
5. Test with quantized model
