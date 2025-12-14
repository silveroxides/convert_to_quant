# Development Log

## 2025-12-14: JSON Layer Config for Per-Layer Quantization

Added `--layer-config PATH` and `--dry-run create-template`:
- Specificity-based pattern matching (numbers+8chars → internal matches → prefix)
- `*` wildcards (fnmatch)
- Strict validation (error on unknown format)
- **Template generation**: `--dry-run create-template` scans model and creates template JSON

**Example:**
```bash
# Generate template
convert_to_quant -i model.safetensors --dry-run create-template

# Use template
convert_to_quant -i model.safetensors --layer-config model_layer_config_template.json --comfy_quant
```

---

## 2025-12-14: Custom Scaling Mode for Mixed Precision FP8

Added `--custom-scaling-mode {tensor,row,block,block2d}` to override FP8 scaling mode for custom-type layers.

---

## 2025-12-14: FP8 Scaled to Comfy Quant Conversion Mode

### Session Summary
Added `--convert-fp8-scaled` mode for offline conversion of legacy `fp8_scaled` format to `comfy_quant` format.

---

### Problem
ComfyUI's `utils.py::convert_old_quants()` incorrectly converts high-precision layers with dummy `.scale_weight` to FP8. This offline conversion tool does it correctly by detecting FP8 layers purely by **weight dtype** (`float8_e4m3fn`).

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--convert-fp8-scaled` | Enable conversion mode (no quantization, format only) |
| `--hp-filter REGEX` | Validate matched layers are high-precision (error if FP8) |
| `--full-precision-mm` | Set `full_precision_matrix_mult=True` in .comfy_quant metadata |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `convert_fp8_scaled_to_comfy_quant()` function and CLI args |

### Usage

```bash
# Basic conversion
convert_to_quant -i old_model.safetensors --convert-fp8-scaled -o new_model.safetensors

# With high-precision validation
convert_to_quant -i model.safetensors --convert-fp8-scaled --hp-filter=".*final_layer.*" -o out.safetensors

# With full precision matrix mult flag
convert_to_quant -i model.safetensors --convert-fp8-scaled --full-precision-mm -o out.safetensors
```

---

## 2025-12-14: FP8 Row-wise & Block-wise Layouts + ComfyUI Fork Sync


### Session Summary
Implemented two new FP8 scaling modes in `convert_to_quant` and fully synced `quant_ops.py` to ComfyUI fork (`support_additional_fp8` branch).

---

### New FP8 Scaling Modes

| Scaling Mode | Scale Shape | CLI Flag | ComfyUI Format |
|-------------|-------------|----------|----------------|
| Row-wise | `(M,)` | `--scaling_mode row` | `float8_e4m3fn_rowwise` |
| 2D Block-wise | `(M//bs, N//bs)` | `--scaling_mode block2d` | `float8_e4m3fn_blockwise` |

### ComfyUI Fork Sync

Branch: `support_additional_fp8` (from `support_bnb_quant`)

**File: `ComfyUI_temp/comfy/quant_ops.py`**
- Added Triton INT8 and NF4/FP4 kernel imports
- Added `RowWiseFP8Layout` class
- Added `BlockWiseFP8Layout` class
- Added `BlockWiseINT8Layout` and `BlockWiseINT8LayoutLodeWise` classes
- Added `NF4Layout` and `FP4Layout` classes
- Updated `QUANT_ALGOS` with all format entries
- Updated `LAYOUTS` registry
- Added all operation handlers (linear, mm, addmm, view, t, gelu, add_, transpose)

### Upstream vs Fork Metadata Handling

| Feature | Upstream | Fork (support_bnb_quant) |
|---------|----------|--------------------------|
| `params.group_size` | Ignored | Read for per-layer override |
| Block size source | `QUANT_ALGOS` only | `layer_conf.params` → `QUANT_ALGOS` fallback |

---

## 2025-12-12: Custom Layer Quantization with Regex Filtering

### Session Summary
Added three-tier quantization priority system with per-type parameter configuration.

---

### CLI Arguments

| Argument | Description |
|----------|-------------|
| `--fallback {fp8,int8,nf4,fp4}` | Quantization type for excluded layers |
| `--custom-layers PATTERN` | Regex pattern for custom layer matching |
| `--custom-type {fp8,int8,nf4,fp4}` | Quantization type for custom matches |
| `--custom-block-size N` | Block size override for custom-type layers |
| `--custom-simple` | Use simple quantization for custom-type |
| `--custom-heur` | Apply performance heuristics to custom-type |
| `--fallback-block-size N` | Block size override for fallback-type layers |
| `--fallback-simple` | Use simple quantization for fallback-type |

### Auto-enable Behavior
- `--comfy_quant` is auto-enabled when `--custom-type` is used (required for mixed precision)

### Priority Order
1. **Custom** (highest): Layers matching `--custom-layers` regex → use `--custom-type`
2. **Primary**: Normal layers → use primary type (--fp4/--nf4/--int8/--fp8)
3. **Fallback**: Excluded layers → use `--fallback` type (or skip if not set)

### Usage

```bash
# Three-tier with per-type config
convert_to_quant -i model.safetensors --fp4 --block_size=64 --fallback=fp8 \
    --custom-layers=".*txt_attn\\.to_out.*" --custom-type=int8 \
    --custom-block-size=128 --custom-simple
```

---

## 2025-12-12: ComfyUI support_bnb_quant Branch Sync

### Session Summary
Synced `convert_to_quant/comfy/` files with ComfyUI's support_bnb_quant branch for compatibility.

---

### Files Synced

| File | Notes |
|------|-------|
| `quant_ops.py` | Import changed: `import comfy.float` → `from . import float as comfy_float` |
| `nf4_kernels.py` | Adds `NF4_CODEBOOK`, `FP4_CODEBOOK_NORMALIZED` exports |
| `int8_kernels.py` | Synced |
| `float.py` | Synced |

### New Handlers Added
- `int8_gelu`, `int8_transpose_int`, `int8_linear_lodewise`, `int8_view_lodewise`, `int8_transpose_lodewise`

---

## 2025-12-12: Package Setup for pip Installation

### Session Summary
Made `convert_to_quant` installable as a pip package with CLI entry point.

---

### New Files Created

| File | Description |
|------|-------------|
| `pyproject.toml` | PEP 621 package config with CLI entry point and dependencies |
| `setup.py` | Minimal shim for legacy pip compatibility |
| `convert_to_quant/__init__.py` | Package init, exposes `main`, `LearnedRoundingConverter`, `convert_to_fp8_scaled` |
| `convert_to_quant/comfy/__init__.py` | Subpackage init for ComfyUI kernels |

---

### Files Modified

#### `convert_to_quant/convert_to_quant.py`
- Changed `from comfy.*` → `from .comfy.*` (relative imports for package compatibility)

#### `convert_to_quant/comfy/quant_ops.py`
- Changed `from comfy.*` → relative imports (`.int8_kernels`, `.nf4_kernels`, `.float`)

---

### Usage

```powershell
# Activate venv and install
.\venv\Scripts\Activate.ps1
pip install -e .

# Run via CLI
convert_to_quant -i model.safetensors -o output.safetensors --comfy_quant

# Or import as module
from convert_to_quant import main, LearnedRoundingConverter
```

---

## 2025-12-11: NF4/FP4 Quantization Implementation & INT8 Lodewise Fix

### Session Summary
Implemented bitsandbytes-style 4-bit quantization (NF4/FP4) and fixed a critical scale shape mismatch bug in INT8 lodewise quantization.

---

### New Files Created

| File | Description |
|------|-------------|
| `kernels/nf4_kernels.py` | Core 4-bit quantization kernels with NF4/FP4 codebooks, packing utilities, and quant/dequant functions |
| `ComfyUI/comfy/nf4_kernels.py` | Copy of kernel file for ComfyUI integration |

---

### Files Modified

#### `quant_ops.py` (convert_to_quant workspace)
- Added `NF4Layout` and `FP4Layout` classes implementing `QuantizedLayout` interface
- Added `bnb_nf4` and `bnb_fp4` entries to `QUANT_ALGOS` dictionary
- Registered layouts in `LAYOUTS` dictionary
- **Fixed `BlockWiseINT8LayoutLodeWise`**: Changed from delegating to `BlockWiseINT8Layout` to implementing proper per-row scaling `(N, K//block_size)` format

#### `ComfyUI/comfy/quant_ops.py`
- Added NF4/FP4 kernel imports
- Added `NF4Layout` and `FP4Layout` classes
- Added `bnb_nf4` and `bnb_fp4` entries to `QUANT_ALGOS` and `LAYOUTS`

#### `convert_to_quant.py`
- Added CLI arguments: `--nf4`, `--fp4`
- Added `_convert_nf4()` and `_convert_fp4()` methods to `LearnedRoundingConverter`
- Updated `convert()` method routing for NF4/FP4 formats
- Updated format detection in `convert_to_fp8_scaled()` with priority: nf4 > fp4 > int8 > fp8
- Updated comfy_quant tensor creation to use `bnb_nf4`/`bnb_fp4` format names and `absmax` scale key
- Updated output filename generation for NF4/FP4
- **Fixed `_convert_int8()`**: Changed from using `lodewise_weight_quant` (incorrect) to `BlockWiseINT8LayoutLodeWise.quantize` (correct per-row format)
- Cleaned up unused imports (`NF4Layout`, `FP4Layout`)

#### `AGENTS.md`
- Added "bitsandbytes Integration Scope" section with guidelines

---

### Bug Fixes

#### INT8 Lodewise Scale Shape Mismatch
**Symptom:** `RuntimeError: Weight scale shape mismatch: scale.shape=torch.Size([90, 30]), expected (11520, 30)`

**Root Cause:** 
- Conversion script was using `lodewise_weight_quant` which was just an alias for `weight_quant`
- `weight_quant` produces scales with shape `(M//block_size, N//block_size)` (2D block grid)
- ComfyUI's `BlockWiseINT8LayoutLodeWise.dequantize` expected `(N, K//block_size)` (per-row)

**Fix:**
1. Updated `BlockWiseINT8LayoutLodeWise.quantize` in workspace `quant_ops.py` to produce per-row scales
2. Changed `_convert_int8()` to use `BlockWiseINT8LayoutLodeWise.quantize/dequantize` for lodewise backend

---

### Usage

```bash
# NF4 quantization
python convert_to_quant.py -i model.safetensors --nf4 --comfy_quant

# FP4 quantization  
python convert_to_quant.py -i model.safetensors --fp4 --comfy_quant

# INT8 lodewise (now fixed)
python convert_to_quant.py -i model.safetensors --int8 --kernel_backend lodewise --comfy_quant
```

---

### Technical Notes

#### NF4 Codebook (from QLoRA paper)
16 values optimized for normal distribution:
```
[-1.0, -0.696, -0.525, -0.395, -0.284, -0.185, -0.091, 0.0,
  0.080, 0.161, 0.246, 0.338, 0.441, 0.563, 0.723, 1.0]
```

#### 4-bit Storage Format
- Packed `uint8`: 2 values per byte (4 bits each)
- `absmax`: per-block absolute maximum for scaling
- Default block size: 64

#### Scale Shape Conventions
| Layout | Weight Scale Shape |
|--------|-------------------|
| `BlockWiseINT8Layout` | `(M//block_size, N//block_size)` |
| `BlockWiseINT8LayoutLodeWise` | `(N, K//block_size)` |
| `NF4Layout` / `FP4Layout` | `(num_blocks,)` where `num_blocks = numel // block_size` |
