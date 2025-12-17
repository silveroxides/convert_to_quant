# Development Log

## 2025-12-17: INT8 Legacy-to-ComfyQuant Format Converter

### Session Summary
Added `--convert-int8-scaled` mode to convert legacy INT8 quantized models (with `.scale_weight` keys) to the comfy_quant format (with `.weight_scale` keys and `.comfy_quant` metadata).

---

### New CLI Argument

```bash
# Basic conversion
convert_to_quant -i model_int8.safetensors --convert-int8-scaled

# With custom block size and kernel backend
convert_to_quant -i model_int8.safetensors --convert-int8-scaled --block_size 128 --kernel_backend blockwise
```

### Key Transformations

| Old Format | New Format |
|------------|------------|
| `.scale_weight` | `.weight_scale` |
| `.scale_input` | `.input_scale` |
| (none) | `.comfy_quant` metadata with `int8_blockwise` or `int8_lodewise` format |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `convert_int8_to_comfy_quant()` function, `--convert-int8-scaled` CLI arg |

---

## 2025-12-17: NF4/FP4/AF4 Learned Rounding Optimization

### Session Summary
Added SVD-based learned rounding optimization to 4-bit quantization formats with full LR schedule parity (adaptive/exponential/plateau).

---

### New Features

| Feature | Description |
|---------|-------------|
| `--af4` | AF4 (AbnormalFloat4) codebook support, optimized for block_size=64 |
| NF4/FP4/AF4 optimization | Absmax scale refinement via gradient descent |
| Full LR schedules | `adaptive`, `exponential`, `plateau` with all parameters |

### New CLI Argument

```bash
convert_to_quant -i model.safetensors --af4 --block_size 64 --comfy_quant
```

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/comfy/nf4_kernels.py` | Added `AF4_CODEBOOK`, `quantize_af4`, `dequantize_af4`, updated `_get_codebook()` |
| `convert_to_quant/convert_to_quant.py` | Added `_convert_af4`, `_optimize_nf4_learned_rounding`, `_optimize_nf4_original`; updated `_convert_nf4`/`_convert_fp4` to call optimizer; added af4 to CLI, formats, and routing |

### Optimization Strategy

For 4-bit codebook quantization, we optimize **absmax scales** (not indices):
1. Compute SVD of weight matrix → project error onto top-k components
2. Iteratively adjust absmax to minimize projected reconstruction error
3. Re-quantize with optimized absmax

This differs from FP8/INT8 where quantized values are directly adjusted.

### Usage

```bash
# NF4 with optimization (default)
convert_to_quant -i model.safetensors --nf4 --block_size 64 --comfy_quant -n 100

# FP4 with plateau schedule
convert_to_quant -i model.safetensors --fp4 --block_size 64 --comfy_quant -n 200 --lr_schedule plateau

# AF4 (AbnormalFloat4) - optimized codebook for block_size=64
convert_to_quant -i model.safetensors --af4 --block_size 64 --comfy_quant

# Simple quantization (no optimization)
convert_to_quant -i model.safetensors --nf4 --block_size 64 --comfy_quant --simple
```

---

## 2025-12-16: Shape-Adaptive Plateau Schedule & Early Stopping Controls

### Session Summary
Added shape-aware LR scaling for plateau schedule and configurable early stopping thresholds. New `--help-advanced` section to keep main help clean.

---

### New CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr-shape-influence` | 1.0 | Scale plateau patience/factor based on tensor aspect ratio (0.0=off, 1.0=full) |
| `--early-stop-loss` | 1e-8 | Loss threshold for early stopping |
| `--early-stop-lr` | 1e-10 | LR floor threshold for early stopping |
| `--early-stop-stall` | 1000 | Worse loss counter threshold for early stopping |

### New Help Section

```bash
convert_to_quant --help-advanced  # or -ha
```

Shows shape-adaptive LR and early stopping options (hidden from main `--help`).

### Shape-Adaptive Plateau Scaling

When using `--lr_schedule plateau`, the patience, factor, and cooldown are automatically adjusted based on tensor aspect ratio:

| Tensor Shape | Aspect Ratio | Effective Patience (base=9) | Effective Factor (base=0.92) |
|--------------|--------------|----------------------------|------------------------------|
| `[3072, 3072]` | 1.0 | 9 | 0.92 |
| `[12288, 3072]` | 4.0 | 18 | 0.96 |
| `[18432, 3072]` | 6.0 | 22 | 0.97 |

Set `--lr-shape-influence 0.0` to disable and use raw values.

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `ADVANCED_ARGS`, `--help-advanced`, shape-aware plateau, configurable early stopping |

### Usage

```bash
# Plateau schedule with shape-adaptive scaling (default)
convert_to_quant -i model.safetensors --lr_schedule plateau --lr_patience 9 --lr_factor 0.92

# Disable shape scaling
convert_to_quant -i model.safetensors --lr_schedule plateau --lr-shape-influence 0.0

# Custom early stopping thresholds
convert_to_quant -i model.safetensors --early-stop-stall 500 --early-stop-lr 1e-8
```

---

## 2025-12-16: CRITICAL FIX - FP8 Quantization Quality Degradation

### Session Summary
Removed 4 incorrect `.clamp_()` operations that were NOT in the original reference script and were causing quality degradation.

---

### The Bug

In `_convert_fp8()` after each optimizer call, there was an added clamp:
```python
final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)  # WRONG!
```

The reference script (`reference_for_tensor_float8_e4m3fn.py`) does NOT have this clamp.

### Why This Caused Issues

1. The scale is computed to map tensor values into FP8 range
2. The optimizer explores slightly outside the range during gradient descent
3. Clamping destroys the learned rounding corrections
4. PyTorch's `.to(float8_e4m3fn)` already handles saturation naturally

### Fix

Removed all 4 clamp operations from lines 1307, 1310, 1313, 1316.

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Removed 4 clamp operations after optimizer calls |

---

## 2025-12-16: README.md Rewrite

### Session Summary
Rewrote README.md to reflect current project state and added GPU-specific PyTorch installation instructions.

---

### Changes

- **Installation section**: Added PyTorch prerequisite with CUDA version-specific pip commands
- **Package structure**: Updated to reflect actual `convert_to_quant/` package layout
- **Supported formats**: Added NF4 and FP4 to format table
- **Model presets**: Added Z-Image refiner and updated flag names
- **Key features**: Added layer config JSON, LR schedules, three-tier quantization
- **Advanced usage**: Added sections for layer config, scaling modes, and help commands

### Files Modified

| File | Changes |
|------|---------|
| `README.md` | Complete rewrite with current features, proper installation flow |

---

## 2025-12-16: CLI Help Restructuring

### Session Summary
Reorganized `--help` output to reduce bloat by splitting arguments into three sections.

---

### New Help Sections

| Command | Description |
|---------|-------------|
| `--help` | Standard FP8 workflow options only |
| `--help-experimental` or `-he` | Experimental quantization options (INT8, NF4, FP4, custom layers, etc.) |
| `--help-filters` or `-hf` | Model-specific exclusion presets (t5xxl, hunyuan, wan, etc.) |

### Experimental Arguments (16)
Moved to `--help-experimental`:
- Format: `--int8`, `--nf4`, `--fp4`, `--fallback`
- Scaling: `--scaling_mode`, `--block_size`, `--kernel_backend`
- Custom layers: `--custom-layers`, `--custom-type`, `--custom-block-size`, `--custom-scaling-mode`, `--custom-simple`, `--custom-heur`
- Fallback: `--fallback-block-size`, `--fallback-simple`
- Performance: `--heur`

### Filter Arguments (12)
Moved to `--help-filters`:
- Text encoders: `--t5xxl`, `--mistral`
- Diffusion: `--distillation_large`, `--distillation_small`, `--nerf_large`, `--nerf_small`, `--radiance`
- Video: `--wan`, `--hunyuan`
- Image: `--qwen`, `--zimage`, `--zimage_refiner`

### Implementation

Added `MultiHelpArgumentParser` class that:
1. Intercepts `--help-experimental` and `--help-filters` before standard parsing
2. Filters experimental/filter args from main `--help` usage line and options list
3. Provides organized section-specific help output

### Usage

```bash
# Standard help (compact)
convert_to_quant --help

# Experimental options (INT8, NF4, scaling modes, etc.)
convert_to_quant --help-experimental

# Model-specific filters
convert_to_quant --help-filters
```

---

## 2025-12-15: INT8 Optimizer LR Schedule Parity

### Session Summary
Fixed `_optimize_int8_original` to match `_optimize_original` (FP8) LR schedule implementation, and unified the `--lr` default fallback value.

---

### Changes

#### LR Default Fallback Fix
The `.get('lr', 0.5)` fallback was inconsistent with CLI default `8.077300000003e-3`:
- Updated `_optimize_original` line 578
- Updated `_optimize_int8_original` line 1035

#### LR Schedule Port (INT8)
`_optimize_int8_original` was missing all LR schedule features. Now supports:

| Feature | Before | After |
|---------|--------|-------|
| `--lr_schedule exponential` | ❌ | ✅ |
| `--lr_schedule plateau` | ❌ | ✅ |
| `--lr_schedule adaptive` | 4 tiers | 9 tiers |
| `--lr_adaptive_mode` | ❌ | ✅ |
| `--lr_threshold` | ❌ | ✅ |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Fixed LR fallback; ported full LR schedule to `_optimize_int8_original` |

### Usage
```bash
# INT8 with plateau schedule
convert_to_quant -i model.safetensors --int8 --block_size 128 --comfy_quant \
    --optimizer original --lr_schedule plateau --lr_patience 9 --lr_factor 0.92

# INT8 with exponential schedule
convert_to_quant -i model.safetensors --int8 --block_size 128 --comfy_quant \
    --optimizer original --lr_schedule exponential --lr_gamma 0.95
```

---

## 2025-12-15: Regex Pattern Matching for Layer Config

### Changes
- **Switched from fnmatch to regex**: Layer config patterns now use Python `re.search()` instead of `fnmatch.fnmatch()`
- **Empty format validation**: Added validation to reject empty `"format": ""` strings (must use `skip: true` or valid format)
- **Pattern compilation**: Regex patterns are compiled and validated at config load time

### Why Regex?
The fnmatch glob patterns were confusing and didn't match intuitively:
- `*.attn*` did NOT match `double_blocks.0.img_attn.proj` (fnmatch `*` doesn't match `.` the way users expect)
- `*.0.img_mod*` worked but was inconsistent with other patterns

Regex is more predictable: `re.search(pattern, layer_name)` matches the pattern anywhere in the layer name.

### Migration Guide
Old fnmatch patterns → New regex patterns (shown as JSON strings):

| Old (fnmatch) | New (regex in JSON) | Notes |
|---------------|---------------------|-------|
| `*.attn*` | `"attn"` | Match "attn" anywhere |
| `*.0.img_mod*` | `"\\.0\\.img_mod"` | Escape dots for literal `.` |
| `img_in` | `"^img_in$"` | Use anchors for exact match |
| `*.txt_mlp.*` | `"\\.txt_mlp\\."` | Escape dots |

### Example Config
```json
{
  "_default": {"format": "float8_e4m3fn"},
  "attn": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true},
  "\\.0\\.img_mod": {"skip": true},
  "^img_in$": {"skip": true}
}
```

> **JSON Escaping**: Backslashes must be doubled in JSON strings.
> - Regex `\.` (literal dot) → JSON `"\\."`
> - Regex `\d` (digit) → JSON `"\\d"`
> - Regex `\w` (word char) → JSON `"\\w"`

---

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
