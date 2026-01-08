# Development Log

## 2026-01-08: Critical FP8 Clamp Fix (Quality Issue)

### Session Summary
Fixed critical missing `.clamp()` calls before `.to(TARGET_FP8_DTYPE)` in most FP8 conversion paths. This could cause values outside FP8 range (±448 for E4M3FN) to overflow, producing NaN or incorrect quantization.

---

### Bugs Fixed

| Line | Method | Path | Fix |
|------|--------|------|-----|
| 1122 | `_convert_fp8` | learned | Added `.clamp(-self.f8_max_val, self.f8_max_val)` |
| 1167 | `_convert_fp8_rowwise` | simple | Added `.clamp()` |
| 1196 | `_convert_fp8_rowwise` | learned | Added `.clamp()` |
| 1240 | `_convert_fp8_block2d` | simple | Added `.clamp()` |
| 1284 | `_convert_fp8_block2d` | learned | Added `.clamp()` |

### Impact
This was likely causing quality degradation in all learned rounding quantizations.

---

## 2026-01-08: Fix Refactoring Bugs (Static Analysis)

### Session Summary
Fixed critical runtime bugs discovered via static analysis (flake8, mypy). These bugs were introduced during the modular refactoring.

---

### Bugs Fixed

| File | Bug | Impact |
|------|-----|--------|
| `formats/format_migration.py` | Missing `import re` | Crash when using `--hp-filter` |
| `formats/nvfp4_conversion.py` | Passed `tensor.shape` instead of `tensor` to `should_skip_layer_for_performance()` | Crash when using `--heur` with `--nvfp4` |
| `formats/int8_conversion.py` | Undefined name `fix_comfy_quant_params_structure` | Crash during INT8-to-comfy_quant conversion |

### Verification

- flake8 F821 (undefined name): ✅ 0 errors (was 1)
- flake8 F401 (unused imports): 21 warnings (cosmetic, non-breaking)
- All format modules import successfully

---

## 2026-01-08: NVFP4 Iterative Scale Refinement

### Session Summary
Added iterative scale refinement for NVFP4 quantization. After learned rounding converges, block scales are recomputed from the optimized weights and optimization reruns with the new scales. This allows scales to better fit the learned values.

---

### Changes

| File | Changes |
|------|---------|
| `converters/learned_nvfp4.py` | Added `scale_refinement_rounds` parameter, `_compute_block_scales()` helper, refinement loop in `convert()` |
| `formats/nvfp4_conversion.py` | Pass `scale_refinement_rounds` to converter |
| `cli/main.py` | Added `--scale-refinement` CLI argument |
| `cli/argument_parser.py` | Added to `ADVANCED_ARGS` for `--help-advanced` |

### Usage

```bash
# Default: no refinement (rounds=1)
convert_to_quant -i model.safetensors --nvfp4

# With 2 refinement rounds
convert_to_quant -i model.safetensors --nvfp4 --scale-refinement 2
```

### Verification

- Syntax check: ✅ All modules pass
- Functional tests: ✅ All 6 tests pass (`test_cli_args.py`)

---


## 2026-01-07: NVFP4 Comfy-Kitchen Compatibility

### Session Summary
Fixed NVFP4 quantization discrepancies to match comfy-kitchen exactly. Added kernel delegation when comfy-kitchen is available.

---

### Changes

| File | Changes |
|------|---------|
| `converters/nvfp4_converter.py` | Uses `ck.quantize_nvfp4()`/`ck.dequantize_nvfp4()` when available; fixed PyTorch fallback |
| `converters/learned_nvfp4.py` | Added comfy-kitchen check, removed `F8_E4M3_EPS` min clamp, added zero-block handling |

### Discrepancies Fixed

| Issue | Before | After |
|-------|--------|-------|
| Block scale clamp | `min=F8_E4M3_EPS, max=F8_E4M3_MAX` | `max=F8_E4M3_MAX` only |
| Zero block handling | Divide-by-zero possible | Safe division with mask |
| Kernel usage | Pure PyTorch only | comfy-kitchen when available |

### Verification

- Syntax check: ✅ All modules pass
- Functional tests: ✅ All 6 tests pass (`test_cli_args.py`)

---


## 2026-01-07: Converter Class Unification (Complete)

### Session Summary
Created `BaseLearnedConverter` ABC to extract shared infrastructure. Both `LearnedRoundingConverter` and `LearnedNVFP4Converter` now inherit from it.

---

### Changes

| File | Changes |
|------|---------|
| `converters/base_converter.py` | **NEW** - Abstract base class with shared `__init__` (17 params), SVD, LR, cleanup |
| `converters/learned_nvfp4.py` | Inherits from base, only defines `block_size`, `pad_to_16x` (-80 lines) |
| `converters/learned_rounding.py` | Inherits from base, only defines `scaling_mode`, `block_size`, `target_format` (-150 lines) |
| `converters/__init__.py` | Added `BaseLearnedConverter` export |

### Deduplication Summary

- **6 SVD computation blocks** → `_compute_svd_components()`
- **Inline LR tier logic** → `_adaptive_lr_update()`  
- **gc.collect/empty_cache patterns** → `_cleanup_tensors()`
- **Shape-aware plateau params** → `_compute_shape_aware_plateau_params()`

### Verification

- Syntax check: ✅ All modules pass
- Functional tests: ✅ All 6 tests pass (`test_cli_args.py`)

### Git

```bash
git checkout feature/converter-unification
# Commits: 59df2e3, d243d96
```

---

## 2026-01-07: Memory-Efficient Tensor Loading (`--low-memory`)

### Session Summary
Added `--low-memory` CLI flag to support streaming tensor loading for large models. Addresses OOM issues when quantizing 60GB+ models with limited RAM.

---

### Changes

| File | Changes |
|------|---------|
| `utils/memory_efficient_loader.py` | **NEW** - `UnifiedSafetensorsLoader` with dual-mode: preload (fast) or streaming (low RAM) |
| `cli/main.py` | Added `--low-memory` flag, passed to FP8 and NVFP4 converters |
| `formats/fp8_conversion.py` | Added `low_memory` param, uses `UnifiedSafetensorsLoader` for all tensor access |
| `formats/nvfp4_conversion.py` | Same - unified loader integration |

### Usage

```bash
# Standard mode (fast, uses 2x model size in RAM)
convert_to_quant -i model.safetensors --int8 --comfy_quant

# Low-memory mode (streaming, ~1x model size in RAM)
convert_to_quant -i model.safetensors --int8 --comfy_quant --low-memory
```

### Technical Details

- `UnifiedSafetensorsLoader` provides consistent interface for both modes
- In standard mode: preloads all tensors (existing behavior)
- In low-memory mode: loads tensors on-demand via `get_tensor()`, cleans up with `mark_processed()`
- Format migration scripts (int8_conversion, format_migration) not updated - they're 1:1 transformations without 2x memory issue

### Verification

- Syntax check: ✅ All modules pass
- CLI --help: ✅ `--low-memory` flag visible

---

## 2026-01-06: DRY Refactor - Centralized Utilities

### Session Summary
Refactored duplicated code patterns identified during code structure audit. Centralized LR tier configuration, calibration data generation, and bias correction utilities.

---

### Changes

| File | Changes |
|------|---------|
| `constants.py` | Added `ADAPTIVE_LR_TIERS_IMPROVE`, `ADAPTIVE_LR_TIERS_DECAY` constants |
| `utils/tensor_utils.py` | Added `generate_calibration_data()`, `adaptive_lr_update()`, `compute_bias_correction()` |
| `utils/__init__.py` | Updated exports for new utilities |
| `converters/learned_rounding.py` | Replaced ~35 lines of inline tier logic with call to `adaptive_lr_update()` |
| `converters/learned_nvfp4.py` | Replaced `_adaptive_lr_update()` method with thin wrapper to shared utility |
| `cli/main.py` | Simplified `nvfp4_excluded` list to cleaner `nvfp4_included` set pattern |

### New Utilities

```python
# tensor_utils.py - shared utilities
generate_calibration_data(tensors, calib_samples, seed, device)  # Calibration data generator
adaptive_lr_update(curr_lr, improved, counter, worse_count, small_mult)  # Tier-based LR update
compute_bias_correction(orig_weight, dequant_weight, bias, calib_data, device)  # Bias correction
```

### Verification

- Syntax check: ✅ All 6 modified files pass
- CLI --help: ✅ All options display correctly
- CLI --help-filters: ✅ Model filters work as expected

---

## 2026-01-06: Model Filter Registry Refactor

### Session Summary
Refactored scattered model filter flags into centralized `MODEL_FILTERS` registry. Adding a new filter now requires editing only `constants.py` instead of 6 files.

---

### Changes

| File | Changes |
|------|---------|
| `constants.py` | Added `MODEL_FILTERS` dict registry + `build_exclusion_patterns()` helper |
| `cli/argument_parser.py` | `FILTER_ARGS` now generated from registry keys, help sections use registry categories |
| `cli/main.py` | Filter argument definitions now generated via loop from registry |
| `formats/fp8_conversion.py` | 50-line conditional block replaced with 26-line registry-driven loop |
| `formats/nvfp4_conversion.py` | Same pattern - exclusion list built from registry |
| `convert_to_quant.py` | Added `MODEL_FILTERS`, `build_exclusion_patterns` to exports |

### Adding New Filter

```python
# In constants.py - single file change
MODEL_FILTERS["mymodel"] = {
    "help": "My model exclusions",
    "category": "diffusion",
    "highprec": ["layer1", "layer2"],
}
```

### Verification

- Syntax check: ✅ All 6 files pass
- CLI --help-filters: ✅ Filters display correctly from registry

---

## 2026-01-06: NVFP4 Console Output & Bias Correction

### Session Summary
Rewrote `nvfp4_conversion.py` to add legacy-style console outputs and bias correction. Now matches FP8 conversion flow with calibration scanning, layer progress, bias correction, and final shapes.

---

### Changes

| File | Changes |
|------|---------|
| `formats/nvfp4_conversion.py` | Complete rewrite: added `calib_samples`/`seed` params, calibration data generation, `(i+1)/(total)` layer progress, bias correction using dequantized weights, final shape outputs |

### New Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `calib_samples` | 3072 | Number of random samples for bias correction |
| `seed` | 42 | Seed for reproducibility |

### Verification

- Syntax check: ✅ Passed

---

## 2026-01-06: Fix Console Output Formatting in Converters

### Session Summary
Fixed SVD print statement formatting in `learned_rounding.py` to match the legacy reference script. Added `    - ` indentation prefix to 6 print statements.

---

### Changes

| File | Changes |
|------|---------|
| `converters/learned_rounding.py` | Added `    - ` prefix to SVD print statements at lines 1236, 1240, 1336, 1340, 1448, 1453 |

### Before/After

```diff
-print("Using torch.linalg.svd with full_matrices=True")
+print("    - Using torch.linalg.svd with full_matrices=True")

-print("Trying svd_lowrank")
+print("    - Trying svd_lowrank")
```

### Verification

- Syntax check: ✅ Passed

---

## 2026-01-06: NVFP4 Format Fixes (Match NVIDIA Official)

### Session Summary
Fixed NVFP4 output format to match NVIDIA FLUX.2-dev-NVFP4 structure. Added `--input-scales` CLI option for calibrated activation scales.

---

### Changes

| File | Changes |
|------|---------|
| `formats/nvfp4_conversion.py` | Block scale as `float8_e4m3fn` (not uint8), per-tensor scale as scalar `[]` (not `[1]`), added `input_scales` param |
| `cli/main.py` | Added `load_input_scales()` helper, `--input-scales` CLI arg (loads JSON or safetensors) |
| `repair_nvfp4_metadata.py` | Squeeze `weight_scale_2` from `[1]` to `[]`, added dtype warning |

### Usage

```bash
# Quantize with calibrated input scales
python -m convert_to_quant -i model.safetensors --nvfp4 --input-scales scales.json

# Or from another NVFP4 model
python -m convert_to_quant -i model.safetensors --nvfp4 --input-scales reference_nvfp4.safetensors
```

### Format Comparison

| Tensor | Before | After (NVIDIA-compatible) |
|--------|--------|--------------------------|
| `.weight_scale` | uint8 | float8_e4m3fn |
| `.weight_scale_2` | `[1]` | `[]` (scalar) |
| `.input_scale` | missing | `[]` (scalar, optional) |

---

## 2026-01-05: NVFP4 (E2M1) Quantization Support

### Session Summary
Added NVIDIA FP4 E2M1 block quantization with two converters: raw (`NVFP4Converter`) and optimized (`LearnedNVFP4Converter` with SVD). Full CLI integration with `--nvfp4` flag.

---

### New Files

| File | Purpose |
|------|---------|
| `utils/float_utils.py` | FP4 encode/decode, uint4 packing, cuBLAS tiled layout |
| `converters/nvfp4_converter.py` | Raw NVFP4Converter (simple quantization) |
| `converters/learned_nvfp4.py` | LearnedNVFP4Converter (SVD optimization, LR schedules) |
| `formats/nvfp4_conversion.py` | File conversion using converter_kwargs pattern |
| `INFERENCE.md` | Runtime/comfy-kitchen reference |

### Changes

| File | Changes |
|------|---------|
| `constants.py` | Added `FP4_E2M1_MAX`, `FP4_BLOCK_SIZE`, `nvfp4` format |
| `cli/argument_parser.py` | Added `nvfp4` to EXPERIMENTAL_ARGS |
| `cli/main.py` | Added `--nvfp4` dispatcher with `nvfp4_kwargs` pattern |

### Usage

```bash
python -m convert_to_quant -i model.safetensors --nvfp4 --comfy_quant  # Optimized
python -m convert_to_quant -i model.safetensors --nvfp4 --simple       # Raw
```

### Git

```bash
git checkout feature/nvfp4-support
```

---

## 2026-01-04: Modular Refactoring of convert_to_quant.py

### Session Summary
Refactored 5138-line `convert_to_quant.py` into modular subdirectory structure to enable focused editing and prevent context window exhaustion.

---

### New Module Structure

```
convert_to_quant/
├── __init__.py                  # Package entry (dynamic version from importlib.metadata)
├── convert_to_quant.py          # Slim entry point (~125 lines, re-exports for backward compat)
├── constants.py                 # All *_AVOID_KEY_NAMES, dtype constants (~150 lines)
├── converters/
│   └── learned_rounding.py      # LearnedRoundingConverter class (~1473 lines)
├── cli/
│   ├── argument_parser.py       # MultiHelpArgumentParser (~280 lines)
│   └── main.py                  # main() function (~870 lines)
├── formats/
│   ├── fp8_conversion.py        # convert_to_fp8_scaled (~640 lines)
│   ├── format_migration.py      # convert_fp8_scaled_to_comfy_quant (~346 lines)
│   ├── int8_conversion.py       # convert_int8_to_comfy_quant (~304 lines)
│   └── legacy_utils.py          # add_legacy_input_scale, cleanup_fp8_scaled (~250 lines)
├── config/
│   └── layer_config.py          # load_layer_config, get_layer_settings (~220 lines)
└── utils/
    ├── tensor_utils.py          # dict_to_tensor, normalize_tensorwise_scales (~70 lines)
    └── comfy_quant.py           # create_comfy_quant_tensor, edit_comfy_quant (~350 lines)
```

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant.py` | Reduced from 5138 to ~125 lines (re-exports only) |
| `__init__.py` | Uses `importlib.metadata.version()` with fallback |

### Verification

- All modules import successfully
- CLI `--help` output works correctly
- pyproject.toml entry point unchanged

### Git

```bash
git checkout feature/modular-refactor
git push origin feature/modular-refactor  # PR available
```

---

## 2026-01-04: Pinned Memory GPU Transfers


### Session Summary
Added pinned memory for faster CPU→GPU tensor transfers during quantization.

---

### New Files

| File | Purpose |
|------|---------|
| `pinned_transfer.py` | Utility for pinned memory transfers with fallback |

### Changes

| File | Changes |
|------|---------|
| `convert_to_quant.py` | Import + use `transfer_to_gpu_pinned()` at line 1377 |

### Technical Details

- Uses PyTorch's `tensor.pin_memory()` for page-locked memory
- `non_blocking=True` transfer with stream sync
- Falls back to regular `.to()` if pinning fails

---

## 2026-01-02: Activation Scale Calibration - Systematic Review Complete

> **STATUS: REVIEWED - Ready for Testing**

### Session Summary
Completed systematic review of `calibrate_activation_scales.py`. Fixed 5 issues identified during review:

1. **FP8 Dequantization**: Added proper handling for tensor-wise, row-wise, and block-wise weight scales
2. **Block Size Detection**: Added `infer_block_size()` to determine block size from scale shape
3. **LoRA Key Matching**: Replaced fuzzy matching with explicit key normalization (strips `model.diffusion_model.`, `lora_unet_`, etc.)
4. **CLI Seed Bug**: Added `--actcal-seed` argument (was referencing undefined `args.seed`)
5. **Metadata Sync**: `patch_model_with_scales()` now updates `.comfy_quant` metadata for comfy_quant models

---

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--actcal` | Calibrate input_scale values using simulated PTQ |
| `--actcal-samples` | Number of calibration samples (default: 64) |
| `--actcal-percentile` | Percentile for absmax (default: 99.9) |
| `--actcal-lora` | LoRA file for informed calibration (uses LoRA_A as input directions) |

### New Files

| File | Purpose |
|------|---------|
| `calibrate_activation_scales.py` | Standalone calibration module |

### Usage

```bash
# Calibrate existing FP8 model
convert_to_quant -i model_fp8.safetensors --actcal -o model_calibrated.safetensors

# With more samples for accuracy
convert_to_quant -i model.safetensors --actcal --actcal-samples 256

# LoRA-informed calibration (uses RL-extracted LoRA directions)
convert_to_quant -i model.safetensors --actcal --actcal-lora rl_lora_rank1.safetensors -o calibrated.safetensors

# Standalone script
python calibrate_activation_scales.py model.safetensors -o calibrated.safetensors
python calibrate_activation_scales.py model.safetensors --lora rl_lora.safetensors -o calibrated.safetensors
```

### Background

ComfyUI's [QUANTIZATION.md](file:///f:/convert_to_quant/ComfyUI/QUANTIZATION.md) specifies PTQ calibration for activation quantization. The default `input_scale=1.0` causes overflow/underflow in diffusion models with dynamic activation magnitudes.

---

## 2025-12-28: Legacy FP8 Cleanup Mode

### Session Summary
Added `--cleanup-fp8-scaled` mode to clean up legacy fp8_scaled models without converting to comfy_quant format.

---

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--cleanup-fp8-scaled` | Enable cleanup mode |
| `--scaled-fp8-marker {0,2}` | Set `scaled_fp8` to `empty((0))` or `empty((2))` |
| `--no-normalize-scales` | Disable normalization of 1-element scale arrays to scalars (testing) |


### Features

- Sets `scaled_fp8` marker to specified size
- Removes orphaned scale_weight/scale_input (where weight is NOT FP8)
- Optionally adds missing `.scale_input` for FP8 layers (via `--input_scale`)
- Normalizes 1-element scales to scalars
- Keeps legacy format (no comfy_quant, no metadata)

### Usage

```bash
# Basic cleanup
convert_to_quant -i model.safetensors --cleanup-fp8-scaled -o cleaned.safetensors

# With marker size 2 and adding missing scale_input
convert_to_quant -i model.safetensors --cleanup-fp8-scaled --scaled-fp8-marker 2 --input_scale -o cleaned.safetensors
```


### Verification

- Syntax check: ✅ Passed

---


## 2025-12-26: Tensorwise Scale Normalization & Metadata Generation

### Session Summary
1. Added centralized `normalize_tensorwise_scales()` helper to ensure all tensorwise scale tensors are saved as proper scalars instead of 1-element arrays. Applied to all 5 editing/conversion functions.
2. Extended `--edit-quant` to support `--save-quant-metadata` for generating `_quantization_metadata` header from existing `.comfy_quant` tensors.

---

### Changes

| Feature | Description |
|---------|-------------|
| `normalize_tensorwise_scales()` | Normalizes `numel==1` scale tensors to scalars before save |
| `--edit-quant --save-quant-metadata` | Generates metadata header from existing `.comfy_quant` tensors |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `normalize_tensorwise_scales()` (~line 179); updated 5 save points; added `save_quant_metadata` param to `edit_comfy_quant()` |

### Usage

```bash
# Fix 1-element scale arrays and generate metadata header
convert_to_quant -i model.safetensors --edit-quant --save-quant-metadata -o fixed.safetensors
```

### Verification

- Syntax check: ✅ Passed

---


## 2025-12-24: Edit-Quant Metadata Sync

### Session Summary
Extended `--edit-quant` to also modify `_quantization_metadata` header entries in sync with `.comfy_quant` tensor edits. Previously only tensor configs were updated.

---

### Changes

| Location | Before | After |
|----------|--------|-------|
| `edit_comfy_quant()` | Only edited `.comfy_quant` tensors | Edits both tensors AND `_quantization_metadata` header |
| Header preservation | Did not read/write header metadata | Preserves all existing metadata keys |
| Summary output | Single summary section | Separate reports for tensors and metadata |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Extended `edit_comfy_quant()` (lines 542-720) with metadata parsing, sync editing, and preservation; updated CLI help text |

### Usage

```bash
# Edit a model with both .comfy_quant tensors and _quantization_metadata header
convert_to_quant -i model.safetensors --edit-quant --add-keys "'full_precision_matrix_mult': true"

# Both formats are now updated in sync
```

---


## 2025-12-22: INT8 input_scale Fix & Scalar Format Consistency

### Session Summary
Fixed `convert_to_quant.py` to always add `.input_scale` for INT8 blockwise (matching reference), and changed all single-value input_scale tensors from 1-element array `[1.0]` to scalar `1.0`.

---

### Changes

| Location | Before | After |
|----------|--------|-------|
| INT8 comfy_quant | Optional (`--input_scale` flag) | Always adds `.input_scale` |
| Tensor format | `torch.tensor([1.0])` (shape `(1,)`) | `torch.tensor(1.0)` (scalar) |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | 6 locations fixed: lines 2612-2616, 2658-2660, 2962-2967, 3265-3266, 3413-3417, 3480-3484 |

### Verification

Cross-compared with reference files in `int8_blockwise_references/`:
- Triton kernels: ✅ IDENTICAL
- BlockWiseINT8Layout: ✅ IDENTICAL
- Runtime dispatch: ✅ Correct
- Converter layer saving: ✅ Fixed (was 2 discrepancies)

---


## 2025-12-22: Metadata Saving Option for Safetensors

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--save-quant-metadata` | Save quantization metadata in safetensors header (key: `_quantization_metadata`) |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added argument, `quant_metadata_layers` collection, and metadata saving logic |

### Usage

```bash
# Quantize and save metadata in header
convert_to_quant -i model.safetensors --save-quant-metadata
```

---


## 2025-12-19: ComfyQuant Layer Config Editor

### Session Summary
Added `--edit-quant` mode to edit `.comfy_quant` layer configurations without re-quantizing. Supports adding and removing keys.

---

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--edit-quant` | Enable comfy_quant editing mode |
| `--remove-keys` | Comma-separated keys to remove (e.g., `full_precision_matrix_mult,group_size`) |
| `--add-keys` | Python-like key:value pairs (e.g., `"'full_precision_matrix_mult': true"`) |
| `--quant-filter` | Regex pattern to filter which layers to edit |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `parse_add_keys_string()`, `edit_comfy_quant()` functions; CLI args; handler |

### Usage

```bash
# Remove full_precision_matrix_mult from all layers
convert_to_quant -i model.safetensors --edit-quant --remove-keys full_precision_matrix_mult

# Add full_precision_matrix_mult to all layers
convert_to_quant -i model.safetensors --edit-quant --add-keys "'full_precision_matrix_mult': true"

# Edit only specific layers
convert_to_quant -i model.safetensors --edit-quant --remove-keys group_size --quant-filter "double_blocks"
```

---


## 2025-12-19: Failed 4-bit NF4 Dequantization Fix Attempts

### Session Summary
**UNRESOLVED**: Multiple attempts to fix `RuntimeError: expected mat1 and mat2 to have the same dtype, but got: struct c10::BFloat16 != unsigned char` when loading 4-bit quantized models in ComfyUI-QuantOps. None of the attempted fixes resolved the issue.

---

### The Error
When loading NF4/FP4/AF4 4-bit quantized models, inference fails at the first linear layer with a dtype mismatch: input is BFloat16 but weight remains as uint8 (packed 4-bit values) instead of being dequantized.

Traceback path:
```
nf4_ops.py:forward() → forward_comfy_cast_weights() → F.linear()
  → comfy/quant_ops.py:__torch_dispatch__() → nf4_layout.py:nf4_linear()
  → F.linear(input_tensor, weight, bias)  ← weight is still uint8!
```

### Attempts Made (All Failed)

| # | Approach | Why It Failed |
|---|----------|---------------|
| 1 | Added `filter_keys=False` in `loader_nodes.py` to preserve quant metadata | Metadata preserved but QuantizedTensor still not dequantizing |
| 2 | Added bitsandbytes import to `nf4_ops.py`, tried `bnb_F.dequantize_4bit()` in `_dequantize_weight()` | `_dequantize_weight()` never reached - code hits QuantizedTensor path |
| 3 | Rewrote `nf4_layout.py:dequantize()` to use `bnb_F.dequantize_4bit()` | Same error - underlying issue not in dequantize method |
| 4 | Added pure PyTorch fallback (unpack nibbles, codebook lookup, blockwise scales) | Fallback never reached |

### Root Cause Analysis (Incomplete)

The model loading flow:
1. `QuantizedUNETLoader.load_unet()` loads state dict
2. `HybridNF4Ops.Linear._load_from_state_dict()` creates `QuantizedTensor` wrapper
3. `forward_comfy_cast_weights()` detects `QuantizedTensor`, calls `F.linear()`
4. PyTorch dispatches to `nf4_linear()` handler
5. Handler calls `weight.dequantize()` → **should call `NF4Layout.dequantize()`**
6. **Something fails** - weight remains uint8

Suspected issues that were NOT fully investigated:
- Is `QuantizedTensor.__torch_dispatch__` actually being called?
- Is `_get_layout_from_args()` finding the QuantizedTensor?
- Is `_LAYOUT_REGISTRY` properly populated with `"NF4Layout"` entry?
- Is there a class identity issue between `QuantizedTensor` in `comfy.quant_ops` vs the one we import?
- Is `load_state_dict()` with `strict=False` silently failing to trigger `_load_from_state_dict`?

### Files Modified (Changes Did Not Fix Issue)

| File | Changes |
|------|---------|
| `ComfyUI-QuantOps/nf4_ops.py` | Added `bitsandbytes.functional` import, rewrote `_dequantize_weight()` to try bnb_F first with PyTorch fallback |
| `ComfyUI-QuantOps/quant_layouts/nf4_layout.py` | Complete rewrite: replaced nonexistent `nf4_kernels.py` dependency with `bitsandbytes.functional.dequantize_4bit()` and pure PyTorch fallback |
| `ComfyUI-QuantOps/nodes/loader_nodes.py` | Changed `filter_keys=True` to `filter_keys=False` to preserve quant metadata |

### What Would Need Further Investigation

1. **Add extensive logging** to trace the exact path through `__torch_dispatch__`
2. **Verify layout registration** - is `"NF4Layout"` actually in `_LAYOUT_REGISTRY[torch.ops.aten.linear.default]`?
3. **Check if `_load_from_state_dict` is called** - add logging to confirm QuantizedTensor creation
4. **Inspect the QuantizedTensor at runtime** - what are its `_layout_type` and `_layout_params`?
5. **Compare with working INT8 flow** - INT8 models work, so compare the code paths

---


## 2025-12-18: Fix NF4/FP4/AF4 Layer Structure for Loader Detection

### Session Summary
Fixed 4-bit quantization output structure so loaders can automatically identify and dequantize NF4/FP4/AF4 weights.

---

### The Problem
When using 4-bit formats without `--comfy_quant`, the script incorrectly wrote `.scale_weight` (FP8/INT8 convention). The comfy_quant path also lacked critical metadata (codebook, dtype, shape) needed for dequantization.

### Changes

**Comfy path (`--comfy_quant`)**: Extended `.comfy_quant` JSON to include:
| Key | Value |
|-----|-------|
| `format` | `bnb_nf4`, `bnb_fp4`, or `bnb_af4` |
| `group_size` | Block size (64 or 128) |
| `quant_type` | `nf4`, `fp4`, or `af4` |
| `dtype` | Original dtype (e.g., `float16`) |
| `shape` | Original tensor shape |
| `quant_map` | 16-value codebook as list |

**Legacy path (no `--comfy_quant`)**: Now uses bitsandbytes-compatible structure:
| Tensor | Content |
|--------|---------|
| `.absmax` | Per-block scales (float32) |
| `.quant_map` | 16-value codebook (float32) |
| `.quant_state.bitsandbytes__<type>` | JSON with quant_type, blocksize, dtype, shape |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added codebook imports, `get_4bit_codebook()` helper, extended `create_comfy_quant_tensor()`, updated comfy path, added legacy bitsandbytes structure |
| `convert_to_quant/comfy/quant_ops.py` | Added `AF4_CODEBOOK` import, `bnb_af4` entry in `QUANT_ALGOS`, updated parameter sets |

---

## 2025-12-18: Fix --input_scale for Non-comfy Mode

### Session Summary
Fixed missing `.scale_input` handling when `--input_scale` is used without `--comfy_quant`. Also fixed T5XXL/Mistral fallback to use correct key format, and `scaled_fp8` marker to use `empty((0))` when `--input_scale` is used.

---

### The Bug

When using `--input_scale` WITHOUT `--comfy_quant`, the script only added `.scale_weight` but never `.scale_input`. The T5XXL/Mistral fallback also incorrectly used `.input_scale` (comfy format) instead of `.scale_input` (legacy format).

### The Fix

| Mode | Model Type | `--input_scale` | Result |
|------|------------|----------------|--------|
| `--comfy_quant` | Regular | Yes | `.input_scale = [1.0]` fp32 |
| `--comfy_quant` | t5xxl/mistral | Auto | `.input_scale = dequant_s` |
| Non-comfy | Regular | Yes | `.scale_input = ones_like(dequant_s)` fp32, `scaled_fp8 = empty((0))` |
| Non-comfy | t5xxl/mistral | Auto | `.scale_input = dequant_s`, `scaled_fp8 = empty((0))` |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `.scale_input` in non-comfy block (lines 2333-2338), fixed T5XXL/Mistral fallback (lines 2370-2375), updated `scaled_fp8` marker logic (line 2398) |


---

## 2025-12-18: input_scale Support & comfy_quant Fixes

### Session Summary
Extended `--input_scale` to work with all quantization methods and legacy conversions. Added automatic fix for incorrect nested `params` structure in existing comfy_quant configs.

---


### Changes

**input_scale for all formats**:
| Format | input_scale value |
|--------|-------------------|
| 4-bit (NF4/FP4/AF4) | `[1.0]` fp32 |
| INT8 | `[1.0]` fp32 |
| FP8 (normal) | `[1.0]` fp32 |
| FP8 (t5xxl/mistral) | `dequant_s` (weight_scale) |

**Legacy conversions**: `--convert-fp8-scaled` and `--convert-int8-scaled` now support `--input_scale`.

**New --legacy_input_add**: Adds `.scale_input = [1.0]` (fp32) to legacy fp8_scaled models without converting to comfy_quant format. Also converts `scaled_fp8` marker to single-element tensor.

**comfy_quant structure fix**: Added `fix_comfy_quant_params_structure()` to detect and fix nested `params` → flat structure during conversions.

**CLI priority**: Detected format now prioritized over CLI default unless explicitly specified.

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added `include_input_scale` param to conversions, `fix_comfy_quant_params_structure()`, CLI-explicit priority, AF4 format fix |

---

## 2025-12-18: Scale Shape Format Detection in Conversion Functions

### Session Summary
Added automatic format and block_size detection from scale tensor shape in `convert_fp8_scaled_to_comfy_quant()` and `convert_int8_to_comfy_quant()`.

---

### Changes

**FP8 Conversion** (`convert_fp8_scaled_to_comfy_quant`):
| Scale Shape | Detected Format |
|-------------|-----------------|
| `()` or `(1,)` | `float8_e4m3fn` |
| `(M,)` | `float8_e4m3fn_rowwise` |
| `(M//bs, N//bs)` | `float8_e4m3fn_blockwise` + inferred block_size |
| `(M, N//bs, 1)` | `float8_e4m3fn_block3d` + inferred block_size |

**INT8 Conversion** (`convert_int8_to_comfy_quant`):
| Scale Shape | Detected Format |
|-------------|-----------------|
| `(M//bs, N//bs)` | `int8_blockwise` + inferred block_size |
| `(N, K//bs)` | `int8_lodewise` + inferred block_size |

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Added format detection logic in both conversion functions, with diagnostic output |

### Verification

Tested on `Chroma-DC-2K-fp8_scaled_original_hybrid_rev2.safetensors`:
- 231 `.comfy_quant` tensors created
- All using flat structure (no `params` nesting)
- Correctly detected `float8_e4m3fn` format from scale numel=1

---

## 2025-12-18: Fix ComfyQuant Layer Configuration Structure

### Session Summary
Fixed `create_comfy_quant_tensor()` to use correct flat structure - `group_size` is now at root level, not nested in `params`.

---

### The Bug
Previous agent incorrectly nested `group_size` inside a `params` sub-object:
```python
# WRONG (was)
{"format": "int8_blockwise", "params": {"group_size": 128}}

# CORRECT (now)
{"format": "int8_blockwise", "group_size": 128}
```

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/convert_to_quant.py` | Simplified `create_comfy_quant_tensor()` - removed params nesting, cleaned up docstring |
| `AGENTS.md` | Updated "ComfyUI Metadata Format" section to show correct flat structure |

### Verification

```python
from convert_to_quant.convert_to_quant import create_comfy_quant_tensor, tensor_to_dict
tensor = create_comfy_quant_tensor("int8_blockwise", block_size=128)
result = tensor_to_dict(tensor)
# Result: {'format': 'int8_blockwise', 'group_size': 128}
# SUCCESS: No 'params' nesting
```

---

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
