
## 2026-02-04: Error Correction LoRA Extraction

### Session Summary
Implemented "Error Correction LoRA" extraction logic, enabling the extraction of quantization error into separate low-rank adapter layers. This allows for high-fidelity reconstruction of sensitive layers while using low-bit quantization for the primary model weights. The implementation strictly avoids any SDNQ (Stochastic Differentiable Neural Quantization) related logic.

### Changes Made
1. **CLI Enhancements**:
   - Added `LORA_ARGS` to `argument_parser.py` with a new help section (`--help-lora`).
   - Integrated LoRA arguments into `main.py`: `--extract-lora`, `--lora-rank`, `--lora-target`, `--lora-depth`, `--lora-ar-threshold`, and `--lora-save-path`.
   - Updated CLI dispatch to pass LoRA parameters to all learned conversion workflows.

2. **Core Logic**:
   - Updated `BaseLearnedConverter` in `base_converter.py` with `_should_extract_lora` (heuristic targeting) and `_extract_error_lora` (SVD-based error extraction).
   - Implemented multi-tier targeting heuristics: depth-based (block index), aspect-ratio (AR < 3.0), and keyword-based (Attention/QKV) automated selection.
   - Fixed critical SVD math error (switched from element-wise `*` to matrix multiplication `@` for rank reconstruction).
   - Modified all learned converters (`LearnedRounding`, `LearnedNVFP4`, `LearnedMXFP8`) to return an `extra_tensors` dictionary containing the extracted LoRA pieces.
   - Updated `convert` method signature across all converters to accept `key` and `depth` context.

3. **Orchestration**:
   - Updated `fp8_conversion.py`, `nvfp4_conversion.py`, and `mxfp8_conversion.py` to collect extracted LoRA tensors during iteration.
   - Implemented depth tracking via layer name regex (e.g., matching block indices).
   - Added logic to save extracted LoRA adapters to a separate `.safetensors` file (defaulting to `*_lora.safetensors`).
   - Ensured all LoRA tensors are `.contiguous()` and moved to CPU/FP32 before saving to maintain full dynamic range and precision, especially for BF16-originated models.

### Files Modified
- `convert_to_quant/cli/argument_parser.py`: Added LoRA help section and constants.
- `convert_to_quant/cli/main.py`: Added LoRA CLI flags and dispatch logic.
- `convert_to_quant/converters/base_converter.py`: Added LoRA targeting and extraction methods.
- `convert_to_quant/converters/learned_rounding.py`: Updated `convert` to return extra tensors.
- `convert_to_quant/converters/learned_nvfp4.py`: Updated `convert` to return extra tensors.
- `convert_to_quant/converters/learned_mxfp8.py`: Updated `convert` to return extra tensors.
- `convert_to_quant/formats/fp8_conversion.py`: Integrated LoRA collection and saving.
- `convert_to_quant/formats/nvfp4_conversion.py`: Integrated LoRA collection and saving.
- `convert_to_quant/formats/mxfp8_conversion.py`: Integrated LoRA collection and saving.
- `DEVELOPMENT.md`: Added this summary.

### Verification
- Created `tests/test_lora_extraction.py` covering heuristics, SVD accuracy, and output contiguity.
- Verified successful extraction for FP8 and NVFP4 formats.

### Usage
```bash
# Quantize to FP8 and extract LoRA for attention layers
convert_to_quant -i model.safetensors --extract-lora --lora-target "attn" --lora-rank 32
```

---

## 2026-01-31: Revert SDNQ Implementation

### Session Summary
Reverted the SDNQ (Stochastic Differentiable Neural Quantization) implementation due to it being non-functional and causing repository bloat. Cleaned up shared files and deleted SDNQ-specific modules and documentation to ensure a clean state for future development. Incremented version to 1.1.0.

### Changes Made
- **Reverted Shared Files**: Removed SDNQ-related logic, constants, and CLI arguments from `constants.py`, `quant_ops.py`, `argument_parser.py`, and `main.py`.
- **Deleted Modules**: Removed `sdnq_converter.py`, `sdnq_math.py`, `sdnq_conversion.py`, and the entire `docs/sdnq_implementation/` directory.
- **Cleaned Tests**: Deleted `tests/verify_sdnq_cli.py` and `tests/test_metadata.py` (which had become SDNQ-specific).
- **Version Bump**: Updated `pyproject.toml` version to `1.1.0`.

### Files Modified
- `pyproject.toml`: Version bump and script removal.
- `convert_to_quant/constants.py`: Removed SDNQ dtypes and registry entry.
- `convert_to_quant/comfy/quant_ops.py`: Removed `SDNQLayout`.
- `convert_to_quant/cli/argument_parser.py`: Removed SDNQ CLI options.
- `convert_to_quant/cli/main.py`: Removed SDNQ dispatch logic.
- `convert_to_quant/converters/__init__.py`: Removed SDNQ exports.
- `DEVELOPMENT.md`: Added this summary.

### Files Deleted
- `convert_to_quant/converters/sdnq_converter.py`
- `convert_to_quant/converters/sdnq_math.py`
- `convert_to_quant/formats/sdnq_conversion.py`
- `tests/verify_sdnq_cli.py`
- `tests/test_metadata.py`
- `REFACTOR_REPORT.md`
- `docs/sdnq_implementation/` (Recursive)


---

## 2026-01-24: Dependency Documentation Alignment

### Session Summary
Aligned `README.md` requirement summaries with ComfyUI's minimum specifications (PyTorch 2.8+, CUDA 12.8+) and the specialized Blackwell stack (PyTorch 2.10+, CUDA 13.0+).

### Files Modified
- `README.md`: Corrected PyTorch and CUDA version requirements in the summary table.

---

## 2026-01-24: Python 3.10 Upgrade

### Session Summary
Upgraded the minimum required Python version to 3.10 across the entire repository, including package metadata, CI/CD workflows, and documentation.

### Files Modified
- `pyproject.toml`: Updated `requires-python` and classifiers.
- `README.md`: Updated badges and requirements table.
- `.github/workflows/build-wheels.yml`: Updated CI build environment to Python 3.10.

---

## 2026-01-24: README Reorganization

### Session Summary
Reorganized `README.md` to prioritize essential installation and requirement information for PyPI users. Added a Requirements Summary table highlighting Blackwell-specific dependencies (CUDA 13.0+, `comfy-kitchen`).

### Files Modified
- `README.md`: Reordered sections for better accessibility and updated requirements details.

---

## 2026-01-24: CI/CD PyPI Integration

### Session Summary
Refactored the CI/CD pipeline to include a dedicated PyPI publishing job. The workflow now uses artifacts to pass build distributions between jobs, adhering to best practices found in the GitHub Marketplace.

### Files Modified
- `.github/workflows/build-wheels.yml`:
    - Added job outputs for `version` and `changed` status.
    - Standardized artifact name to `dist` for cross-job compatibility.
    - Added a separate `publish` job that uses `pypa/gh-action-pypi-publish@release/v1`.
    - Added `id-token: write` permissions for enhanced compatibility.

### Usage
1. Ensure `PYPI_ACCESS_TOKEN` is set in Repository Secrets.
2. Update `version` in `pyproject.toml` and push to `main`.
3. `build-wheels.yml` will automatically create a GitHub Release.
4. Once the release is published, `publish-pypi.yml` will build the package and upload it to PyPI.

---

## 2026-01-24: Formats Documentation

### Session Summary
Created comprehensive technical documentation for all supported quantization formats in `docs/FORMATS.md`. This guide details the internal mechanics of FP8, INT8, NVFP4, and MXFP8 formats, including scaling strategies, SVD-based learned rounding optimization, bias correction techniques, and hardware requirements, strictly based on the implementation in the codebase.

### Files Created
- `docs/FORMATS.md`: Comprehensive guide to quantization formats.

---

## 2026-01-23: Tensor-wise INT8 Quantization Support

### Session Summary
Implemented tensor-wise INT8 quantization alongside the existing block-wise implementation. The new mode uses global scaling (1 scale per tensor) and leverages `torch._scaled_mm` for efficient inference in ComfyUI. Learned rounding optimization is supported for this mode.

### Changes Made

1. **New Quantization Format**: Added `int8_tensorwise` support to the entire pipeline.
2. **Efficient Inference**: Implemented `TensorWiseINT8Layout` and corresponding operation handlers using `torch._scaled_mm` for maximum performance.
3. **CLI Enhancements**: Added support for `--scaling_mode tensor` with `--int8`, and relaxed the `block_size` requirement for this mode.

### Files Modified

| File | Changes |
|------|---------|
| `convert_to_quant/constants.py` | Registered `int8_tensorwise` format. |
| `convert_to_quant/comfy/quant_ops.py` | Added `TensorWiseINT8Layout` and operation handlers. |
| `convert_to_quant/converters/learned_rounding.py` | Added tensor-wise INT8 quantization and optimization logic. |
| `convert_to_quant/formats/fp8_conversion.py` | Updated metadata generation and conditional `input_scale` addition. |
| `convert_to_quant/cli/main.py` | Updated argument validation and filename generation for tensor-wise INT8. |

---

## 2026-01-23: Restored Original Adaptive Mode in Learned Rounding

### Session Summary
Restored the original inline tier-based adaptive LR schedule from `learned_rounding_before_refactor.py` to all learned rounding optimizers. The previous refactored abstraction (`adaptive_lr_update()`) was causing issues. Also implemented new dimension-aware `small_mult` formula.

### Changes Made

1. **Restored Original Adaptive Mode Logic**: Replaced calls to centralized `_adaptive_lr_update()` with original inline tier-based logic in all 4 `_optimize_original` methods.

2. **New Dimension-Aware `small_mult` Formula**:
   - Square: `math.gamma((M ** (1/3) / M) + 1)`
   - Tall (M > N): `math.pow(100, M / N^2)`
   - Wide (M < N): `math.pow(10, N / M^2)`

### Files Modified

| File | Changes |
|------|---------|
| `converters/learned_rounding.py` | Restored inline adaptive mode in `_optimize_original()` and `_optimize_int8_original()` |
| `converters/learned_mxfp8.py` | Restored inline adaptive mode in `_optimize_original()` |
| `converters/learned_nvfp4.py` | Restored inline adaptive mode in `_optimize_original()` |

---

## 2026-01-22: Hybrid MXFP8 Support & Edit Quant Enhancements

### Session Summary
Added Hybrid MXFP8 conversion support (`--make-hybrid-mxfp8`), which adds a tensorwise scale fallback to standard MXFP8 models. This enables compatibility with Ada Lovelace (SM 8.9) GPUs which lack native MXFP8 hardware support but can use the tensorwise scale for standard FP8 operations. Also clarified `--edit-quant` functionality for updating existing keys.

Additionally, updated `ComfyUI-QuantOps` to support loading and inference of Hybrid MXFP8 models using `comfy-kitchen` integration.

---

### New CLI Arguments

| Argument | Description |
|----------|-------------|
| `--make-hybrid-mxfp8` | Convert existing MXFP8 model to Hybrid MXFP8 (adds tensorwise fallback) |
| `--tensor-scales PATH` | Path to tensorwise FP8 model to import scales from (optional, otherwise computed) |

### Features Added

1. **Hybrid MXFP8 Conversion**:
   - Takes standard MXFP8 model (block scales only)
   - Adds `.weight_scalar` (tensorwise scale)
   - Computes scalar from block scales (max) OR imports from external model
   - Updates `.comfy_quant` format to `hybrid_mxfp8`
   - Updates `_quantization_metadata` header

2. **Edit Quant Updates**:
   - Verified `--edit-quant --add-keys` correctly updates existing keys in both layer config and header metadata.
   - Updated help text to clarify "add or update" behavior.

3. **ComfyUI-QuantOps Integration**:
   - Updated `fp8_ops.py` to handle `HybridMXFP8Layout`, reading `.weight_scalar` from state dict.
   - Updated `loader_nodes.py` to add `hybrid_mxfp8` to format options and enable `HybridFP8Ops`.

### Files Modified

| File | Changes |
|------|---------|
| `formats/hybrid_mxfp8_conversion.py` | **NEW** - Implementation of conversion logic and scale computation |
| `cli/main.py` | Added CLI args and dispatch logic for hybrid conversion |
| `cli/argument_parser.py` | Added new args to help sections |
| `constants.py` | Added `hybrid_mxfp8` to `VALID_QUANT_FORMATS` |
| `ComfyUI-QuantOps/fp8_ops.py` | Added Hybrid MXFP8 layout handling and scalar loading |
| `ComfyUI-QuantOps/nodes/loader_nodes.py` | Added format option to loaders |

### Usage

```bash
# Convert MXFP8 to Hybrid (compute scales)
convert_to_quant -i model_mxfp8.safetensors --make-hybrid-mxfp8 -o model_hybrid.safetensors

# Convert using scales from another model
convert_to_quant -i model_mxfp8.safetensors --make-hybrid-mxfp8 --tensor-scales model_fp8.safetensors
```

### Verification

- Created `tests/test_hybrid_mxfp8.py` covering computation, external scales, and metadata updates.
- All tests passed.
