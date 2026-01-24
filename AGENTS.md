# AI Agent Instructions

## 1. CRITICAL: Workflow Mandates
- **Environment**: ALWAYS activate virtual environment.
- **Documentation**: Update [DEVELOPMENT.md](DEVELOPMENT.md) after EVERY task (Session Summary, Files, Usage).
- **Safety**: Test syntax (`--help`) before verifying.

## 2. Documentation Map
**Context & Status**:
- [ACTIVE.md](ACTIVE.md) - Current work, active features, and focus areas.
- [PLANNED.md](PLANNED.md) - Roadmap and future tasking.
- [DEVELOPMENT.md](DEVELOPMENT.md) - Chronological development log.

**Guides**:
- [MANUAL.md](MANUAL.md) - End-user documentation and CLI reference.
- [FORMATS.md](docs/FORMATS.md) - Technical guide for quantization formats and SVD optimization.
- [python.instructions.md](python.instructions.md) - Coding standards and strict style guide.

## 3. System Architecture & Workflows

### System Map
- **Root**:
    - `convert_to_quant.py`: Entry shim for backward compatibility.
    - `constants.py`: Global constants, dtypes, and `MODEL_FILTERS` registry.
- **CLI (`convert_to_quant/cli/`)**:
    - `main.py`: Main entry point, orchestrates logging, args, and dispatch.
    - `argument_parser.py`: Multi-section help parser and dynamic flag generation.
- **Formats (`convert_to_quant/formats/`)**: _High-level conversion workflows_
    - `fp8_conversion.py`: Unified path for FP8, INT8, and mixed formats.
    - `nvfp4_conversion.py`: Specialized NVFP4 (4-bit) conversion logic.
    - `mxfp8_conversion.py`: Specialized MXFP8 (microscaling) conversion logic.
    - `hybrid_mxfp8_conversion.py`: Logic for creating dual-scale Blackwell/Ada checkpoints.
    - `format_migration.py`: Upgrades legacy checkpoints to ComfyUI format.
- **Converters (`convert_to_quant/converters/`)**: _Quantization algorithms_
    - `learned_rounding.py`: SVD-based optimizer (AdamW/RAdam/Original) for FP8/INT8.
    - `learned_nvfp4.py`: Specialized NVFP4 optimizer with scale refinement.
    - `learned_mxfp8.py`: Specialized MXFP8 optimizer.
    - `base_converter.py`: Abstract base for all learned optimizers.
    - `nvfp4_converter.py` / `mxfp8_converter.py`: Basic round-to-nearest quantizers.
- **Utils (`convert_to_quant/utils/`)**:
    - `logging.py`: Structured logging system (`setup_logging`).
    - `comfy_quant.py`: ComfyUI metadata generation and tensor editing.
    - `tensor_utils.py`: Tensor normalization and dictionary conversion.
    - `memory_efficient_loader.py`: Unified loader for low-memory streaming.

### Core Workflows
1. **Initialization**: CLI -> `cli/main.py` -> `utils.logging.setup_logging`.
2. **Parsing**: Registry-driven parsing via `cli/argument_parser.py`.
3. **Dispatch**: `main.py` routes to specialized format scripts (e.g., `--nvfp4` -> `formats.nvfp4_conversion.convert_to_nvfp4`).
4. **Optimization**:
    - Format script instantiates a Converter (e.g., `converters.LearnedRoundingConverter`).
    - Converter runs optimization loop: SVD initialization -> Optimizer step -> Loss minimization in feature space.
5. **Output**: Gathers quantized tensors and `.comfy_quant` metadata -> Saves `.safetensors` via `safetensors.torch.save_file`.

### Key Relationships
- **Metadata**: All modern formats MUST include `.comfy_quant` tensors for loader compatibility.
- **Inheritance**: All learned converters inherit from `base_converter.py` to share SVD and LR logic.

## 4. Common Commands
```bash
# Standard FP8 with metadata
convert_to_quant -i model.safetensors --comfy_quant

# INT8 Block-wise (SVD optimized)
convert_to_quant -i model.safetensors --int8 --block_size 128 --comfy_quant

# Blackwell NVFP4 (4-bit)
convert_to_quant -i model.safetensors --nvfp4 --comfy_quant
```

## 5. Developer Guide
**Adding New Model Support:**
*Model filters are registry-driven.*
1. Edit `convert_to_quant/constants.py`.
2. Add entry to `MODEL_FILTERS` with `exclude` (skip quantization) or `highprec` (keep in FP16/BF16) patterns.

**Adding New Formats:**
1. Register identifier in `VALID_QUANT_FORMATS` in `constants.py`.
2. Implement `QuantizedLayout` and register in `QUANT_ALGOS` within `comfy/quant_ops.py`.
3. Add conversion logic in `formats/` and `converters/`.

## 6. Implementation Constraints
- ✅ **Safetensors only**: Primary format for serialization.
- 📦 **Dependencies**: `safetensors`, `torch`, `tqdm`. Avoid adding heavy frameworks unless necessary.

## 7. Testing Practices
- **Coverage**: Test all conversion paths (FP8, INT8, MXFP8, NVFP4, low-memory mode).
- **Automation**: Use `tests/test_functional.py` and format-specific test scripts for verification.
