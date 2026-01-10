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
- [python.instructions.md](python.instructions.md) - Coding standards and strict style guide.

## 3. System Architecture & Workflows

### System Map
- **Root**:
    - `convert_to_quant.py`: Entry shim.
    - `constants.py`: Global constants & `MODEL_FILTERS` registry.
- **CLI (`convert_to_quant/cli/`)**:
    - `main.py`: Entry point, orchestrates logging, args, and dispatch.
    - `argument_parser.py`: Dynamic arg generation from registry.
- **Formats (`convert_to_quant/formats/`)**: _High-level logic per target format_
    - `fp8_conversion.py`: FP8 scaled logic (Target: `float8_e4m3fn`).
    - `nvfp4_conversion.py`: NVFP4 logic (Target: `float4_e2m1`).
    - `int8_conversion.py`: INT8 logic (Target: `int8` + block scales).
    - `format_migration.py`: Upgrades legacy checkpoints.
- **Converters (`convert_to_quant/converters/`)**: _Quantization algorithms_
    - `learned_rounding.py`: Generic optimizer (SVD, AdamW) for FP8/INT8.
    - `learned_nvfp4.py`: Specialized NVFP4 optimizer (Block-wise).
    - `base_converter.py`: Abstract base for optimizers.
- **Utils (`convert_to_quant/utils/`)**:
    - `logging.py`: Structured logging system (`setup_logging`, `@log_debug`).
    - `comfy_quant.py`: ComfyUI metadata generation & editing.
    - `tensor_utils.py`: Tensor normalization & dict glue.
    - `memory_efficient_loader.py`: Robust `safe_open` wrapper.

### Core Workflows
1. **Initialization**: `convert_to_quant` shim -> `cli/main.py` -> `utils.logging.setup_logging`.
2. **Parsing**: `constants.MODEL_FILTERS` -> `cli/argument_parser.py` -> `args` object.
3. **Dispatch**: `main.py` selects format (e.g., `--fp8` -> `formats.fp8_conversion.convert_to_fp8_scaled`).
4. **Optimization**:
    - Format script instantiates a Converter (e.g., `converters.LearnedRoundingConverter`).
    - Converter runs optimization loop (SVD initialization -> AdamW tuning).
5. **Output**: Format script gathers quantized tensors + `utils.comfy_quant.create_comfy_quant_tensor` metadata -> Saves `.safetensors`.

### Key Relationships
- **Logging**: All modules import `convert_to_quant.utils.logging`.
- **Registry**: `fp8_conversion` and `main` consume `constants.MODEL_FILTERS`.
- **Inheritance**: `learned_rounding.py` and `learned_nvfp4.py` inherit from `base_converter.py`.

## 4. Common Commands
```bash
# Basic FP8
convert_to_quant -i model.safetensors --comfy_quant

# INT8 with heuristics
convert_to_quant -i model.safetensors --int8 --comfy_quant --heur
```

## 5. Developer Guide
**Adding New Model Support:**
*Model filters are registry-driven. No CLI code changes required.*
1. Edit `convert_to_quant/constants.py`.
2. Add entry to `MODEL_FILTERS`.
```python
MODEL_FILTERS["mymodel"] = {
    "help": "My model exclusions",
    "category": "diffusion",
    "exclude": ["layer1.norm", "layer2.bias"],
    "highprec": ["sensitive_layer"]
}
```

**ComfyUI Metadata Format:**
```python
comfy_quant = {
    "format": "float8_e4m3fn",  # or int8_blockwise
    "group_size": 64,           # Optional: for block-based
    "full_precision_matrix_mult": True
}
```

## 6. Implementation Constraints
- ✅ **Safetensors only**: Primary format for serialization.
- 📦 **Dependencies**: `safetensors`, `torch`, `tqdm`. Avoid adding heavy frameworks unless necessary.
