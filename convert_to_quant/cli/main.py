"""
CLI main function for convert_to_quant.

Entry point that handles argument parsing and dispatches to appropriate conversion functions.
"""
import argparse
import os
import sys
import torch
from safetensors.torch import load_file, save_file

from .argument_parser import (
    MultiHelpArgumentParser,
    EXPERIMENTAL_ARGS,
    FILTER_ARGS,
    ADVANCED_ARGS,
    LEARNED_ROUNDING_ARGS,
    MODES_ARGS,
)
from ..constants import (
    NORMALIZE_SCALES_ENABLED,
    TARGET_FP8_DTYPE,
    MODEL_FILTERS,
)
from ..config.layer_config import load_layer_config, generate_config_template
from ..formats.fp8_conversion import convert_to_fp8_scaled
from ..formats.format_migration import convert_fp8_scaled_to_comfy_quant
from ..formats.int8_conversion import convert_int8_to_comfy_quant
from ..formats.legacy_utils import add_legacy_input_scale, cleanup_fp8_scaled
from ..formats.nvfp4_conversion import convert_to_nvfp4
from ..formats.mxfp8_conversion import convert_to_mxfp8
from ..formats.hybrid_mxfp8_conversion import convert_to_hybrid_mxfp8
from ..formats.sdnq_conversion import convert_to_sdnq
from ..utils.comfy_quant import edit_comfy_quant
from ..pinned_transfer import set_verbose as set_pinned_verbose
import json
from safetensors import safe_open

def load_input_scales(path: str) -> dict:
    """Load input scales from JSON or safetensors file.

    Args:
        path: Path to JSON file or safetensors model with .input_scale tensors

    Returns:
        Dict mapping layer base names to input_scale values (float)
    """
    if path.endswith('.json'):
        with open(path) as f:
            return json.load(f)
    elif path.endswith('.safetensors'):
        scales = {}
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if key.endswith('.input_scale'):
                    base = key.rsplit('.input_scale', 1)[0]
                    scales[base] = f.get_tensor(key).item()
        return scales
    else:
        raise ValueError(f"Unsupported input scales format: {path}. Use .json or .safetensors")

def extract_filter_flags(args) -> dict:
    """Extract model filter flags from parsed args with validation.

    Validates that every filter in MODEL_FILTERS has a corresponding
    argparse attribute. Fails fast if argparse is missing a filter,
    which indicates a bug (filter added to constants.py but not argument_parser.py).

    Args:
        args: Parsed argparse namespace

    Returns:
        Dict mapping filter names to bool values, e.g. {"radiance": True, "t5xxl": False}
    """
    flags = {}
    for name in MODEL_FILTERS.keys():
        if not hasattr(args, name):
            raise RuntimeError(
                f"BUG: Filter '{name}' in MODEL_FILTERS but not in argparse. "
                f"Add --{name} to argument_parser.py"
            )
        if getattr(args, name):
            flags[name] = True
    return flags
from ..utils.logging import setup_logging, info, minimal, warning

def main():
    # Parse args to get verbosity level
    parser = MultiHelpArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Convert safetensors weights to Scaled FP8 format.\n\n"
        "Default behavior: FP8 quantization with per-tensor scaling.\n"
        "For INT8 and other experimental options, see --help-experimental.\n"
        "For model-specific layer exclusions, see --help-filters.\n"
        "For advanced LR tuning and early stopping, see --help-advanced.\n"
        "For conversion and utility modes, see --help-modes.",
        experimental_args=EXPERIMENTAL_ARGS,
        filter_args=FILTER_ARGS,
        advanced_args=ADVANCED_ARGS,
        learned_rounding_args=LEARNED_ROUNDING_ARGS,
        modes_args=MODES_ARGS,
    )


    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Input safetensors file path."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output safetensors file path. Auto-generated if not provided.",
    )
    parser.add_argument(
        "--comfy_quant", action="store_true", help="Use Comfy quantization method."
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Use INT8 block-wise quantization instead of FP8.",
    )
    parser.add_argument(
        "--nvfp4",
        action="store_true",
        help="Use NVFP4 (FP4 E2M1) block quantization. Requires Blackwell GPU (SM >= 10.0/12.0) for inference.",
    )
    parser.add_argument(
        "--mxfp8",
        action="store_true",
        help="Use MXFP8 (Microscaling FP8) block quantization. Requires Blackwell GPU (SM >= 10.0) for inference.",
    )
    parser.add_argument(
        "--sdnq",
        action="store_true",
        help="Use SDNQ (Stochastic Differentiable Neural Quantization). Supports arbitrary bit-widths and SVD correction.",
    )
    parser.add_argument(
        "--make-hybrid-mxfp8",
        action="store_true",
        dest="make_hybrid_mxfp8",
        help="Convert an existing MXFP8 model to Hybrid MXFP8 (adds tensorwise fallback for Ada GPUs).",
    )
    parser.add_argument(
        "--tensor-scales",
        type=str,
        default=None,
        dest="tensor_scales_path",
        help="Path to tensorwise FP8 model to steal scales from (for --make-hybrid-mxfp8).",
    )
    parser.add_argument(
        "--fallback",
        type=str,
        default=None,
        choices=["fp8", "int8", "mxfp8", "nvfp4"],
        help="Fallback quantization type for excluded layers (instead of keeping original precision).",
    )
    parser.add_argument(
        "--custom-layers",
        type=str,
        default=None,
        dest="custom_layers",
        help="Regex pattern for layers to quantize with custom type. Takes priority over exclusions.",
    )
    parser.add_argument(
        "--exclude-layers",
        type=str,
        default=None,
        dest="exclude_layers",
        help="Regex pattern for layers to exclude from quantization (keep original precision or use fallback).",
    )
    parser.add_argument(
        "--custom-type",
        type=str,
        default=None,
        dest="custom_type",
        choices=["fp8", "int8", "mxfp8", "nvfp4"],
        help="Quantization type for custom layer matches.",
    )
    # Custom-type parameter overrides
    parser.add_argument(
        "--custom-block-size",
        type=int,
        default=None,
        dest="custom_block_size",
        help="Block size for custom-type layers (default: inherit --block_size)",
    )
    parser.add_argument(
        "--custom-scaling-mode",
        type=str,
        default=None,
        dest="custom_scaling_mode",
        choices=["tensor", "row", "block", "block3d", "block2d"],
        help="FP8 scaling mode for custom-type layers (default: inherit --scaling_mode). 'block2d' is deprecated alias for 'block'.",
    )
    parser.add_argument(
        "--custom-simple",
        action="store_true",
        dest="custom_simple",
        help="Use simple quantization for custom-type layers",
    )
    parser.add_argument(
        "--custom-heur",
        action="store_true",
        dest="custom_heur",
        help="Apply performance heuristics to custom-type layers",
    )
    # Fallback-type parameter overrides
    parser.add_argument(
        "--fallback-block-size",
        type=int,
        default=None,
        dest="fallback_block_size",
        help="Block size for fallback-type layers (default: inherit --block_size)",
    )
    parser.add_argument(
        "--fallback-simple",
        action="store_true",
        dest="fallback_simple",
        help="Use simple quantization for fallback-type layers",
    )
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Skip SVD optimization, use simple quantization.",
    )
    parser.add_argument(
        "--full_precision_matrix_mult",
        action="store_true",
        help="Add full_precision_matrix_mult=True to .comfy_quant metadata.",
    )
    parser.add_argument(
        "--heur",
        action="store_true",
        help="Skip layers with poor quantization characteristics (aspect ratio, size).",
    )
    parser.add_argument(
        "--input_scale",
        action="store_true",
        help="Include input_scale tensor (fp32, 1.0) for quantized layers. Works with oconvert-fp8-scaled and --convert-int8-scaled. Always enabled for T5XXL.",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="NORMAL",
        choices=["DEBUG", "VERBOSE", "NORMAL", "MINIMAL"],
        help="Set verbosity: NORMAL (default), VERBOSE (increased), MINIMAL (reduced), DEBUG (all).",
    )
    # SDNQ-specific arguments
    parser.add_argument(
        "--sdnq-dtype",
        type=str,
        default="int8",
        dest="sdnq_dtype",
        help="[SDNQ] Target weight dtype (e.g., int8, int4, fp8, float8_e4m3fn). (default: int8)",
    )
    parser.add_argument(
        "--sdnq-group-size",
        type=int,
        default=0,
        dest="sdnq_group_size",
        help="[SDNQ] Group size for quantization (0 for auto). (default: 0)",
    )
    parser.add_argument(
        "--sdnq-use-svd",
        action="store_true",
        dest="sdnq_use_svd",
        help="[SDNQ] Enable SVD-based low-rank correction.",
    )
    parser.add_argument(
        "--sdnq-svd-rank",
        type=int,
        default=32,
        dest="sdnq_svd_rank",
        help="[SDNQ] SVD rank for correction. (default: 32)",
    )
    parser.add_argument(
        "--sdnq-svd-steps",
        type=int,
        default=8,
        dest="sdnq_svd_steps",
        help="[SDNQ] SVD iterations. (default: 8)",
    )
    parser.add_argument(
        "--sdnq-stochastic",
        action="store_true",
        dest="sdnq_stochastic",
        help="[SDNQ] Enable stochastic rounding.",
    )

    # Model filter flags - generated from MODEL_FILTERS registry

    for filter_name, filter_cfg in MODEL_FILTERS.items():
        parser.add_argument(
            f"--{filter_name}",
            action="store_true",
            help=filter_cfg.get("help", f"Apply {filter_name} model exclusions"),
        )
    parser.add_argument(
        "--full_matrix",
        action="store_true",
        help="If should use torch.linalg.svd with full matices instead of the torch.svd_lowrank.",
    )
    parser.add_argument(
        "--scaling_mode",
        type=str,
        default="tensor",
        choices=["tensor", "row", "block", "block3d", "block2d"],
        help="FP8 scaling mode: 'tensor' (1 global scale), 'row' (per-row scale), 'block' (2D tiles like INT8), 'block3d' (per-row-group 3D, legacy). 'block2d' is deprecated alias for 'block'.",
    )

    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Block size for block-wise quantization (REQUIRED for INT8). Common values: 64, 128.",
    )
    parser.add_argument(
        "--calib_samples",
        type=int,
        default=6144,
        help="Number of random samples for bias correction.",
    )
    parser.add_argument(
        "--manual_seed",
        type=int,
        default=-1,
        help="Set a manual seed for reproducibility. Use -1 for random.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="original",
        choices=["original", "adamw", "radam"],
        help="Optimization algorithm.",
    )
    parser.add_argument(
        "--num_iter",
        type=int,
        default=1000,
        help="Total optimization iterations per tensor.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=8.077300000003e-3,
        help="[AdamW/RAdam/Original] Initial learning rate.",
    )
    parser.add_argument(
        "--lr_schedule",
        type=str,
        default="adaptive",
        choices=["adaptive", "exponential", "plateau"],
        help="LR schedule for 'original' optimizer: 'adaptive' (default custom), 'exponential' (gamma decay), 'plateau' (reduce on stall)",
    )
    parser.add_argument(
        "--lr_gamma",
        type=float,
        default=0.99,
        help="[exponential] Decay factor per step (default: 0.99)",
    )
    parser.add_argument(
        "--lr_patience", type=int, default=9, help="[plateau] Steps before decay"
    )
    parser.add_argument(
        "--lr_factor", type=float, default=0.95, help="[plateau] LR reduction factor"
    )
    parser.add_argument(
        "--lr_min", type=float, default=1e-10, help="[plateau] Minimum LR bound"
    )
    parser.add_argument(
        "--lr_cooldown",
        type=int,
        default=6,
        help="[plateau] Steps to wait after reduction",
    )
    parser.add_argument(
        "--lr_threshold",
        type=float,
        default=0.0,
        help="[plateau] Min improvement to reset patience",
    )
    parser.add_argument(
        "--lr_adaptive_mode",
        type=str,
        default="simple-reset",
        choices=["simple-reset", "no-reset"],
        help="[adaptive] Counter reset behavior (see MANUAL.md)",
    )
    # Advanced LR tuning (--help-advanced)
    parser.add_argument(
        "--lr-shape-influence",
        type=float,
        default=1.0,
        dest="lr_shape_influence",
        help="[plateau] Scale factor based on tensor aspect ratio. 0.0=disabled, 1.0=full effect. Elongated tensors get more aggressive decay. (default: 1.0)",
    )
    parser.add_argument(
        "--lr-threshold-mode",
        type=str,
        default="rel",
        choices=["rel", "abs"],
        dest="lr_threshold_mode",
        help="[plateau] How to interpret --lr_threshold: 'rel' (relative to best loss) or 'abs' (absolute). (default: rel)",
    )
    # Early stopping thresholds (--help-advanced)
    parser.add_argument(
        "--early-stop-loss",
        type=float,
        default=1e-8,
        dest="early_stop_loss",
        help="Early stop when loss drops below this value. (default: 1e-8)",
    )
    parser.add_argument(
        "--early-stop-lr",
        type=float,
        default=1e-10,
        dest="early_stop_lr",
        help="Early stop when LR drops below this value. (default: 1e-10)",
    )
    parser.add_argument(
        "--early-stop-stall",
        type=int,
        default=1000,
        dest="early_stop_stall",
        help="Early stop when worse_loss_counter exceeds this. (default: 1000)",
    )
    # NVFP4 scale optimization (--help-advanced)
    parser.add_argument(
        "--scale-refinement",
        type=int,
        default=1,
        dest="scale_refinement_rounds",
        help="[NVFP4] Number of scale refinement rounds for 'iterative' mode (default: 1)",
    )
    parser.add_argument(
        "--scale-optimization",
        type=str,
        default="fixed",
        dest="scale_optimization",
        choices=["fixed", "iterative", "joint"],
        help="[NVFP4] Scale optimization mode: 'fixed' (default, scales computed once), "
             "'iterative' (scales recomputed periodically), 'joint' (STE-based joint optimization)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.2,
        help="Proportion of principal components (SVD) to use.",
    )
    parser.add_argument(
        "--min_k", type=int, default=64, help="Minimum number of principal components."
    )
    parser.add_argument(
        "--max_k",
        type=int,
        default=1024,
        help="Maximum number of principal components.",
    )

    # FP8 scaled to comfy_quant conversion mode
    parser.add_argument(
        "--convert-fp8-scaled",
        action="store_true",
        dest="convert_fp8_scaled",
        help="Convert fp8_scaled model to comfy_quant format (no quantization, just format conversion)",
    )
    parser.add_argument(
        "--hp-filter",
        type=str,
        default=None,
        dest="hp_filter",
        help="Regex pattern for high-precision layers to validate (error if they have FP8 weights)",
    )
    parser.add_argument(
        "--full-precision-mm",
        action="store_true",
        dest="full_precision_mm",
        help="Set full_precision_matrix_mult=True in .comfy_quant metadata (for --convert-fp8-scaled)",
    )

    # INT8 to comfy_quant conversion mode
    parser.add_argument(
        "--convert-int8-scaled",
        action="store_true",
        dest="convert_int8_scaled",
        help="Convert legacy INT8 model (.scale_weight) to comfy_quant format (.weight_scale + metadata)",
    )

    # Legacy input scale addition mode
    parser.add_argument(
        "--legacy_input_add",
        action="store_true",
        help="Add .scale_input tensors to legacy fp8_scaled models (keeps legacy format, adds missing input scales)",
    )

    # Legacy FP8 cleanup mode
    parser.add_argument(
        "--cleanup-fp8-scaled",
        action="store_true",
        dest="cleanup_fp8_scaled",
        help="Clean up legacy fp8_scaled model: remove orphaned scales, set scaled_fp8 marker, normalize scales",
    )
    parser.add_argument(
        "--scaled-fp8-marker",
        type=int,
        default=0,
        choices=[0, 2],
        dest="scaled_fp8_marker",
        help="Size for scaled_fp8 marker tensor: 0=empty((0)), 2=empty((2)). (default: 0)",
    )

    # Activation scale calibration mode
    parser.add_argument(
        "--actcal",
        action="store_true",
        dest="actcal",
        help="Calibrate input_scale values using simulated PTQ. Patches existing FP8 model with computed scales.",
    )
    parser.add_argument(
        "--actcal-samples",
        type=int,
        default=64,
        dest="actcal_samples",
        help="Number of calibration samples for --actcal (default: 64)",
    )
    parser.add_argument(
        "--actcal-percentile",
        type=float,
        default=99.9,
        dest="actcal_percentile",
        help="Percentile for absmax in calibration (default: 99.9, use 100 for true max)",
    )
    parser.add_argument(
        "--actcal-lora",
        dest="actcal_lora",
        help="LoRA file for informed calibration (uses LoRA_A as input directions)",
    )
    parser.add_argument(
        "--actcal-seed",
        type=int,
        default=42,
        dest="actcal_seed",
        help="Random seed for calibration (default: 42). Use for reproducible results.",
    )
    parser.add_argument(
        "--actcal-device",
        type=str,
        default=None,
        dest="actcal_device",
        help="Device for calibration: 'cpu', 'cuda', 'cuda:0', etc. (default: auto-detect CUDA)",
    )

    # Metadata saving option
    parser.add_argument(
        "--save-quant-metadata",
        action="store_true",
        dest="save_quant_metadata",
        help="Save quantization metadata in safetensors header (under _quantization_metadata key)",
    )

    # Scale normalization toggle (for testing)
    parser.add_argument(
        "--no-normalize-scales",
        action="store_true",
        dest="no_normalize_scales",
        help="Disable normalization of 1-element scale arrays to scalars (for testing/compatibility)",
    )

    # NVFP4 input scales (from calibration or another NVFP4 model)
    parser.add_argument(
        "--input-scales",
        type=str,
        default=None,
        dest="input_scales_path",
        help="Path to input scales file (.json or .safetensors). "
             "JSON format: {'layer.name': 0.015, ...}. "
             "Safetensors: extracts .input_scale tensors from an existing NVFP4 model.",
    )

    # ComfyQuant layer config editing mode
    parser.add_argument(
        "--edit-quant",
        action="store_true",
        dest="edit_quant",
        help="Edit .comfy_quant tensors and _quantization_metadata header (add/remove keys)",
    )
    parser.add_argument(
        "--remove-keys",
        type=str,
        default=None,
        dest="remove_keys",
        help="Comma-separated keys to remove (e.g., 'full_precision_matrix_mult,group_size')",
    )
    parser.add_argument(
        "--add-keys",
        type=str,
        default=None,
        dest="add_keys",
        help="Python-like key:value pairs to add or update (e.g., \"'full_precision_matrix_mult': true, 'group_size': 64\")",
    )
    parser.add_argument(
        "--quant-filter",
        type=str,
        default=None,
        dest="quant_filter",
        help="Regex pattern to filter which layers to edit (default: all layers)",
    )

    # Per-layer quantization config (JSON file)
    parser.add_argument(
        "--layer-config",
        type=str,
        default=None,
        dest="layer_config",
        help="""Path to JSON file with per-layer quantization settings (regex patterns).
Example config:
{
  "_default": {"format": "float8_e4m3fn"},
  "attn": {"format": "float8_e4m3fn", "full_precision_matrix_mult": true},
  "\\\\.0\\\\.img_mod": {"skip": true}
}
By default, patterns use re.search (substring match). Use --fullmatch for full string matching.
In JSON, backslashes must be doubled (\\\\. for literal dot). See DEVELOPMENT.md for details.""",
    )
    parser.add_argument(
        "--fullmatch",
        action="store_true",
        dest="layer_config_fullmatch",
        help="Use re.fullmatch instead of re.search for --layer-config patterns. "
        "With fullmatch, patterns must match the entire layer name (use .* for wildcards).",
    )

    # Dry run / template generation
    parser.add_argument(
        "--dry-run",
        type=str,
        nargs="?",
        const="analyze",
        default=None,
        dest="dry_run",
        choices=["analyze", "create-template"],
        help="Dry run mode: 'analyze' shows what would be processed, 'create-template' generates config template",
    )

    # Verbose output for pinned memory transfers
    parser.add_argument(
        "--verbose-pinned",
        action="store_true",
        dest="verbose_pinned",
        help="Print per-tensor pinned memory transfer details",
    )

    # Memory-efficient loading mode
    parser.add_argument(
        "--low-memory",
        action="store_true",
        dest="low_memory",
        help="Use streaming tensor loading to reduce RAM usage (recommended for models >50%% of available RAM)",
    )

    args = parser.parse_args()

    # Initialize logging framework with user's verbosity preference
    setup_logging(args.verbose)

    # Set global scale normalization flag from CLI
    global NORMALIZE_SCALES_ENABLED
    NORMALIZE_SCALES_ENABLED = not args.no_normalize_scales

    # Set pinned memory verbosity
    set_pinned_verbose(args.verbose_pinned)

    # Handle dry-run create-template mode (separate workflow)
    if args.dry_run == "create-template":
        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        template_path = os.path.splitext(args.input)[0] + "_layer_config_template.json"
        generate_config_template(
            args.input, template_path, block_size=args.block_size or 128
        )
        return

    # Handle fp8_scaled conversion mode first (separate workflow)
    if args.convert_fp8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_fp8mixed.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        convert_fp8_scaled_to_comfy_quant(
            args.input,
            args.output,
            hp_filter=args.hp_filter,
            full_precision_mm=args.full_precision_mm,
            include_input_scale=args.input_scale,
        )
        return

    # Handle int8 to comfy_quant conversion mode (separate workflow)
    if args.convert_int8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_int8_comfy.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        # Use block_size from args or default to 128
        int8_block_size = args.block_size if args.block_size else 128

        convert_int8_to_comfy_quant(
            args.input,
            args.output,
            block_size=int8_block_size,
            include_input_scale=args.input_scale,
            save_quant_metadata=args.save_quant_metadata,
        )
        return

    # Handle NVFP4 quantization mode (separate workflow OR unified if mixing formats)
    if args.nvfp4:
        # Check if we need mixed format support
        needs_mixing = args.custom_type or args.fallback

        if needs_mixing:
            # Route through unified path for mixed format support
            print("NVFP4 with custom/fallback: using unified quantization path")
            if not args.output:
                base = os.path.splitext(args.input)[0]
                args.output = f"{base}_nvfp4_mixed.safetensors"
            # Fall through to convert_to_fp8_scaled with target_format="nvfp4"
            args.int8 = False  # Ensure not INT8
            # Continue to main FP8 path below with nvfp4 as target_format
        else:
            # Use dedicated NVFP4 path for simple cases
            if not args.output:
                base = os.path.splitext(args.input)[0]
                # Build filename: {simple_|learned_}nvfp4[mixed]
                prefix = "simple_" if args.simple else "learned_"
                # Check for filters or custom-layers
                filter_flags = extract_filter_flags(args)
                has_filters = any(filter_flags.values())
                has_custom = bool(args.custom_layers)
                mixed_suffix = "mixed" if (has_filters or has_custom) else ""
                args.output = f"{base}_{prefix}nvfp4{mixed_suffix}.safetensors"

            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                return

            if os.path.abspath(args.input) == os.path.abspath(args.output):
                print("Error: Output file cannot be same as input.")
                return

            # Compute seed early (same logic as FP8)
            seed = (
                int(torch.randint(0, 2**32 - 1, ()).item())
                if args.manual_seed == -1
                else args.manual_seed
            )
            print(f"Using seed: {seed}")

            # Extract filter flags with validation
            filter_flags = extract_filter_flags(args)

            # Load input scales if provided
            input_scales = None
            if args.input_scales_path:
                if not os.path.exists(args.input_scales_path):
                    print(f"Error: Input scales file not found: {args.input_scales_path}")
                    return
                input_scales = load_input_scales(args.input_scales_path)
                print(f"Loaded {len(input_scales)} input scales from: {args.input_scales_path}")

            # Call convert_to_nvfp4 with explicit args (no **kwargs footgun)
            convert_to_nvfp4(
                args.input,
                args.output,
                # Filter flags
                filter_flags=filter_flags,
                exclude_layers=args.exclude_layers,
                # Quantization options
                simple=args.simple,
                num_iter=args.num_iter,
                heur=args.heur,
                calib_samples=args.calib_samples,
                seed=seed,
                # Optimizer/LR options
                optimizer=args.optimizer,
                lr=args.lr,
                lr_schedule=args.lr_schedule,
                top_p=args.top_p,
                min_k=args.min_k,
                max_k=args.max_k,
                full_matrix=args.full_matrix,
                # LR schedule tuning
                lr_gamma=args.lr_gamma,
                lr_patience=args.lr_patience,
                lr_factor=args.lr_factor,
                lr_min=args.lr_min,
                lr_cooldown=args.lr_cooldown,
                lr_threshold=args.lr_threshold,
                lr_adaptive_mode=args.lr_adaptive_mode,
                lr_shape_influence=args.lr_shape_influence,
                lr_threshold_mode=args.lr_threshold_mode,
                # Early stopping
                early_stop_loss=args.early_stop_loss,
                early_stop_lr=args.early_stop_lr,
                early_stop_stall=args.early_stop_stall,
                # Scale optimization
                scale_refinement_rounds=args.scale_refinement_rounds,
                scale_optimization=args.scale_optimization,
                # Input scales
                input_scales=input_scales,
                # Memory mode
                low_memory=args.low_memory,
            )
            return

    # Handle Hybrid MXFP8 conversion mode (separate workflow)
    if args.make_hybrid_mxfp8:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_hybrid.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        convert_to_hybrid_mxfp8(
            args.input,
            args.output,
            tensor_scales_path=args.tensor_scales_path,
        )
        return

    # Handle MXFP8 quantization mode (separate workflow OR unified if mixing formats)
    if args.mxfp8:
        # Check if we need mixed format support
        needs_mixing = args.custom_type or args.fallback

        if needs_mixing:
            # Route through unified path for mixed format support
            print("MXFP8 with custom/fallback: using unified quantization path")
            if not args.output:
                base = os.path.splitext(args.input)[0]
                args.output = f"{base}_mxfp8_mixed.safetensors"
            # Fall through to convert_to_fp8_scaled with target_format="mxfp8"
            args.int8 = False  # Ensure not INT8
            # Continue to main FP8 path below with mxfp8 as target_format
        else:
            # Use dedicated MXFP8 path for simple cases
            if not args.output:
                base = os.path.splitext(args.input)[0]
                # Build filename: {simple_|learned_}mxfp8[mixed]
                prefix = "simple_" if args.simple else "learned_"
                # Check for filters or custom-layers
                filter_flags = extract_filter_flags(args)
                has_filters = any(filter_flags.values())
                has_custom = bool(args.custom_layers)
                mixed_suffix = "mixed" if (has_filters or has_custom) else ""
                args.output = f"{base}_{prefix}mxfp8{mixed_suffix}.safetensors"

            if not os.path.exists(args.input):
                print(f"Error: Input file not found: {args.input}")
                return

            if os.path.abspath(args.input) == os.path.abspath(args.output):
                print("Error: Output file cannot be same as input.")
                return

            # Compute seed early (same logic as FP8/NVFP4)
            seed = (
                int(torch.randint(0, 2**32 - 1, ()).item())
                if args.manual_seed == -1
                else args.manual_seed
            )
            print(f"Using seed: {seed}")

            # Extract filter flags with validation
            filter_flags = extract_filter_flags(args)

            # Call convert_to_mxfp8 with explicit args
            convert_to_mxfp8(
                args.input,
                args.output,
                # Filter flags
                filter_flags=filter_flags,
                exclude_layers=args.exclude_layers,
                # Quantization options
                simple=args.simple,
                num_iter=args.num_iter,
                heur=args.heur,
                calib_samples=args.calib_samples,
                seed=seed,
                # Optimizer/LR options
                optimizer=args.optimizer,
                lr=args.lr,
                lr_schedule=args.lr_schedule,
                top_p=args.top_p,
                min_k=args.min_k,
                max_k=args.max_k,
                full_matrix=args.full_matrix,
                # LR schedule tuning
                lr_gamma=args.lr_gamma,
                lr_patience=args.lr_patience,
                lr_factor=args.lr_factor,
                lr_min=args.lr_min,
                lr_cooldown=args.lr_cooldown,
                lr_threshold=args.lr_threshold,
                lr_adaptive_mode=args.lr_adaptive_mode,
                lr_shape_influence=args.lr_shape_influence,
                lr_threshold_mode=args.lr_threshold_mode,
                # Early stopping
                early_stop_loss=args.early_stop_loss,
                early_stop_lr=args.early_stop_lr,
                early_stop_stall=args.early_stop_stall,
                # Scale optimization
                scale_refinement_rounds=args.scale_refinement_rounds,
                scale_optimization=args.scale_optimization,
                # Memory mode
                low_memory=args.low_memory,
            )
            return

    # Handle SDNQ quantization mode (separate workflow)
    if args.sdnq:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            # Build filename: sdnq_{dtype}[_svd]
            svd_suffix = "_svd" if args.sdnq_use_svd else ""
            args.output = f"{base}_sdnq_{args.sdnq_dtype}{svd_suffix}.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        # Extract filter flags with validation
        filter_flags = extract_filter_flags(args)

        convert_to_sdnq(
            args.input,
            args.output,
            weights_dtype=args.sdnq_dtype,
            group_size=args.sdnq_group_size,
            use_svd=args.sdnq_use_svd,
            svd_rank=args.sdnq_svd_rank,
            svd_steps=args.sdnq_svd_steps,
            use_quantized_matmul=args.full_precision_matrix_mult is False, # Logic flip? No, sdnq_math uses use_quantized_matmul
            use_stochastic_rounding=args.sdnq_stochastic,
            active_filters=filter_flags,
            save_comfy_quant=args.comfy_quant,
        )
        return

    # Handle legacy input scale addition mode (separate workflow)
    if args.legacy_input_add:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_with_input_scale.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        add_legacy_input_scale(args.input, args.output)
        return

    # Handle legacy FP8 cleanup mode (separate workflow)
    if args.cleanup_fp8_scaled:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_cleaned.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        cleanup_fp8_scaled(
            args.input,
            args.output,
            marker_size=args.scaled_fp8_marker,
            add_scale_input=args.input_scale,
        )
        return

    # Handle activation scale calibration mode (separate workflow)
    if args.actcal:
        try:
            from .calibrate_activation_scales import calibrate_model, patch_model_with_scales, load_lora_tensors
        except ImportError:
            from calibrate_activation_scales import calibrate_model, patch_model_with_scales, load_lora_tensors

        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_calibrated.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        print(f"Loading model: {args.input}")
        tensors = load_file(args.input)
        print(f"  Total tensors: {len(tensors)}")

        # Load LoRA if specified
        lora_tensors = None
        if args.actcal_lora:
            if not os.path.exists(args.actcal_lora):
                print(f"Error: LoRA file not found: {args.actcal_lora}")
                return
            print(f"\nLoading LoRA: {args.actcal_lora}")
            lora_tensors = load_lora_tensors(args.actcal_lora)
            print(f"  LoRA layers found: {len(lora_tensors)}")

        mode = "LoRA-informed" if lora_tensors else "random"
        print(f"\nCalibrating input_scale using {mode} PTQ ({args.actcal_samples} samples)...")
        scales = calibrate_model(
            tensors,
            calib_samples=args.actcal_samples,
            seed=args.actcal_seed,
            percentile=args.actcal_percentile,
            verbose=True,
            lora_tensors=lora_tensors,
            device=args.actcal_device,
        )
        print(f"\nCalibrated {len(scales)} layers")

        print(f"\nPatching model with calibrated scales...")
        patched = patch_model_with_scales(tensors, scales)

        print(f"Saving to: {args.output}")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_file(patched, args.output)
        print("Done!")
        return

    # Handle comfy_quant editing mode (separate workflow)
    if args.edit_quant:
        if not args.output:
            base = os.path.splitext(args.input)[0]
            args.output = f"{base}_edited.safetensors"

        if not os.path.exists(args.input):
            print(f"Error: Input file not found: {args.input}")
            return

        if os.path.abspath(args.input) == os.path.abspath(args.output):
            print("Error: Output file cannot be same as input.")
            return

        if not args.remove_keys and not args.add_keys and not args.save_quant_metadata:
            print(
                "Error: --edit-quant requires at least one of --remove-keys, --add-keys, or --save-quant-metadata"
            )
            return

        # Parse remove_keys from comma-separated string
        remove_keys_list = None
        if args.remove_keys:
            remove_keys_list = [
                k.strip() for k in args.remove_keys.split(",") if k.strip()
            ]

        edit_comfy_quant(
            args.input,
            args.output,
            remove_keys=remove_keys_list,
            add_keys_str=args.add_keys,
            layer_filter=args.quant_filter,
            save_quant_metadata=args.save_quant_metadata,
        )
        return

    # Determine which formats require block_size
    primary_needs_block_size = args.int8 and args.scaling_mode != "tensor"
    custom_needs_block_size = args.custom_type == "int8" and args.custom_scaling_mode != "tensor"
    fallback_needs_block_size = args.fallback == "int8"

    # Validate block_size for primary format
    if primary_needs_block_size and args.block_size is None:
        print("Error: --block_size is required when using INT8 quantization.")
        print("       Example: --block_size 128")
        sys.exit(1)

    # Validate custom-block-size for custom format
    if args.custom_type and custom_needs_block_size and args.custom_block_size is None:
        print(
            f"Error: --custom-block-size is required when using --custom-type {args.custom_type}."
        )
        print("       Example: --custom-block-size 128")
        sys.exit(1)

    # Validate fallback-block-size for fallback format
    if args.fallback and fallback_needs_block_size and args.fallback_block_size is None:
        print(
            f"Error: --fallback-block-size is required when using --fallback {args.fallback}."
        )
        print("       Example: --fallback-block-size 128")
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    # Auto-enable comfy_quant if custom-type is used (required for mixed precision)
    if args.custom_type and not args.comfy_quant:
        print(
            "Note: --comfy_quant auto-enabled (required for --custom-type mixed precision)"
        )
        args.comfy_quant = True

    # Only check FP8 support if not using INT8
    if not args.int8:
        try:
            _ = torch.zeros(
                1,
                dtype=TARGET_FP8_DTYPE,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        except (RuntimeError, TypeError):
            print(
                "Error: This hardware/PyTorch version does not support the target FP8 dtype."
            )
            return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        # Build filename: {simple_|learned_}{format}[mixed]_{scaling}
        # TODO: SVD stats (k, top_p, lr) should be saved to _convert_to_quant_stats metadata entry
        prefix = "simple_" if args.simple else "learned_"
        if args.int8:
            format_str = "int8"
            if args.scaling_mode == "tensor":
                scaling_str = "_tensorwise"
            else:
                scaling_str = f"_bs{args.block_size}"
        else:
            format_str = "fp8"
            scaling_str = f"_{args.scaling_mode}"
        # Check for filters or custom-layers (metadata tracks specifics)
        filter_flags = extract_filter_flags(args)
        has_filters = any(filter_flags.values())
        has_custom = bool(args.custom_layers)
        mixed_suffix = "mixed" if (has_filters or has_custom) else ""
        output_file = f"{base}_{prefix}{format_str}{mixed_suffix}{scaling_str}.safetensors"
    else:
        output_file = args.output

    # Extract filter flags (needed for convert_to_fp8_scaled call)
    # This is done after output filename logic to avoid duplicate call when auto-generating filename
    if args.output:
        filter_flags = extract_filter_flags(args)

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = (
        int(torch.randint(0, 2**32 - 1, ()).item())
        if args.manual_seed == -1
        else args.manual_seed
    )
    print(f"Using seed: {seed}")

    # Load layer config if specified
    layer_config_data = None
    if args.layer_config:
        layer_config_data = load_layer_config(args.layer_config)

    # Call convert_to_fp8_scaled with explicit args (no **kwargs footgun)
    # Determine primary_format for NVFP4/MXFP8 mixed mode (when they fall through here)
    primary_format = None
    if args.nvfp4 and (args.custom_type or args.fallback):
        primary_format = "nvfp4"
    elif args.mxfp8 and (args.custom_type or args.fallback):
        primary_format = "mxfp8"

    convert_to_fp8_scaled(
        args.input,
        output_file,
        args.comfy_quant,
        # Filter flags
        filter_flags=filter_flags,
        # Calibration
        calib_samples=args.calib_samples,
        seed=seed,
        # Format options
        int8=args.int8,
        primary_format=primary_format,
        fallback=args.fallback,
        # Custom layer options
        custom_layers=args.custom_layers,
        exclude_layers=args.exclude_layers,
        custom_type=args.custom_type,
        custom_block_size=args.custom_block_size,
        custom_scaling_mode=args.custom_scaling_mode,
        custom_simple=args.custom_simple,
        custom_heur=args.custom_heur,
        # Fallback options
        fallback_block_size=args.fallback_block_size,
        fallback_simple=args.fallback_simple,
        # Precision options
        full_precision_matrix_mult=args.full_precision_matrix_mult,
        skip_inefficient_layers=args.heur,
        include_input_scale=args.input_scale,
        no_learned_rounding=args.simple,
        # Layer config
        layer_config=layer_config_data,
        layer_config_fullmatch=args.layer_config_fullmatch,
        # Output options
        save_quant_metadata=args.save_quant_metadata,
        low_memory=args.low_memory,
        # Optimizer/LR options (passed to LearnedRoundingConverter)
        optimizer=args.optimizer,
        num_iter=args.num_iter,
        lr=args.lr,
        lr_schedule=args.lr_schedule,
        top_p=args.top_p,
        min_k=args.min_k,
        max_k=args.max_k,
        full_matrix=args.full_matrix,
        scaling_mode=args.scaling_mode,
        block_size=args.block_size,
        # LR schedule tuning
        lr_gamma=args.lr_gamma,
        lr_patience=args.lr_patience,
        lr_factor=args.lr_factor,
        lr_min=args.lr_min,
        lr_cooldown=args.lr_cooldown,
        lr_threshold=args.lr_threshold,
        lr_adaptive_mode=args.lr_adaptive_mode,
        lr_shape_influence=args.lr_shape_influence,
        lr_threshold_mode=args.lr_threshold_mode,
        # Early stopping
        early_stop_loss=args.early_stop_loss,
        early_stop_lr=args.early_stop_lr,
        early_stop_stall=args.early_stop_stall,
    )

if __name__ == "__main__":
    main()
