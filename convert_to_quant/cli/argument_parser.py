"""
CLI argument parser for convert_to_quant.

Provides MultiHelpArgumentParser with categorized help sections
for experimental, filter, and advanced options.
"""
import argparse
import sys

from ..constants import MODEL_FILTERS


# --- CLI Help Sections ---
# Arguments categorized for multi-section help output

EXPERIMENTAL_ARGS = {
    "int8",
    "nvfp4",
    "mxfp8",
    "sdnq",
    "make_hybrid_mxfp8",
    "tensor_scales_path",
    "fallback",
    "custom_layers",
    "custom_type",
    "custom_block_size",
    "custom_scaling_mode",
    "custom_simple",
    "custom_heur",
    "fallback_block_size",
    "fallback_simple",
    "layer_config",
    "layer_config_fullmatch",
    "exclude_layers",
    "scaling_mode",
    "block_size",
    "input_scales_path",
    "input_scale",
    "no_normalize_scales",
}

# Generated from MODEL_FILTERS registry
FILTER_ARGS = set(MODEL_FILTERS.keys())

SDNQ_ARGS = {
    "sdnq_dtype",
    "sdnq_group_size",
    "sdnq_use_svd",
    "sdnq_svd_rank",
    "sdnq_svd_steps",
    "sdnq_stochastic",
}

ADVANCED_ARGS = {
    # LR Schedule parameters
    "lr_gamma",
    "lr_patience",
    "lr_factor",
    "lr_min",
    "lr_cooldown",
    "lr_threshold",
    "lr_adaptive_mode",
    "lr_shape_influence",
    "lr_threshold_mode",
    # Early stopping
    "early_stop_loss",
    "early_stop_lr",
    "early_stop_stall",
    # NVFP4 scale optimization
    "scale_refinement_rounds",
    "scale_optimization",
}

LEARNED_ROUNDING_ARGS = {
    # SVD and calibration
    "full_matrix",
    "calib_samples",
    # Optimizer settings
    "optimizer",
    "num_iter",
    "lr",
    "lr_schedule",
    # SVD component selection
    "top_p",
    "min_k",
    "max_k",
}

MODES_ARGS = {
    # Format conversion modes
    "convert_fp8_scaled",
    "convert_int8_scaled",
    "make_hybrid_mxfp8",
    "tensor_scales_path",
    # Legacy operations
    "legacy_input_add",
    "cleanup_fp8_scaled",
    "scaled_fp8_marker",
    # Activation calibration
    "actcal",
    "actcal_samples",
    "actcal_percentile",
    "actcal_lora",
    "actcal_seed",
    "actcal_device",
    # Metadata editing
    "edit_quant",
    "remove_keys",
    "add_keys",
    "quant_filter",
    # Development tools
    "dry_run",
    # Conversion-specific options
    "hp_filter",
    "full_precision_mm",
}


class MultiHelpArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with multiple help sections for experimental and filter args."""

    def __init__(
        self,
        *args,
        experimental_args=None,
        filter_args=None,
        advanced_args=None,
        learned_rounding_args=None,
        modes_args=None,
        sdnq_args=None,
        **kwargs,
    ):
        self._experimental_args = experimental_args or set()
        self._filter_args = filter_args or set()
        self._advanced_args = advanced_args or set()
        self._learned_rounding_args = learned_rounding_args or set()
        self._modes_args = modes_args or set()
        self._sdnq_args = sdnq_args or set()
        self._all_actions = []  # Track all actions for section-specific help
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        action = super().add_argument(*args, **kwargs)
        if hasattr(self, "_all_actions"):
            self._all_actions.append(action)
        return action

    def parse_args(self, args=None, namespace=None):
        if args is None:
            args = sys.argv[1:]

        # Check for special help flags before parsing
        if "--help-learned" in args or "-hl" in args:
            self._print_learned_help()
            sys.exit(0)
        elif "--help-experimental" in args or "-he" in args:
            self._print_experimental_help()
            sys.exit(0)
        elif "--help-filters" in args or "-hf" in args:
            self._print_filters_help()
            sys.exit(0)
        elif "--help-advanced" in args or "-ha" in args:
            self._print_advanced_help()
            sys.exit(0)
        elif "--help-modes" in args or "-hm" in args:
            self._print_modes_help()
            sys.exit(0)
        elif "--help-sdnq" in args or "-hs" in args:
            self._print_sdnq_help()
            sys.exit(0)

        return super().parse_args(args, namespace)

    def _get_dest_name(self, action):
        """Get the destination name for an action."""
        return action.dest

    def _format_action_help(self, action):
        """Format a single action for help output."""
        # Get option strings
        opts = (
            ", ".join(action.option_strings) if action.option_strings else action.dest
        )

        # Get help text
        help_text = action.help or ""
        if help_text == argparse.SUPPRESS:
            return None

        # Format default if present and not suppressed
        if action.default is not None and action.default != argparse.SUPPRESS:
            if action.default is not False and action.default != "":
                if isinstance(action.default, str):
                    help_text += f" (default: '{action.default}')"
                else:
                    help_text += f" (default: {action.default})"

        # Format choices if present
        if action.choices:
            choices_str = ", ".join(str(c) for c in action.choices)
            help_text += f" [choices: {choices_str}]"

        return f"  {opts:30s} {help_text}"

    def _print_learned_help(self):
        """Print help for learned rounding optimization options."""
        print("Learned Rounding Optimization Options")
        print("=" * 60)
        print()
        print("These options control the learned rounding optimization process.")
        print("Use --simple to skip learned rounding and use faster quantization.")
        print()
        print("SVD & Calibration:")
        print("-" * 40)

        svd_args = ["full_matrix", "calib_samples"]
        for action in self._all_actions:
            if self._get_dest_name(action) in svd_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Optimizer Settings:")
        print("-" * 40)

        opt_args = ["optimizer", "num_iter", "lr", "lr_schedule"]
        for action in self._all_actions:
            if self._get_dest_name(action) in opt_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("SVD Component Selection:")
        print("-" * 40)

        comp_args = ["top_p", "min_k", "max_k"]
        for action in self._all_actions:
            if self._get_dest_name(action) in comp_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def _print_experimental_help(self):
        """Print help for experimental features."""
        print("Experimental Quantization Features")
        print("=" * 60)
        print()
        print("These are advanced/experimental options for non-default quantization")
        print("formats and fine-grained control. Use --help for standard options.")
        print()
        print("Alternative Quantization Formats:")
        print("-" * 40)

        format_args = [
            "int8",
            "nvfp4",
            "mxfp8",
            "make_hybrid_mxfp8",
            "tensor_scales_path",
            "fallback",
            "block_size",
            "scaling_mode",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in format_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Custom Layer Quantization:")
        print("-" * 40)

        custom_args = [
            "custom_layers",
            "custom_type",
            "custom_block_size",
            "custom_scaling_mode",
            "custom_simple",
            "custom_heur",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in custom_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Fallback Layer Options:")
        print("-" * 40)

        fallback_args = ["fallback", "fallback_block_size", "fallback_simple"]
        for action in self._all_actions:
            if self._get_dest_name(action) in fallback_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Performance Tuning:")
        print("-" * 40)

        perf_args = ["heur"]
        for action in self._all_actions:
            if self._get_dest_name(action) in perf_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def _print_filters_help(self):
        """Print help for model-specific filter presets."""
        print("Model-Specific Exclusion Filters")
        print("=" * 60)
        print()
        print("These flags keep certain model-specific layers in high precision")
        print("(not quantized). Multiple filters can be combined.")
        print()

        # Group filters by category from MODEL_FILTERS registry
        categories = {
            "text": "Text Encoders",
            "diffusion": "Diffusion Models (Flux-style)",
            "video": "Video Models",
            "image": "Image Models",
        }

        for cat_key, cat_name in categories.items():
            # Get filters in this category
            cat_filters = [
                name for name, cfg in MODEL_FILTERS.items()
                if cfg.get("category") == cat_key
            ]
            if not cat_filters:
                continue

            print(f"{cat_name}:")
            print("-" * 40)

            for action in self._all_actions:
                if self._get_dest_name(action) in cat_filters:
                    line = self._format_action_help(action)
                    if line:
                        print(line)
            print()

    def _print_advanced_help(self):
        """Print help for advanced LR tuning and early stopping options."""
        print("Advanced LR Tuning & Early Stopping Options")
        print("=" * 60)
        print()
        print("These options provide fine-grained control over the optimizer")
        print("learning rate schedules and early stopping behavior.")
        print()
        print("LR Schedule (Exponential):")
        print("-" * 40)

        exp_args = ["lr_gamma"]
        for action in self._all_actions:
            if self._get_dest_name(action) in exp_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("LR Schedule (Plateau):")
        print("-" * 40)

        plateau_args = [
            "lr_patience",
            "lr_factor",
            "lr_min",
            "lr_cooldown",
            "lr_threshold",
            "lr_shape_influence",
            "lr_threshold_mode",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in plateau_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("LR Schedule (Adaptive):")
        print("-" * 40)

        adaptive_args = ["lr_adaptive_mode"]
        for action in self._all_actions:
            if self._get_dest_name(action) in adaptive_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Early Stopping Thresholds:")
        print("-" * 40)

        early_args = ["early_stop_loss", "early_stop_lr", "early_stop_stall"]
        for action in self._all_actions:
            if self._get_dest_name(action) in early_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("NVFP4/MXFP8 Scale Optimization:")
        print("-" * 40)

        scale_args = ["scale_refinement_rounds", "scale_optimization"]
        for action in self._all_actions:
            if self._get_dest_name(action) in scale_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()


    def _print_modes_help(self):
        """Print help for conversion and utility modes."""
        print("Conversion & Utility Modes")
        print("=" * 60)
        print()
        print("These are specialized workflows separate from quantization.")
        print("Use --help for standard quantization options.")
        print()
        print("Format Conversion:")
        print("-" * 40)

        conversion_args = [
            "convert_fp8_scaled",
            "hp_filter",
            "full_precision_mm",
            "convert_int8_scaled",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in conversion_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Hybrid Format Creation:")
        print("-" * 40)

        hybrid_args = ["make_hybrid_mxfp8", "tensor_scales_path"]
        for action in self._all_actions:
            if self._get_dest_name(action) in hybrid_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Legacy Operations:")
        print("-" * 40)

        legacy_args = ["legacy_input_add", "cleanup_fp8_scaled", "scaled_fp8_marker"]
        for action in self._all_actions:
            if self._get_dest_name(action) in legacy_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Activation Scale Calibration:")
        print("-" * 40)

        actcal_args = [
            "actcal",
            "actcal_samples",
            "actcal_percentile",
            "actcal_lora",
            "actcal_seed",
            "actcal_device",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in actcal_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Metadata Editing:")
        print("-" * 40)

        edit_args = ["edit_quant", "remove_keys", "add_keys", "quant_filter"]
        for action in self._all_actions:
            if self._get_dest_name(action) in edit_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Development Tools:")
        print("-" * 40)

        dev_args = ["dry_run"]
        for action in self._all_actions:
            if self._get_dest_name(action) in dev_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()

    def _print_sdnq_help(self):
        """Print help for SDNQ options."""
        print("SDNQ (Stochastic Differentiable Neural Quantization) Options")
        print("=" * 60)
        print()
        print("Options for SDNQ quantization mode. Enable with --sdnq.")
        print()
        print("SDNQ Settings:")
        print("-" * 40)

        sdnq_args_list = [
            "sdnq_dtype",
            "sdnq_group_size",
            "sdnq_use_svd",
            "sdnq_svd_rank",
            "sdnq_svd_steps",
            "sdnq_stochastic",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in sdnq_args_list:
                line = self._format_action_help(action)
                if line:
                    print(line)
        print()

    def format_help(self):
        """Override to add section hints and hide experimental/filter args."""
        # Build custom help output
        formatter = self._get_formatter()

        # Add standard arguments only (filter out all special categories)
        standard_actions = []
        for action in self._actions:
            dest = self._get_dest_name(action)
            if (
                dest not in self._experimental_args
                and dest not in self._filter_args
                and dest not in self._advanced_args
                and dest not in self._modes_args
                and dest not in self._learned_rounding_args
                and dest not in self._sdnq_args
            ):
                standard_actions.append(action)

        # Add usage with only standard actions
        formatter.add_usage(
            self.usage, standard_actions, self._mutually_exclusive_groups
        )

        # Add description
        formatter.add_text(self.description)

        # Group standard actions
        formatter.start_section("Standard Options")
        formatter.add_arguments(standard_actions)
        formatter.end_section()

        # Add section hints
        formatter.add_text("")
        formatter.add_text("Additional Help Sections:")
        formatter.add_text(
            "  --help-learned, -hl         Show learned rounding optimization options"
        )
        formatter.add_text(
            "                              (optimizer, num_iter, lr, top_p, etc.)"
        )
        formatter.add_text(
            "  --help-experimental, -he    Show experimental quantization options"
        )
        formatter.add_text(
            "                              (int8, nvfp4, mxfp8, custom-layers, etc.)"
        )
        formatter.add_text(
            "  --help-filters, -hf         Show model-specific exclusion filters"
        )
        formatter.add_text(
            "                              (t5xxl, hunyuan, wan, qwen, etc.)"
        )
        formatter.add_text(
            "  --help-advanced, -ha        Show advanced LR tuning and early stopping"
        )
        formatter.add_text(
            "                              (lr-gamma, lr-patience, early-stop-*, etc.)"
        )
        formatter.add_text(
            "  --help-modes, -hm           Show conversion and utility modes"
        )
        formatter.add_text(
            "                              (convert-fp8-scaled, actcal, edit-quant, etc.)"
        )
        formatter.add_text(
            "  --help-sdnq, -hs            Show SDNQ quantization options"
        )
        formatter.add_text(
            "                              (sdnq-dtype, sdnq-group-size, sdnq-use-svd, etc.)"
        )

        # Add epilog
        formatter.add_text(self.epilog)

        return formatter.format_help()
