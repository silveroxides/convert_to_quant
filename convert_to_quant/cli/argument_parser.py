"""
CLI argument parser for convert_to_quant.

Provides MultiHelpArgumentParser with categorized help sections
for experimental, filter, and advanced options.
"""
import argparse
import sys


# --- CLI Help Sections ---
# Arguments categorized for multi-section help output

EXPERIMENTAL_ARGS = {
    "int8",
    "nvfp4",
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
    "gen_layer_template",
}

FILTER_ARGS = {
    "t5xxl",
    "mistral",
    "visual",
    "flux2",
    "distillation_large",
    "distillation_small",
    "nerf_large",
    "nerf_small",
    "radiance",
    "wan",
    "qwen",
    "hunyuan",
    "zimage",
    "zimage_refiner",
}

ADVANCED_ARGS = {
    "lr_shape_influence",
    "lr_threshold_mode",
    "early_stop_loss",
    "early_stop_lr",
    "early_stop_stall",
}

class MultiHelpArgumentParser(argparse.ArgumentParser):
    """ArgumentParser with multiple help sections for experimental and filter args."""

    def __init__(
        self,
        *args,
        experimental_args=None,
        filter_args=None,
        advanced_args=None,
        **kwargs,
    ):
        self._experimental_args = experimental_args or set()
        self._filter_args = filter_args or set()
        self._advanced_args = advanced_args or set()
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
        if "--help-experimental" in args or "-he" in args:
            self._print_experimental_help()
            sys.exit(0)
        elif "--help-filters" in args or "-hf" in args:
            self._print_filters_help()
            sys.exit(0)
        elif "--help-advanced" in args or "-ha" in args:
            self._print_advanced_help()
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

        fallback_args = ["fallback_block_size", "fallback_simple"]
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
        print("Text Encoders:")
        print("-" * 40)

        text_args = ["t5xxl", "mistral", "visual"]
        for action in self._all_actions:
            if self._get_dest_name(action) in text_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Diffusion Models (Flux-style):")
        print("-" * 40)

        diffusion_args = [
            "flux2",
            "distillation_large",
            "distillation_small",
            "nerf_large",
            "nerf_small",
            "radiance",
        ]
        for action in self._all_actions:
            if self._get_dest_name(action) in diffusion_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Video Models:")
        print("-" * 40)

        video_args = ["wan", "hunyuan"]
        for action in self._all_actions:
            if self._get_dest_name(action) in video_args:
                line = self._format_action_help(action)
                if line:
                    print(line)

        print()
        print("Image Models:")
        print("-" * 40)

        image_args = ["qwen", "zimage", "zimage_refiner"]
        for action in self._all_actions:
            if self._get_dest_name(action) in image_args:
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
        print("Shape-Adaptive LR (Plateau Schedule):")
        print("-" * 40)

        shape_args = ["lr_shape_influence", "lr_threshold_mode"]
        for action in self._all_actions:
            if self._get_dest_name(action) in shape_args:
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

    def format_help(self):
        """Override to add section hints and hide experimental/filter args."""
        # Build custom help output
        formatter = self._get_formatter()

        # Add standard arguments only (filter out experimental and filter args)
        standard_actions = []
        for action in self._actions:
            dest = self._get_dest_name(action)
            if (
                dest not in self._experimental_args
                and dest not in self._filter_args
                and dest not in self._advanced_args
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
            "  --help-experimental, -he    Show experimental quantization options"
        )
        formatter.add_text(
            "                              (int8, custom-layers, scaling_mode, etc.)"
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
            "                              (lr-shape-influence, early-stop-*, etc.)"
        )

        # Add epilog
        formatter.add_text(self.epilog)

        return formatter.format_help()
