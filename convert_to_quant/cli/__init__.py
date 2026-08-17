"""CLI package for convert_to_quant."""

from .argument_parser import ADVANCED_ARGS, EXPERIMENTAL_ARGS, FILTER_ARGS, MultiHelpArgumentParser


def main(*args, **kwargs):
    """Load the CLI entry point lazily so ``python -m ...cli.main`` stays warning-free."""
    from .main import main as cli_main

    return cli_main(*args, **kwargs)

__all__ = ["MultiHelpArgumentParser", "EXPERIMENTAL_ARGS", "FILTER_ARGS", "ADVANCED_ARGS", "main"]
