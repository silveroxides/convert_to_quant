"""
convert_to_quant - Quantization toolkit for safetensors models.

Provides tools for converting model weights to FP8/INT8 quantized formats
with optional learned rounding optimization for ComfyUI inference.
"""

try:
    from importlib.metadata import version
    __version__ = version("convert_to_quant")
except Exception:
    __version__ = "0.0.0"  # Fallback when not installed as package

from .convert_to_quant import main

__all__ = ["main", "__version__"]

