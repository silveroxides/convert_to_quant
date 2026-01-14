"""
Programmatic API for convert_to_quant.

This module provides a Python-callable interface for quantization,
designed for environments where subprocess calls are not allowed
(e.g., HuggingFace ZeroGPU Spaces).

Usage:
    from convert_to_quant.api import convert, ConversionConfig

    config = ConversionConfig(
        input_path="model.safetensors",
        output_path="model_fp8.safetensors",
        quant_format="fp8",
        comfy_quant=True,
    )
    result = convert(config)
"""
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import os
import torch

from .formats.fp8_conversion import convert_to_fp8_scaled
from .formats.nvfp4_conversion import convert_to_nvfp4
from .formats.mxfp8_conversion import convert_to_mxfp8
from .formats.format_migration import convert_fp8_scaled_to_comfy_quant
from .formats.int8_conversion import convert_int8_to_comfy_quant
from .utils.logging import setup_logging


@dataclass
class ConversionConfig:
    """Configuration for model quantization.
    
    Attributes:
        input_path: Path to input safetensors file
        output_path: Path for output file (auto-generated if None)
        quant_format: Quantization format: 'fp8', 'int8', 'nvfp4', 'mxfp8'
        comfy_quant: Use ComfyUI quantization metadata format
        scaling_mode: FP8 scaling: 'tensor', 'row', 'block', 'block3d'
        block_size: Block size for block-wise quantization
        simple: Skip learned rounding optimization, use simple quantization
        heur: Skip layers with poor quantization characteristics
        filter_flags: Dict of model filter flags (e.g., {'radiance': True})
        exclude_layers: Regex pattern for layers to exclude
        custom_layers: Regex pattern for custom quantization layers
        custom_type: Quantization type for custom layers
        num_iter: Optimization iterations per tensor
        calib_samples: Number of calibration samples
        seed: Random seed (-1 for random)
        low_memory: Use streaming tensor loading
        verbose: Logging level: 'DEBUG', 'VERBOSE', 'NORMAL', 'MINIMAL'
    """
    input_path: str
    output_path: Optional[str] = None
    quant_format: str = "fp8"  # 'fp8', 'int8', 'nvfp4', 'mxfp8'
    comfy_quant: bool = True
    
    # Scaling options
    scaling_mode: str = "tensor"  # 'tensor', 'row', 'block', 'block3d'
    block_size: Optional[int] = None
    
    # Optimization options
    simple: bool = False
    heur: bool = False
    num_iter: int = 1000
    calib_samples: int = 6144
    seed: int = -1
    
    # Filter options
    filter_flags: Dict[str, bool] = field(default_factory=dict)
    exclude_layers: Optional[str] = None
    custom_layers: Optional[str] = None
    custom_type: Optional[str] = None
    custom_block_size: Optional[int] = None
    custom_scaling_mode: Optional[str] = None
    custom_simple: bool = False
    custom_heur: bool = False
    
    # Fallback options
    fallback: Optional[str] = None
    fallback_block_size: Optional[int] = None
    fallback_simple: bool = False
    
    # Advanced options
    optimizer: str = "original"
    lr: float = 8.077300000003e-3
    lr_schedule: str = "adaptive"
    lr_gamma: float = 0.99
    lr_patience: int = 9
    lr_factor: float = 0.92
    lr_min: float = 1e-10
    lr_cooldown: int = 6
    lr_threshold: float = 0.0
    lr_adaptive_mode: str = "simple-reset"
    lr_shape_influence: float = 1.0
    lr_threshold_mode: str = "rel"
    
    # Early stopping
    early_stop_loss: float = 1e-8
    early_stop_lr: float = 1e-10
    early_stop_stall: int = 1000
    
    # SVD options
    top_p: float = 0.2
    min_k: int = 64
    max_k: int = 1024
    full_matrix: bool = False
    
    # Output options
    full_precision_matrix_mult: bool = False
    include_input_scale: bool = False
    save_quant_metadata: bool = False
    low_memory: bool = False
    
    # Logging
    verbose: str = "NORMAL"


@dataclass
class ConversionResult:
    """Result of a quantization conversion.
    
    Attributes:
        success: Whether conversion completed successfully
        output_path: Path to output file
        error: Error message if failed
        stats: Optional statistics about the conversion
    """
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None
    stats: Optional[Dict[str, Any]] = None


def _generate_output_path(config: ConversionConfig) -> str:
    """Generate output path based on input and quantization settings."""
    base = os.path.splitext(config.input_path)[0]
    
    prefix = "simple_" if config.simple else "learned_"
    
    if config.quant_format == "nvfp4":
        format_str = "nvfp4"
    elif config.quant_format == "mxfp8":
        format_str = "mxfp8"
    elif config.quant_format == "int8":
        format_str = "int8"
        prefix = "simple_" if config.simple else "comfy_"
    else:  # fp8
        format_str = "fp8"
        prefix = "comfy_" if config.comfy_quant else prefix
    
    # Check for mixed quantization
    has_filters = any(config.filter_flags.values())
    has_custom = bool(config.custom_layers)
    mixed_suffix = "mixed" if (has_filters or has_custom) else ""
    
    # Add scaling mode for FP8
    scaling_suffix = ""
    if config.quant_format == "fp8" and config.scaling_mode != "tensor":
        scaling_suffix = f"_{config.scaling_mode}"
    
    return f"{base}_{prefix}{format_str}{mixed_suffix}{scaling_suffix}.safetensors"


def convert(config: ConversionConfig) -> ConversionResult:
    """Convert a model to quantized format.
    
    Args:
        config: ConversionConfig with all quantization parameters
        
    Returns:
        ConversionResult with success status and output path
        
    Example:
        >>> from convert_to_quant.api import convert, ConversionConfig
        >>> config = ConversionConfig(
        ...     input_path="model.safetensors",
        ...     quant_format="mxfp8",
        ...     comfy_quant=True,
        ... )
        >>> result = convert(config)
        >>> print(result.output_path)
    """
    # Initialize logging
    setup_logging(config.verbose)
    
    # Validate input
    if not os.path.exists(config.input_path):
        return ConversionResult(
            success=False,
            error=f"Input file not found: {config.input_path}"
        )
    
    # Generate output path if not provided
    output_path = config.output_path or _generate_output_path(config)
    
    # Validate output path
    if os.path.abspath(config.input_path) == os.path.abspath(output_path):
        return ConversionResult(
            success=False,
            error="Output file cannot be same as input"
        )
    
    # Compute seed
    seed = (
        int(torch.randint(0, 2**32 - 1, ()).item())
        if config.seed == -1
        else config.seed
    )
    
    try:
        if config.quant_format == "nvfp4":
            convert_to_nvfp4(
                config.input_path,
                output_path,
                filter_flags=config.filter_flags,
                exclude_layers=config.exclude_layers,
                simple=config.simple,
                num_iter=config.num_iter,
                heur=config.heur,
                calib_samples=config.calib_samples,
                seed=seed,
                optimizer=config.optimizer,
                lr=config.lr,
                lr_schedule=config.lr_schedule,
                top_p=config.top_p,
                min_k=config.min_k,
                max_k=config.max_k,
                full_matrix=config.full_matrix,
                lr_gamma=config.lr_gamma,
                lr_patience=config.lr_patience,
                lr_factor=config.lr_factor,
                lr_min=config.lr_min,
                lr_cooldown=config.lr_cooldown,
                lr_threshold=config.lr_threshold,
                lr_adaptive_mode=config.lr_adaptive_mode,
                lr_shape_influence=config.lr_shape_influence,
                lr_threshold_mode=config.lr_threshold_mode,
                early_stop_loss=config.early_stop_loss,
                early_stop_lr=config.early_stop_lr,
                early_stop_stall=config.early_stop_stall,
                low_memory=config.low_memory,
            )
            
        elif config.quant_format == "mxfp8":
            convert_to_mxfp8(
                config.input_path,
                output_path,
                filter_flags=config.filter_flags,
                exclude_layers=config.exclude_layers,
                simple=config.simple,
                num_iter=config.num_iter,
                heur=config.heur,
                calib_samples=config.calib_samples,
                seed=seed,
                optimizer=config.optimizer,
                lr=config.lr,
                lr_schedule=config.lr_schedule,
                top_p=config.top_p,
                min_k=config.min_k,
                max_k=config.max_k,
                full_matrix=config.full_matrix,
                lr_gamma=config.lr_gamma,
                lr_patience=config.lr_patience,
                lr_factor=config.lr_factor,
                lr_min=config.lr_min,
                lr_cooldown=config.lr_cooldown,
                lr_threshold=config.lr_threshold,
                lr_adaptive_mode=config.lr_adaptive_mode,
                lr_shape_influence=config.lr_shape_influence,
                lr_threshold_mode=config.lr_threshold_mode,
                early_stop_loss=config.early_stop_loss,
                early_stop_lr=config.early_stop_lr,
                early_stop_stall=config.early_stop_stall,
                low_memory=config.low_memory,
            )
            
        else:  # fp8 or int8
            convert_to_fp8_scaled(
                config.input_path,
                output_path,
                config.comfy_quant,
                filter_flags=config.filter_flags,
                calib_samples=config.calib_samples,
                seed=seed,
                int8=config.quant_format == "int8",
                fallback=config.fallback,
                custom_layers=config.custom_layers,
                exclude_layers=config.exclude_layers,
                custom_type=config.custom_type,
                custom_block_size=config.custom_block_size,
                custom_scaling_mode=config.custom_scaling_mode,
                custom_simple=config.custom_simple,
                custom_heur=config.custom_heur,
                fallback_block_size=config.fallback_block_size,
                fallback_simple=config.fallback_simple,
                full_precision_matrix_mult=config.full_precision_matrix_mult,
                skip_inefficient_layers=config.heur,
                include_input_scale=config.include_input_scale,
                no_learned_rounding=config.simple,
                save_quant_metadata=config.save_quant_metadata,
                low_memory=config.low_memory,
                optimizer=config.optimizer,
                num_iter=config.num_iter,
                lr=config.lr,
                lr_schedule=config.lr_schedule,
                top_p=config.top_p,
                min_k=config.min_k,
                max_k=config.max_k,
                full_matrix=config.full_matrix,
                scaling_mode=config.scaling_mode,
                block_size=config.block_size,
                lr_gamma=config.lr_gamma,
                lr_patience=config.lr_patience,
                lr_factor=config.lr_factor,
                lr_min=config.lr_min,
                lr_cooldown=config.lr_cooldown,
                lr_threshold=config.lr_threshold,
                lr_adaptive_mode=config.lr_adaptive_mode,
                lr_shape_influence=config.lr_shape_influence,
                lr_threshold_mode=config.lr_threshold_mode,
                early_stop_loss=config.early_stop_loss,
                early_stop_lr=config.early_stop_lr,
                early_stop_stall=config.early_stop_stall,
            )
        
        return ConversionResult(
            success=True,
            output_path=output_path,
        )
        
    except Exception as e:
        return ConversionResult(
            success=False,
            error=str(e),
        )


__all__ = ["ConversionConfig", "ConversionResult", "convert"]
