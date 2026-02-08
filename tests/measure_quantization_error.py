"""
Quantization Error Measurement Tool

Loads two safetensors models (one BF16, one FP8 quantized) and measures the
quantization error between them. Provides detailed per-layer and aggregate statistics.

Usage:
    python tests/measure_quantization_error.py \
        --original model_bf16.safetensors \
        --quantized model_fp8.safetensors \
        [--layers-to-compare "layer1,layer2"] \
        [--output-report report.json]
"""

import argparse
import json
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_to_quant.utils.tensor_utils import tensor_to_dict
from convert_to_quant.utils.logging import setup_logging, info, verbose, warning, error
from convert_to_quant.utils.memory_efficient_loader import MemoryEfficientSafeOpen


@dataclass
class LayerErrorMetrics:
    """Per-layer quantization error metrics."""
    
    layer_name: str
    original_dtype: str
    quantized_dtype: str
    tensor_shape: Tuple[int, ...]
    tensor_elements: int
    
    # Error metrics
    mean_absolute_error: float
    relative_error: float  # MAE / mean(|original|)
    max_absolute_error: float
    min_absolute_error: float
    std_absolute_error: float
    
    # Squared error metrics
    mean_squared_error: float
    root_mean_squared_error: float
    
    # Statistical metrics
    max_relative_error: float
    mean_relative_error: float
    
    # Power metrics
    original_norm: float
    quantized_norm: float
    error_norm: float
    
    # Signal-to-noise ratio
    snr_db: float  # 20 * log10(signal_norm / error_norm)
    
    # Correlation metrics (fitness measures)
    pearson_correlation: float  # -1 to 1, how well aligned are the weights
    cosine_similarity: float  # 0 to 1, angular similarity
    
    # Quantized-specific metadata
    scale_factor: Optional[float] = None
    block_size: Optional[int] = None
    comfy_format: Optional[str] = None


@dataclass
class AggregateErrorMetrics:
    """Aggregate quantization error statistics."""
    
    total_layers: int
    total_tensors_compared: int
    total_elements: int
    
    # Overall error metrics
    mean_absolute_error: float
    median_absolute_error: float
    max_absolute_error: float
    
    # Squared error metrics
    mean_squared_error: float
    root_mean_squared_error: float
    
    # Relative error metrics
    mean_relative_error: float
    median_relative_error: float
    max_relative_error: float
    
    # Power metrics
    total_original_norm: float
    total_quantized_norm: float
    total_error_norm: float
    overall_snr_db: float  # 20 * log10(signal_norm / error_norm)
    
    # Correlation and fitness metrics
    mean_pearson_correlation: float  # Average -1 to 1
    median_pearson_correlation: float
    mean_cosine_similarity: float  # Average 0 to 1
    median_cosine_similarity: float
    
    # Distribution
    relative_error_percentiles: Dict[str, float]


class QuantizationErrorMeasurer:
    """Measures quantization error between original and quantized models."""
    
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        low_memory: bool = False,
    ):
        self.device = device
        self.low_memory = low_memory
        self.layer_metrics: List[LayerErrorMetrics] = []
        
        info(f"Using device: {self.device}")
        if low_memory:
            info("Memory-efficient mode: Tensors will be streamed on-demand")
    
    def load_model(self, path: str) -> Tuple[Dict[str, torch.Tensor], Optional[Dict]]:
        """Load a safetensors model using memory-efficient loader."""
        info(f"Loading model: {path}")
        tensors = {}
        metadata = None
        
        try:
            loader = MemoryEfficientSafeOpen(path, low_memory=self.low_memory)
            metadata = loader.metadata()
            
            # Load all tensors (standard mode) or stream on-demand (low-memory mode)
            if self.low_memory:
                # In low-memory mode, load tensors on-demand
                for key in loader.keys():
                    tensors[key] = loader.get_tensor(key)
            else:
                # In standard mode, all tensors are already loaded
                for key in loader.keys():
                    tensors[key] = loader.get_tensor(key)
            
            loader.close()
            
            info(f"  Loaded {len(tensors)} tensors")
            if metadata:
                verbose(f"  Metadata keys: {list(metadata.keys())}")
            
            return tensors, metadata
        except Exception as e:
            error(f"Failed to load model '{path}': {e}")
            return None, None
    
    def parse_quantization_metadata(self, metadata: Optional[Dict]) -> Dict[str, Any]:
        """Parse quantization metadata from safetensors header."""
        if not metadata or "_quantization_metadata" not in metadata:
            return {}
        
        try:
            return json.loads(metadata["_quantization_metadata"])
        except Exception as e:
            warning(f"Failed to parse quantization metadata: {e}")
            return {}
    
    def dequantize_fp8(
        self,
        quantized: torch.Tensor,
        scale: torch.Tensor,
        original_dtype: torch.dtype
    ) -> torch.Tensor:
        """Dequantize FP8 tensor back to original dtype."""
        # FP8 dequantization: dequant = quant * scale
        dequantized = quantized.to(torch.float32) * scale.to(torch.float32)
        return dequantized.to(original_dtype)
    
    def get_scale_from_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        layer_name: str
    ) -> Optional[torch.Tensor]:
        """Extract scale tensor for a quantized layer."""
        # Try common scale tensor naming conventions
        scale_names = [
            f"{layer_name}.weight_scale",
            f"{layer_name}.scale",
            f"{layer_name}_scale",
        ]
        
        for scale_name in scale_names:
            if scale_name in tensors:
                return tensors[scale_name]
        
        return None
    
    def get_comfy_quant_config(
        self,
        tensors: Dict[str, torch.Tensor],
        layer_name: str
    ) -> Optional[Dict]:
        """Extract .comfy_quant metadata for a layer."""
        comfy_quant_key = f"{layer_name}.comfy_quant"
        if comfy_quant_key not in tensors:
            return None
        
        try:
            return tensor_to_dict(tensors[comfy_quant_key])
        except Exception as e:
            verbose(f"Failed to parse .comfy_quant for {layer_name}: {e}")
            return None
    
    def compute_error_metrics(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
        layer_name: str,
        scale: Optional[torch.Tensor] = None,
        comfy_config: Optional[Dict] = None,
    ) -> Optional[LayerErrorMetrics]:
        """Compute detailed error metrics for a single layer."""
        
        # Check for empty tensors
        if original.numel() == 0 or quantized.numel() == 0:
            warning(f"  Skipping {layer_name}: empty tensor")
            return None
        
        # Move to device and convert to float32 for computation
        original_f32 = original.to(self.device, dtype=torch.float32)
        quantized_f32 = quantized.to(self.device, dtype=torch.float32)
        
        # Dequantize if scale is available
        if scale is not None:
            scale_f32 = scale.to(self.device, dtype=torch.float32)
            # Dequantize: quantized_value * scale
            quantized_f32 = quantized_f32 * scale_f32
        
        # Compute absolute error
        absolute_error = torch.abs(original_f32 - quantized_f32)
        
        # Compute relative error
        original_abs = torch.abs(original_f32)
        original_mean = original_abs.mean().item()
        
        if original_mean > 1e-7:
            relative_error = (absolute_error / (original_abs + 1e-8)).clamp(max=1e6)
        else:
            relative_error = absolute_error
        
        # Compute squared error metrics
        squared_error = (original_f32 - quantized_f32) ** 2
        mse = squared_error.mean().item()
        rmse = torch.sqrt(torch.tensor(mse)).item()
        
        # Compute norms
        original_norm = torch.norm(original_f32).item()
        quantized_norm = torch.norm(quantized_f32).item()
        error_norm = torch.norm(absolute_error).item()
        
        # Compute SNR (Signal-to-Noise Ratio)
        if error_norm > 1e-10:
            snr_db = 20.0 * torch.log10(torch.tensor(original_norm / error_norm)).item()
        else:
            snr_db = float('inf')
        
        # Compute Pearson correlation coefficient
        try:
            original_flat = original_f32.flatten()
            quantized_flat = quantized_f32.flatten()
            
            # Standardize
            orig_mean = original_flat.mean()
            quant_mean = quantized_flat.mean()
            orig_std = original_flat.std()
            quant_std = quantized_flat.std()
            
            if orig_std > 1e-10 and quant_std > 1e-10:
                correlation = (
                    ((original_flat - orig_mean) * (quantized_flat - quant_mean)).mean()
                    / (orig_std * quant_std)
                ).item()
                # Clamp to [-1, 1] to handle numerical errors
                correlation = max(-1.0, min(1.0, correlation))
            else:
                correlation = 0.0
        except Exception:
            correlation = 0.0
        
        # Compute cosine similarity
        try:
            original_flat = original_f32.flatten()
            quantized_flat = quantized_f32.flatten()
            
            dot_product = (original_flat * quantized_flat).sum().item()
            orig_norm_flat = torch.norm(original_flat).item()
            quant_norm_flat = torch.norm(quantized_flat).item()
            
            if orig_norm_flat > 1e-10 and quant_norm_flat > 1e-10:
                cosine_sim = dot_product / (orig_norm_flat * quant_norm_flat)
                # Clamp to [0, 1] range
                cosine_sim = max(0.0, min(1.0, cosine_sim))
            else:
                cosine_sim = 0.0
        except Exception:
            cosine_sim = 0.0
        
        # Gather statistics
        mae = absolute_error.mean().item()
        max_ae = absolute_error.max().item() if absolute_error.numel() > 0 else 0.0
        min_ae = absolute_error.min().item() if absolute_error.numel() > 0 else 0.0
        std_ae = absolute_error.std().item() if absolute_error.numel() > 1 else 0.0
        
        max_rel_err = relative_error.max().item()
        mean_rel_err = relative_error.mean().item()
        
        # Extract scale if available
        scale_value = None
        if scale is not None:
            scale_value = scale.mean().item() if scale.numel() > 1 else scale.item()
        
        # Extract comfy config info
        block_size = None
        comfy_format = None
        if comfy_config:
            block_size = comfy_config.get("group_size") or comfy_config.get("block_size")
            comfy_format = comfy_config.get("format")
        
        return LayerErrorMetrics(
            layer_name=layer_name,
            original_dtype=str(original.dtype),
            quantized_dtype=str(quantized.dtype),
            tensor_shape=tuple(original.shape),
            tensor_elements=original.numel(),
            mean_absolute_error=mae,
            relative_error=mae / max(original_mean, 1e-7),
            max_absolute_error=max_ae,
            min_absolute_error=min_ae,
            std_absolute_error=std_ae,
            mean_squared_error=mse,
            root_mean_squared_error=rmse,
            max_relative_error=max_rel_err,
            mean_relative_error=mean_rel_err,
            original_norm=original_norm,
            quantized_norm=quantized_norm,
            error_norm=error_norm,
            snr_db=snr_db,
            pearson_correlation=correlation,
            cosine_similarity=cosine_sim,
            scale_factor=scale_value,
            block_size=block_size,
            comfy_format=comfy_format,
        )
    
    def compare_models(
        self,
        original_tensors: Dict[str, torch.Tensor],
        quantized_tensors: Dict[str, torch.Tensor],
        original_metadata: Optional[Dict] = None,
        quantized_metadata: Optional[Dict] = None,
        layer_filter: Optional[List[str]] = None,
    ) -> bool:
        """
        Compare original and quantized models.
        
        Args:
            original_tensors: Tensors from original (BF16) model
            quantized_tensors: Tensors from quantized (FP8) model
            original_metadata: Metadata from original model
            quantized_metadata: Metadata from quantized model
            layer_filter: Optional list of layer names to compare
        
        Returns:
            True if comparison succeeded, False otherwise
        """
        
        # Parse quantization metadata
        quant_metadata = self.parse_quantization_metadata(quantized_metadata)
        quant_layers = quant_metadata.get("layers", {})
        
        # Find matching tensor pairs
        matched_pairs = []
        skipped_layers = []
        
        for key in original_tensors.keys():
            # Skip non-weight tensors
            if key.endswith((".comfy_quant", "_scale", "bias")):
                continue
            
            # Filter by layer name if specified
            if layer_filter and not any(layer in key for layer in layer_filter):
                continue
            
            # Look for corresponding quantized tensor
            if key in quantized_tensors:
                matched_pairs.append(key)
            else:
                skipped_layers.append(key)
        
        if not matched_pairs:
            error("No matching layers found between original and quantized models")
            return False
        
        info(f"Comparing {len(matched_pairs)} layers")
        if skipped_layers:
            verbose(f"Skipped {len(skipped_layers)} layers (not in quantized model)")
        
        # Compare each pair
        for layer_name in matched_pairs:
            original_tensor = original_tensors[layer_name]
            quantized_tensor = quantized_tensors[layer_name]
            
            # Check shape compatibility
            if original_tensor.shape != quantized_tensor.shape:
                warning(f"  Shape mismatch for {layer_name}: "
                       f"{original_tensor.shape} vs {quantized_tensor.shape}")
                continue
            
            # Get scale tensor
            scale = self.get_scale_from_tensors(quantized_tensors, layer_name)
            
            # Get comfy config
            comfy_config = self.get_comfy_quant_config(quantized_tensors, layer_name)
            
            # Compute metrics
            metrics = self.compute_error_metrics(
                original_tensor,
                quantized_tensor,
                layer_name,
                scale=scale,
                comfy_config=comfy_config,
            )
            
            if metrics:
                self.layer_metrics.append(metrics)
        
        return len(self.layer_metrics) > 0
    
    def compute_aggregate_metrics(self) -> Optional[AggregateErrorMetrics]:
        """Compute aggregate error metrics from all layers."""
        
        if not self.layer_metrics:
            warning("No layer metrics available")
            return None
        
        # Collect statistics
        mae_list = [m.mean_absolute_error for m in self.layer_metrics]
        mse_list = [m.mean_squared_error for m in self.layer_metrics]
        rmse_list = [m.root_mean_squared_error for m in self.layer_metrics]
        rel_err_list = [m.relative_error for m in self.layer_metrics]
        max_rel_err_list = [m.max_relative_error for m in self.layer_metrics]
        pearson_list = [m.pearson_correlation for m in self.layer_metrics]
        cosine_list = [m.cosine_similarity for m in self.layer_metrics]
        
        total_elements = sum(m.tensor_elements for m in self.layer_metrics)
        total_original_norm = sum(m.original_norm for m in self.layer_metrics)
        total_quantized_norm = sum(m.quantized_norm for m in self.layer_metrics)
        total_error_norm = sum(m.error_norm for m in self.layer_metrics)
        
        # Sort for percentiles and medians
        sorted_mae = sorted(mae_list)
        sorted_rel_err = sorted(rel_err_list)
        sorted_pearson = sorted(pearson_list)
        sorted_cosine = sorted(cosine_list)
        
        # Overall SNR
        if total_error_norm > 1e-10:
            overall_snr_db = 20.0 * torch.log10(
                torch.tensor(total_original_norm / total_error_norm)
            ).item()
        else:
            overall_snr_db = float('inf')
        
        return AggregateErrorMetrics(
            total_layers=len(self.layer_metrics),
            total_tensors_compared=len(self.layer_metrics),
            total_elements=total_elements,
            mean_absolute_error=sum(mae_list) / len(mae_list),
            median_absolute_error=sorted_mae[len(sorted_mae) // 2],
            max_absolute_error=max(mae_list),
            mean_squared_error=sum(mse_list) / len(mse_list),
            root_mean_squared_error=sum(rmse_list) / len(rmse_list),
            mean_relative_error=sum(rel_err_list) / len(rel_err_list),
            median_relative_error=sorted_rel_err[len(sorted_rel_err) // 2],
            max_relative_error=max(max_rel_err_list),
            total_original_norm=total_original_norm,
            total_quantized_norm=total_quantized_norm,
            total_error_norm=total_error_norm,
            overall_snr_db=overall_snr_db,
            mean_pearson_correlation=sum(pearson_list) / len(pearson_list),
            median_pearson_correlation=sorted_pearson[len(sorted_pearson) // 2],
            mean_cosine_similarity=sum(cosine_list) / len(cosine_list),
            median_cosine_similarity=sorted_cosine[len(sorted_cosine) // 2],
            relative_error_percentiles={
                "p50": sorted_rel_err[int(0.50 * len(sorted_rel_err))],
                "p75": sorted_rel_err[int(0.75 * len(sorted_rel_err))],
                "p90": sorted_rel_err[int(0.90 * len(sorted_rel_err))],
                "p95": sorted_rel_err[int(0.95 * len(sorted_rel_err))],
                "p99": sorted_rel_err[int(0.99 * len(sorted_rel_err))],
            }
        )
    
    def print_report(self, aggregate: AggregateErrorMetrics, top_n: int = 10):
        """Print human-readable error report."""
        
        info("=" * 80)
        info("QUANTIZATION ERROR REPORT")
        info("=" * 80)
        
        # Aggregate statistics
        info("\nAGGREGATE STATISTICS:")
        info(f"  Total layers compared:       {aggregate.total_layers}")
        info(f"  Total elements:              {aggregate.total_elements:,}")
        info(f"\nERROR METRICS:")
        info(f"  Mean Absolute Error (MAE):   {aggregate.mean_absolute_error:.6e}")
        info(f"  Median Absolute Error:       {aggregate.median_absolute_error:.6e}")
        info(f"  Max Absolute Error:          {aggregate.max_absolute_error:.6e}")
        info(f"\nSQUARED ERROR METRICS:")
        info(f"  Mean Squared Error (MSE):    {aggregate.mean_squared_error:.6e}")
        info(f"  Root Mean Squared Error:     {aggregate.root_mean_squared_error:.6e}")
        info(f"\nRELATIVE ERROR:")
        info(f"  Mean Relative Error:         {aggregate.mean_relative_error:.6e}")
        info(f"  Median Relative Error:       {aggregate.median_relative_error:.6e}")
        info(f"  Max Relative Error:          {aggregate.max_relative_error:.6e}")
        info(f"\nSIGNAL QUALITY:")
        info(f"  Overall SNR (dB):            {aggregate.overall_snr_db:.2f}")
        info(f"  Original Norm:               {aggregate.total_original_norm:.6e}")
        info(f"  Quantized Norm:              {aggregate.total_quantized_norm:.6e}")
        info(f"  Error Norm:                  {aggregate.total_error_norm:.6e}")
        info(f"\nCORRELATION & FITNESS:")
        info(f"  Mean Pearson Correlation:    {aggregate.mean_pearson_correlation:.6f}")
        info(f"  Mean Cosine Similarity:      {aggregate.mean_cosine_similarity:.6f}")
        
        # Relative error percentiles
        info(f"\nRELATIVE ERROR PERCENTILES:")
        for pct, value in aggregate.relative_error_percentiles.items():
            info(f"  {pct}:  {value:.6e}")
        
        # Per-layer statistics (top N worst layers)
        if self.layer_metrics:
            info(f"\nTOP {min(top_n, len(self.layer_metrics))} WORST LAYERS (by MAE):")
            info("-" * 80)
            
            sorted_layers = sorted(self.layer_metrics, key=lambda m: m.mean_absolute_error, reverse=True)
            for i, metrics in enumerate(sorted_layers[:top_n], 1):
                info(f"\n{i}. {metrics.layer_name}")
                info(f"   Shape:                {metrics.tensor_shape}")
                info(f"   Dtype:                {metrics.original_dtype} -> {metrics.quantized_dtype}")
                info(f"   MAE:                  {metrics.mean_absolute_error:.6e}")
                info(f"   MSE:                  {metrics.mean_squared_error:.6e}")
                info(f"   RMSE:                 {metrics.root_mean_squared_error:.6e}")
                info(f"   Rel. Error:           {metrics.relative_error:.6e}")
                info(f"   Max Rel. Error:       {metrics.max_relative_error:.6e}")
                info(f"   SNR (dB):             {metrics.snr_db:.2f}")
                info(f"   Pearson Correlation:  {metrics.pearson_correlation:.6f}")
                info(f"   Cosine Similarity:    {metrics.cosine_similarity:.6f}")
                if metrics.scale_factor is not None:
                    info(f"   Scale:                {metrics.scale_factor:.6e}")
                if metrics.comfy_format:
                    info(f"   Format:               {metrics.comfy_format}")
                if metrics.block_size:
                    info(f"   Block Size:           {metrics.block_size}")
        
        info("=" * 80)
    
    def save_report(self, filepath: str, aggregate: AggregateErrorMetrics):
        """Save detailed report to JSON file."""
        
        report = {
            "aggregate": asdict(aggregate),
            "layers": [asdict(m) for m in self.layer_metrics],
        }
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        
        info(f"Report saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure quantization error between BF16 and FP8 models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic comparison
  python measure_quantization_error.py \\
    --original model_bf16.safetensors \\
    --quantized model_fp8.safetensors

  # With filtered layers
  python measure_quantization_error.py \\
    --original model_bf16.safetensors \\
    --quantized model_fp8.safetensors \\
    --layers-to-compare "attention,mlp"

  # Save detailed report
  python measure_quantization_error.py \\
    --original model_bf16.safetensors \\
    --quantized model_fp8.safetensors \\
    --output-report error_report.json
        """
    )
    
    # Setup logging first
    setup_logging("VERBOSE")
    
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original (BF16/FP32) model"
    )
    parser.add_argument(
        "--quantized",
        type=str,
        required=True,
        help="Path to quantized (FP8) model"
    )
    parser.add_argument(
        "--layers-to-compare",
        type=str,
        default=None,
        help="Comma-separated layer names to compare (e.g., 'attention,mlp')"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default=None,
        help="Path to save detailed JSON report"
    )
    parser.add_argument(
        "--top-layers",
        type=int,
        default=10,
        help="Number of worst layers to show in report (default: 10)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Device to use for computation (default: auto-detect)"
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Enable memory-efficient mode (stream tensors on-demand instead of preloading)"
    )
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.original):
        error(f"Original model not found: {args.original}")
        return 1
    
    if not os.path.exists(args.quantized):
        error(f"Quantized model not found: {args.quantized}")
        return 1
    
    # Auto-detect device if not specified
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Parse layer filter
    layer_filter = None
    if args.layers_to_compare:
        layer_filter = [l.strip() for l in args.layers_to_compare.split(",")]
    
    # Create measurer
    measurer = QuantizationErrorMeasurer(device=device, low_memory=args.low_memory)
    
    # Load models
    original_tensors, original_metadata = measurer.load_model(args.original)
    if original_tensors is None:
        return 1
    
    quantized_tensors, quantized_metadata = measurer.load_model(args.quantized)
    if quantized_tensors is None:
        return 1
    
    # Compare models
    if not measurer.compare_models(
        original_tensors,
        quantized_tensors,
        original_metadata,
        quantized_metadata,
        layer_filter=layer_filter
    ):
        return 1
    
    # Compute aggregate metrics
    aggregate = measurer.compute_aggregate_metrics()
    if aggregate is None:
        return 1
    
    # Print report
    measurer.print_report(aggregate, top_n=args.top_layers)
    
    # Save detailed report if requested
    if args.output_report:
        measurer.save_report(args.output_report, aggregate)
    
    return 0


if __name__ == "__main__":
    exit(main())
