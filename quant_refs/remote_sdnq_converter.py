"""
SDNQ (Stochastic Differentiable Neural Quantization) Converter.

Provides a class-based interface for SDNQ quantization, matching the project's
standard converter pattern. Operates on raw tensors.
"""
from typing import Tuple, Optional, Dict, Any
import torch

from .sdnq_math import sdnq_quantize_layer_weight
from ..utils.logging import verbose
from ..pinned_transfer import transfer_to_gpu_pinned

class SDNQConverter:
    """
    SDNQ quantization converter.
    
    Supports arbitrary bit-widths, stochastic rounding, and SVD-based
    low-rank decomposition.
    """

    def __init__(
        self,
        weights_dtype: str = "int8",
        group_size: int = 0,
        use_svd: bool = False,
        svd_rank: int = 32,
        svd_steps: int = 8,
        use_quantized_matmul: bool = False,
        use_stochastic_rounding: bool = False,
        dequantize_fp32: bool = False,
        **kwargs
    ):
        """
        Initialize SDNQ converter with settings.
        
        Args:
            weights_dtype: Target weight dtype (e.g., 'int4', 'fp8')
            group_size: Quantization group size (0 for auto)
            use_svd: Enable SVD low-rank correction
            svd_rank: Rank for SVD decomposition
            svd_steps: Iterations for SVD computation
            use_quantized_matmul: Optimize layout for quantized matmul
            use_stochastic_rounding: Enable stochastic rounding during quantization
            dequantize_fp32: Keep auxiliary tensors in FP32
        """
        self.settings = {
            "weights_dtype": weights_dtype,
            "group_size": group_size,
            "use_svd": use_svd,
            "svd_rank": svd_rank,
            "svd_steps": svd_steps,
            "use_quantized_matmul": use_quantized_matmul,
            "use_stochastic_rounding": use_stochastic_rounding,
            "dequantize_fp32": dequantize_fp32,
            **kwargs
        }
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        verbose(f"SDNQConverter initialized on device {self.device} with settings: {self.settings}")

    def quantize(
        self, weight: torch.Tensor, layer_class_name: Optional[str] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Quantize a weight tensor using SDNQ.
        
        Args:
            weight: Input weight tensor
            layer_class_name: Optional hint for layer type ('Linear', 'Conv2d', etc.)
            
        Returns:
            Tuple of (quantized_weight, scale, zero_point, svd_up, svd_down, info)
        """
        # Infer layer type if not provided
        if layer_class_name is None:
            if weight.ndim == 2:
                layer_class_name = "Linear"
            elif weight.ndim == 4:
                layer_class_name = "Conv2d"
            elif weight.ndim == 3:
                layer_class_name = "Conv1d"
            elif weight.ndim == 5:
                layer_class_name = "Conv3d"
            else:
                layer_class_name = "Unknown"

        # 1. Move to GPU for quantization
        weight_gpu = transfer_to_gpu_pinned(weight, self.device)
        
        # 2. Quantize
        qdata, scale, zero_point, svd_up, svd_down, info = sdnq_quantize_layer_weight(
            weight_gpu,
            layer_class_name=layer_class_name,
            settings=self.settings
        )
        
        # 3. Move results back to CPU for saving
        qdata = qdata.cpu()
        scale = scale.cpu()
        if zero_point is not None:
            zero_point = zero_point.cpu()
        if svd_up is not None:
            svd_up = svd_up.cpu()
            svd_down = svd_down.cpu()
            
        return qdata, scale, zero_point, svd_up, svd_down, info
