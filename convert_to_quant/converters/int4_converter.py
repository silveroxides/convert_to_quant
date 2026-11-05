"""
INT4 ConvRot W4A4 converter for convert_to_quant.

Implements row-wise symmetric INT4 quantization with group-wise Hadamard rotation (ConvRot)
and 2-per-byte row-major packing for ComfyUI's convrot_w4a4 runtime format.
"""

from typing import Dict, Optional, Tuple

import torch

from ..constants import (
    COMPUTE_DTYPE,
    INT4_SYMMETRIC_MAX,
    SCALE_DTYPE,
    TARGET_INT4_DTYPE,
)
from ..pinned_transfer import transfer_to_gpu_pinned
from ..utils.convrot import (
    build_hadamard,
    find_max_compatible_group_size,
    pack_int4_row_major,
    quantize_int4_rowwise,
    rotate_weight,
)
from ..utils.logging import verbose
from ..utils.tensor_utils import prepare_calibration_data


class INT4Converter:
    """
    Converter for row-wise INT4 ConvRot (W4A4) quantization.

    Applies group-wise Hadamard rotation (static or dynamic group size),
    row-wise symmetric INT4 scaling (range [-7, 7]), and packs 2 nibbles
    per byte in row-major order.
    """

    def __init__(
        self,
        convrot: bool = True,
        convrot_group_size: int = 64,
        dynamic_convrot: bool = True,
        device: Optional[str] = None,
        no_learned_rounding: bool = True,
        calib_scale: float = 1.0,
        has_bias: bool = False,
        **kwargs,
    ):
        self.convrot = convrot
        self.convrot_group_size = convrot_group_size
        self.dynamic_convrot = dynamic_convrot
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.no_learned_rounding = no_learned_rounding
        self.calib_scale = calib_scale
        self.has_bias = has_bias
        self.block_size = None
        self.last_group_size: Optional[int] = None
        self._current_extra_tensors: Dict[str, torch.Tensor] = {}

    def _cleanup_tensors(self, *tensors):
        """Delete local tensors and free memory."""
        for t in tensors:
            if t is not None:
                del t

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        depth: int = -1,
        calibration_data: Optional[torch.Tensor] = None,
        has_bias: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Convert a 2D weight tensor to packed INT4 ConvRot W4A4.

        Args:
            W_orig: Original 2D weight tensor.
            key: Optional layer name for diagnostics.
            depth: Optional layer depth.
            calibration_data: Optional simulated/real calibration activation tensor.
            has_bias: Optional override for whether layer has a bias parameter.

        Returns:
            Tuple of (packed_qdata, scale, dequantized_weight, extra_tensors):
            - packed_qdata: torch.int8 of shape (rows, cols // 2)
            - scale: torch.float32 of shape (rows,)
            - dequantized_weight: torch.float32 of shape (rows, cols) in original (un-rotated) basis
            - extra_tensors: Dict containing bias_correction if computed
        """
        layer_has_bias = self.has_bias if has_bias is None else has_bias
        W_float32 = transfer_to_gpu_pinned(W_orig, self.device, COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            verbose("  - Tensor is all zeros, skipping optimization.")
            rows, cols = W_float32.shape
            packed_zeros = torch.zeros((rows, cols // 2), dtype=TARGET_INT4_DTYPE, device=self.device)
            scales = torch.ones(rows, dtype=SCALE_DTYPE, device=self.device)
            dequant_zeros = torch.zeros_like(W_float32)
            self.last_group_size = self.convrot_group_size
            return packed_zeros, scales, dequant_zeros, {}

        rows, N = W_float32.shape
        if N % 2 != 0:
            raise ValueError(f"INT4 ConvRot requires even in_features, got {N}")

        # Resolve group size
        layer_group_size = self.convrot_group_size
        if self.dynamic_convrot:
            min_grp = min(self.convrot_group_size, 64)
            resolved = find_max_compatible_group_size(N, min_group_size=min_grp)
            if resolved is not None:
                layer_group_size = resolved
        elif N % layer_group_size != 0 and N % 64 == 0:
            layer_group_size = 64

        if layer_group_size is None or N % layer_group_size != 0:
            raise ValueError(
                f"INT4 ConvRot W4A4 requires in_features ({N}) divisible by group_size ({layer_group_size})."
            )

        self.last_group_size = layer_group_size

        # Apply group-wise Hadamard rotation
        H = build_hadamard(layer_group_size, device=self.device, dtype=COMPUTE_DTYPE)
        W_rot = rotate_weight(W_float32, H, layer_group_size)
        verbose(f"    - Applied ConvRot Hadamard rotation (group_size={layer_group_size}).")

        # Prepare calibration activations if provided and bias adjustment is needed
        X_rot = None
        Y_ref = None
        if layer_has_bias and calibration_data is not None:
            X_rot, Y_ref, _ = prepare_calibration_data(
                W_rot,
                calibration_data,
                True,
                layer_group_size,
                self.device,
                COMPUTE_DTYPE,
                calib_scale=self.calib_scale,
            )

        # Simple row-wise INT4 quantization: scale = amax / 7.0, values in [-7, 7]
        qdata, scale = quantize_int4_rowwise(W_rot)

        # Dequantize in rotated space: dq_rot = qdata * scale
        scale_broadcast = scale.unsqueeze(1)
        dequantized_rotated = qdata.to(COMPUTE_DTYPE) * scale_broadcast

        # Un-rotate to return dequantized weights in the original basis:
        # Since H is symmetric and orthogonal (involution), rotate_weight with H inverts the rotation.
        dequantized = rotate_weight(dequantized_rotated, H, layer_group_size)

        extra_tensors: Dict[str, torch.Tensor] = {}

        # Residual bias calibration (if layer has bias and calibration data exists)
        if layer_has_bias and X_rot is not None and Y_ref is not None:
            with torch.no_grad():
                Y_quant = X_rot @ dequantized_rotated.T
                bias_adj = (Y_ref - Y_quant).mean(dim=0)
                extra_tensors["bias_correction"] = bias_adj.cpu()
                verbose(
                    f"    - Phase 4: Residual Bias Calibration (bias correction norm: {bias_adj.norm().item():.6f})"
                )

        # Pack signed int4 (in int8 container) 2-per-byte row-major
        packed = pack_int4_row_major(qdata)

        self._cleanup_tensors(W_float32, W_rot, X_rot, Y_ref, dequantized_rotated, H)

        return (
            packed,
            scale.to(device=self.device, dtype=SCALE_DTYPE),
            dequantized,
            extra_tensors,
        )
