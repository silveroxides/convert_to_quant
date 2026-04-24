# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SVDQuant W4A4 weight converter.

Produces the five per-layer tensors that TensorCoreSVDQuantW4A4Layout expects
at inference time:

    qweight        (N, K // 2) int8      packed signed int4 residual
    wscales        (K // 64, N) bf16     per-group weight scales, transposed
    proj_down      (K, R) bf16           SVD low-rank down projection (V^T)
    proj_up        (N, R) bf16           SVD low-rank up projection (U diag(S))
    smooth_factor  (K,) bf16             per-channel input divisor

Nibble packing convention (must match svdquant_utils.cuh::pack_int4_pair):
    byte & 0x0F  = q[n, 2k]    (even column  -> low  nibble)
    byte >> 4    = q[n, 2k+1]  (odd  column  -> high nibble)
    packed byte  = (lo & 0x0F) | ((hi & 0x0F) << 4)

This is opposite to convert_to_quant's FP4 hi_first=True convention.
Do NOT use pack_uint4() from float_utils here.

Smooth factor modes
-------------------
"ones"        -- identity (no smoothing). Simplest; lowest quality for
                 outlier-heavy channels.
"weight_only" -- channel-wise power-law heuristic:
                 smooth[k] = W.abs().amax(dim=0)[k] ** smooth_alpha
                 Standard SmoothQuant approximation when real activation
                 statistics are unavailable. Default alpha=0.5.
"external"    -- caller supplies a pre-computed (K,) tensor, e.g. from a
                 calibration run or a reference nunchaku checkpoint.

The real SVDQuant pipeline (DeepCompressor / nunchaku) derives smooth from
activation statistics. This converter intentionally does not run forward
passes -- it is an offline weight-only tool.
"""
from __future__ import annotations

import gc
import logging
from typing import Optional, Tuple, Dict

import torch

logger = logging.getLogger(__name__)

# Group size is hardcoded in the kitchen CUDA kernels (svdquant_utils.cuh).
# Do not make this configurable -- changing it requires a kernel recompile.
_INT4_GROUP_SIZE = 64

# Signed int4 emission range. -8 is representable but intentionally not
# emitted to preserve dequant symmetry (matches nunchaku gemm_w4a4.cuh:435
# and kitchen svdquant_utils.cuh:20).
_INT4_QMAX = 7
_UINT4_QMAX = 15


def _pack_int4_weight(q_vals: torch.Tensor) -> torch.Tensor:
    """Pack (N, K) int8 values in [-7,7] to (N, K//2) int8 nibble pairs.

    Packing: even column -> low nibble, odd column -> high nibble.
    This matches svdquant_utils.cuh::pack_int4_pair and the kitchen
    quantize_svdquant_w4a4 CUDA kernel.
    """
    N, K = q_vals.shape
    assert K % 2 == 0, f"K={K} must be even for int4 packing"
    q32 = q_vals.to(torch.int32)
    lo = q32[:, 0::2] & 0x0F   # even columns -> low  nibble
    hi = q32[:, 1::2] & 0x0F   # odd  columns -> high nibble
    packed = (lo | (hi << 4)).to(torch.int8)  # (N, K//2)
    return packed


def _unpack_int4_weight(packed: torch.Tensor, K: int) -> torch.Tensor:
    """Unpack (N, K//2) int8 to (N, K) int8 signed values.

    Inverse of _pack_int4_weight. Used for dequantization.
    """
    p32 = packed.to(torch.int32)
    lo4 = p32 & 0x0F
    hi4 = (p32 >> 4) & 0x0F
    # Sign-extend from 4 bits
    lo = torch.where(lo4 >= 8, lo4 - 16, lo4).to(torch.int8)
    hi = torch.where(hi4 >= 8, hi4 - 16, hi4).to(torch.int8)
    # Interleave: even cols from lo, odd cols from hi
    N = packed.shape[0]
    out = torch.empty(N, K, dtype=torch.int8, device=packed.device)
    out[:, 0::2] = lo
    out[:, 1::2] = hi
    return out


class SVDQuantW4A4Converter:
    """Offline SVDQuant W4A4 weight quantizer.

    Quantizes a single float weight tensor to the five-tensor layout that
    TensorCoreSVDQuantW4A4Layout loads at inference time.

    Args:
        rank: SVD rank for the low-rank LoRA correction (default 32).
            Set to 0 to disable the correction (plain symmetric int4 with
            smoothing only). Clamped to min(N, K) for small layers.
        group_size: Per-group quantization block size. Must be 64 (hardcoded
            in kitchen CUDA kernels). Do not change.
        compute_dtype: Dtype for intermediate computation and output tensors.
            Must be bf16 or fp16 (wscales must stay in model compute dtype).
        smooth_mode: How to compute the smooth factor.
            "weight_only" (default) -- power-law channel heuristic.
            "ones"                  -- no smoothing (identity).
            "external"              -- caller supplies smooth via convert().
        smooth_alpha: Power for "weight_only" mode (default 0.5, standard
            SmoothQuant geometric-mean approximation).
        device: Override device ("cuda" / "cpu"). Auto-detected if None.
    """

    def __init__(
        self,
        rank: int = 32,
        group_size: int = _INT4_GROUP_SIZE,
        compute_dtype: torch.dtype = torch.bfloat16,
        smooth_mode: str = "weight_only",
        smooth_alpha: float = 0.5,
        device: Optional[str] = None,
    ):
        if group_size != _INT4_GROUP_SIZE:
            raise ValueError(
                f"group_size must be {_INT4_GROUP_SIZE} (hardcoded in kitchen kernels). "
                f"Got {group_size}."
            )
        valid_smooth = ("weight_only", "ones", "external")
        if smooth_mode not in valid_smooth:
            raise ValueError(
                f"smooth_mode must be one of {valid_smooth}. Got '{smooth_mode}'."
            )
        if compute_dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(
                "compute_dtype must be bfloat16 or float16 (wscales must stay in "
                f"model compute dtype). Got {compute_dtype}."
            )

        self.rank = rank
        self.group_size = group_size
        self.compute_dtype = compute_dtype
        self.smooth_mode = smooth_mode
        self.smooth_alpha = smooth_alpha

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        W_orig: torch.Tensor,
        key: Optional[str] = None,
        smooth: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,   # qweight        (N, K//2) int8
        torch.Tensor,   # wscales        (K//64, N) bf16  transposed
        torch.Tensor,   # proj_down      (K, R) bf16
        torch.Tensor,   # proj_up        (N, R) bf16
        torch.Tensor,   # smooth_factor  (K,) bf16
        torch.Tensor,   # dequantized    (N, K) bf16  for bias correction
    ]:
        """Quantize a single float weight tensor to SVDQuant W4A4 format.

        Args:
            W_orig: (N, K) float weight tensor (any float dtype).
            key: Optional layer key name for logging.
            smooth: (K,) pre-computed smooth tensor. Required when
                smooth_mode="external"; ignored otherwise.

        Returns:
            Six tensors: qweight, wscales, proj_down, proj_up,
            smooth_factor, dequantized.
        """
        N, K = W_orig.shape
        label = key or f"({N}, {K})"

        if K % self.group_size != 0:
            raise ValueError(
                f"Layer {label}: K={K} is not divisible by group_size={self.group_size}. "
                "Zero-pad the weight before converting, or skip this layer."
            )

        # Move to compute device
        W = W_orig.to(device=self.device, dtype=torch.float32)

        # ---- Step 1: smooth factor ----------------------------------------
        smooth_factor = self._compute_smooth(W, smooth, label)

        # ---- Step 2: apply smooth to weight ---------------------------------
        # W_smoothed = W * smooth[None, :]
        # At inference: activation / smooth is fed to the quantize kernel,
        # so W * smooth balances the per-channel magnitude distribution.
        W_smoothed = W * smooth_factor.to(dtype=torch.float32)[None, :]

        # ---- Step 3: SVD low-rank factorization ----------------------------
        effective_rank = min(self.rank, min(N, K))
        if effective_rank < self.rank:
            logger.warning(
                "Layer %s: requested rank=%d clamped to %d (min(N,K)=%d)",
                label, self.rank, effective_rank, min(N, K)
            )

        if effective_rank > 0:
            # SVD is computed on W (original, unsmoothed weight) because the LoRA branch
            # runs on raw unsmoothed activations at inference time:
            #   lora_act = x @ proj_down   (kernel applies LoRA before smooth, on raw x)
            #   quant_path = quantize(x / smooth) @ W_residual_int4
            # Using W_smoothed here would produce proj_down columns that expect x/smooth,
            # but the kernel feeds raw x — causing a systematic scale error that corrupts
            # output. The residual is still taken from W_smoothed so that the int4 path
            # is quantizing the smooth-adjusted remainder (lower outlier magnitude).
            proj_up, proj_down = self._compute_svd_correction(
                W, effective_rank, label
            )
            W_residual = W_smoothed - (proj_up.float() @ proj_down.float().T)
        else:
            # rank=0: disable LoRA correction, quantize smoothed weight directly
            proj_up = torch.zeros(N, 0, dtype=self.compute_dtype, device=self.device)
            proj_down = torch.zeros(K, 0, dtype=self.compute_dtype, device=self.device)
            W_residual = W_smoothed

        # ---- Step 4: per-group absmax int4 quantization of residual ---------
        qweight, wscales, W_dequant = self._quantize_int4(
            W_residual, proj_up, proj_down
        )

        # ---- Step 5: move outputs to CPU and compute dtype ------------------
        result = (
            qweight.cpu(),
            wscales.to(self.compute_dtype).cpu(),
            proj_down.to(self.compute_dtype).cpu(),
            proj_up.to(self.compute_dtype).cpu(),
            smooth_factor.to(self.compute_dtype).cpu(),
            W_dequant.to(self.compute_dtype).cpu(),
        )

        # Cleanup
        del W, W_smoothed, W_residual
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_smooth(
        self,
        W: torch.Tensor,
        external: Optional[torch.Tensor],
        label: str,
    ) -> torch.Tensor:
        """Compute or validate the (K,) smooth factor tensor."""
        N, K = W.shape
        if self.smooth_mode == "external":
            if external is None:
                raise ValueError(
                    f"Layer {label}: smooth_mode='external' requires a smooth tensor."
                )
            s = external.to(device=self.device, dtype=torch.float32)
            if s.shape != (K,):
                raise ValueError(
                    f"Layer {label}: smooth shape {s.shape} does not match K={K}."
                )
            # Clamp to avoid divide-by-zero when smooth is applied to activations
            return s.clamp(min=1e-6)

        if self.smooth_mode == "ones":
            return torch.ones(K, dtype=torch.float32, device=self.device)

        # "weight_only": per-channel power-law heuristic
        # smooth[k] = max(|W[:, k]|) ^ alpha
        # Motivation: when alpha=0.5 this is the geometric mean of weight-max
        # and (hypothetical) uniform activation-max, which is the standard
        # SmoothQuant approximation when activation stats are unavailable.
        col_absmax = W.abs().amax(dim=0)  # (K,)
        smooth = col_absmax.pow(self.smooth_alpha).clamp(min=1e-6)
        return smooth

    def _compute_svd_correction(
        self,
        W_smoothed: torch.Tensor,
        rank: int,
        label: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute (proj_up, proj_down) via truncated SVD of W_smoothed.

        Returns:
            proj_up   (N, R) float32 -- U @ diag(S[:R])
            proj_down (K, R) float32 -- Vh[:R, :].T
        """
        try:
            # svd_lowrank is faster and sufficient for large matrices.
            # niter=4 is standard; q = rank + 16 oversampling improves accuracy.
            U, S, Vh = torch.svd_lowrank(W_smoothed, q=rank + 16, niter=4)
            # U: (N, rank+16), S: (rank+16,), Vh: (K, rank+16)
            U = U[:, :rank]
            S = S[:rank]
            Vh = Vh[:, :rank]  # svd_lowrank returns Vh transposed relative to linalg.svd
        except Exception:
            logger.debug("Layer %s: svd_lowrank failed, falling back to linalg.svd", label)
            U_full, S_full, Vh_full = torch.linalg.svd(W_smoothed, full_matrices=False)
            U = U_full[:, :rank]
            S = S_full[:rank]
            Vh = Vh_full[:rank, :].T  # (K, rank)

        proj_up = (U * S).to(torch.float32).contiguous()    # (N, R) -- absorbs singular values
        proj_down = Vh.to(torch.float32).contiguous()        # (K, R)
        return proj_up, proj_down

    def _quantize_int4(
        self,
        W_residual: torch.Tensor,
        proj_up: torch.Tensor,
        proj_down: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-group absmax symmetric int4 quantization of W_residual.

        Args:
            W_residual: (N, K) float32 residual weight.
            proj_up:    (N, R) float32 LoRA up-projection (for dequantized reconstruction).
            proj_down:  (K, R) float32 LoRA down-projection.

        Returns:
            qweight:    (N, K//2) int8 nibble-packed.
            wscales:    (K//64, N) float32 (caller casts to compute_dtype).
            W_dequant:  (N, K) float32 reconstructed weight for bias correction.
        """
        N, K = W_residual.shape
        G = self.group_size  # 64

        # Reshape to blocks for absmax computation
        W_groups = W_residual.view(N, K // G, G)          # (N, K/G, G)
        absmax = W_groups.abs().amax(dim=-1)               # (N, K/G)

        # Scale: absmax / 7  (signed symmetric, -8 never emitted by design)
        wscales = (absmax / _INT4_QMAX).clamp(min=1e-8)   # (N, K/G) float32

        # Quantize
        W_scaled = W_groups / wscales.unsqueeze(-1)        # (N, K/G, G)
        q_vals = W_scaled.round().clamp(-_INT4_QMAX, _INT4_QMAX).to(torch.int8)
        q_vals = q_vals.view(N, K)                         # (N, K) int8

        # Nibble-pack: even col -> low nibble, odd col -> high nibble
        qweight = _pack_int4_weight(q_vals)                # (N, K//2) int8

        # Dequantized reconstruction for bias correction
        # W_dequant ≈ W_orig (residual component + low-rank component)
        q_dq = q_vals.float() * wscales.view(N, K // G, 1).expand_as(
            W_groups
        ).reshape(N, K)                                    # (N, K) float32

        if proj_up.shape[1] > 0:
            lora_contribution = (proj_up.float() @ proj_down.float().T)  # (N, K)
            W_dequant = q_dq + lora_contribution
        else:
            W_dequant = q_dq

        # Transpose wscales to storage layout: (K//G, N)
        wscales_T = wscales.T.contiguous()                 # (K//G, N)

        return qweight, wscales_T, W_dequant
