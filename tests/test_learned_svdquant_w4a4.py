# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LearnedSVDQuantW4A4Converter.

Covers:
- Learned path produces lower reconstruction error than simple (round-to-nearest)
- Output shapes and dtypes match SVDQuantW4A4Converter (same signature)
- no_learned_rounding=True degrades to simple quantization
- All four optimizers run without error and produce valid int8 output
- All three LR schedules run without error (exponential, plateau, adaptive)
- Gradient scaling: original optimizer actually moves the loss downward
- Early stopping: stall counter triggers correctly
- All-zeros tensor shortcut
- Invalid smooth mode / group size raise at construction
"""
from __future__ import annotations

import pytest
import torch

from convert_to_quant.converters.learned_svdquant_w4a4 import LearnedSVDQuantW4A4Converter
from convert_to_quant.converters.svdquant_w4a4_converter import (
    _unpack_int4_weight,
    _INT4_QMAX,
)

# Tiny iteration counts so tests run fast.
_FAST = dict(num_iter=10, early_stop_stall=9999, early_stop_loss=0.0, early_stop_lr=0.0)


def _make_converter(**kwargs) -> LearnedSVDQuantW4A4Converter:
    defaults = dict(rank=4, smooth_mode="ones", optimizer="original", **_FAST)
    defaults.update(kwargs)
    return LearnedSVDQuantW4A4Converter(**defaults)


def _make_weight(N=32, K=64, seed=0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(N, K) * 0.3


# ---------------------------------------------------------------------------
# Output shape / dtype contract
# ---------------------------------------------------------------------------

class TestOutputContract:
    @pytest.mark.parametrize("N,K,R", [(32, 64, 4), (64, 128, 8)])
    def test_shapes_match_simple_converter(self, N, K, R):
        W = _make_weight(N, K)
        conv = _make_converter(rank=R)
        qweight, wscales, proj_down, proj_up, smooth_factor, dequant = conv.convert(W)

        assert qweight.shape    == (N, K // 2)
        assert qweight.dtype    == torch.int8
        assert wscales.shape    == (K // 64, N)
        assert wscales.dtype    == torch.bfloat16
        assert proj_down.shape  == (K, R)
        assert proj_up.shape    == (N, R)
        assert smooth_factor.shape == (K,)
        assert dequant.shape    == (N, K)
        assert dequant.dtype    == torch.bfloat16

    def test_quantizer_never_emits_neg8(self):
        W = _make_weight(64, 128) * 50.0
        conv = _make_converter(rank=0)
        qweight, _, _, _, _, _ = conv.convert(W)
        unpacked = _unpack_int4_weight(qweight, 128)
        assert unpacked.min().item() >= -_INT4_QMAX
        assert unpacked.max().item() <= _INT4_QMAX

    def test_all_zeros_shortcut(self):
        W = torch.zeros(32, 64)
        conv = _make_converter()
        qweight, wscales, proj_down, proj_up, smooth_factor, dequant = conv.convert(W)
        assert qweight.shape == (32, 32)
        assert torch.all(qweight == 0)

    def test_no_learned_rounding_flag(self):
        """no_learned_rounding=True must behave like simple round-to-nearest."""
        torch.manual_seed(1)
        W = _make_weight(32, 64)
        conv_simple = _make_converter(rank=0, no_learned_rounding=True)
        conv_rtn    = LearnedSVDQuantW4A4Converter.__new__(LearnedSVDQuantW4A4Converter)
        # Use _simple_quantize path directly via no_learned_rounding flag
        q1, _, _, _, _, _ = conv_simple.convert(W.clone())
        from convert_to_quant.converters.svdquant_w4a4_converter import SVDQuantW4A4Converter
        q2, _, _, _, _, _ = SVDQuantW4A4Converter(rank=0, smooth_mode="ones").convert(W.clone())
        assert torch.equal(q1, q2)


# ---------------------------------------------------------------------------
# Reconstruction quality
# ---------------------------------------------------------------------------

class TestQuality:
    def test_learned_reduces_svd_projected_loss(self):
        """The optimizer must reduce the SVD-projected loss that it minimises.

        Measures ||U_k.T @ (W_dq - W) @ Vh_k.T||_F after learned rounding
        versus round-to-nearest. Uses a small tensor so 500 iterations run
        quickly in CI; real-world usage is typically 2000-4000 iterations.

        rank=0 so the full reconstruction is the int4 quantised weight alone,
        isolating the optimiser contribution from the SVD correction.
        """
        torch.manual_seed(42)
        # Small enough for fast CI, large enough for meaningful statistics.
        W = _make_weight(32, 64)

        from convert_to_quant.converters.svdquant_w4a4_converter import (
            SVDQuantW4A4Converter,
            _INT4_GROUP_SIZE,
        )

        G = _INT4_GROUP_SIZE
        U, _, Vh = torch.linalg.svd(W, full_matrices=False)
        k = min(16, U.shape[1])
        U_k, Vh_k = U[:, :k], Vh[:k, :]

        def svd_loss(W_dq: torch.Tensor) -> float:
            err = W_dq.float() - W
            return (U_k.T @ err @ Vh_k.T).norm().item()

        # Baseline: round-to-nearest (simple converter)
        _, _, _, _, _, dq_simple = SVDQuantW4A4Converter(
            rank=0, smooth_mode="ones"
        ).convert(W.clone())

        # Learned: 500 iterations — representative lower bound for real usage
        conv = LearnedSVDQuantW4A4Converter(
            rank=0, smooth_mode="ones", optimizer="original",
            num_iter=500,
            early_stop_stall=9999,
            early_stop_loss=0.0,
            early_stop_lr=0.0,
        )
        _, _, _, _, _, dq_learned = conv.convert(W.clone())

        loss_simple  = svd_loss(dq_simple)
        loss_learned = svd_loss(dq_learned)

        assert loss_learned < loss_simple, (
            f"Learned SVD-loss ({loss_learned:.4f}) should be lower than "
            f"simple ({loss_simple:.4f}) after 500 iters"
        )


# ---------------------------------------------------------------------------
# All four optimizers run without error
# ---------------------------------------------------------------------------

class TestOptimizers:
    @pytest.mark.parametrize("opt", ["original", "adamw", "radam"])
    def test_optimizer_runs(self, opt):
        W = _make_weight(32, 64)
        conv = _make_converter(optimizer=opt, rank=4)
        qweight, wscales, _, _, _, dequant = conv.convert(W)
        assert qweight.shape == (32, 32)
        assert torch.isfinite(dequant.float()).all()

    def test_prodigy_runs(self):
        pytest.importorskip("prodigyplus", reason="prodigy-plus-schedule-free not installed")
        W = _make_weight(32, 64)
        conv = _make_converter(optimizer="prodigy", rank=4)
        qweight, _, _, _, _, dequant = conv.convert(W)
        assert qweight.shape == (32, 32)
        assert torch.isfinite(dequant.float()).all()


# ---------------------------------------------------------------------------
# All three LR schedules run without error
# ---------------------------------------------------------------------------

class TestSchedules:
    @pytest.mark.parametrize("schedule", ["exponential", "plateau", "adaptive"])
    def test_schedule_runs(self, schedule):
        W = _make_weight(32, 64)
        conv = _make_converter(optimizer="original", lr_schedule=schedule, rank=4)
        qweight, _, _, _, _, _ = conv.convert(W)
        assert qweight.shape == (32, 32)


# ---------------------------------------------------------------------------
# Gradient scaling — original optimizer
# ---------------------------------------------------------------------------

class TestGradient:
    def test_loss_decreases_over_iterations(self):
        """With enough iterations the loss should drop from iter 0."""
        torch.manual_seed(7)
        W = _make_weight(64, 128) * 0.5

        losses = []

        # Monkey-patch pbar to capture losses
        import tqdm as tqdm_mod
        original_tqdm = tqdm_mod.tqdm

        class CaptureTqdm(original_tqdm):
            def set_postfix(self, d=None, **kwargs):
                if d and "loss" in d:
                    losses.append(float(d["loss"]))
                super().set_postfix(d, **kwargs)

        import convert_to_quant.converters.learned_svdquant_w4a4 as mod
        orig = mod.tqdm
        mod.tqdm = CaptureTqdm
        try:
            conv = LearnedSVDQuantW4A4Converter(
                rank=4, smooth_mode="ones", optimizer="original",
                num_iter=20, early_stop_stall=9999,
                early_stop_loss=0.0, early_stop_lr=0.0,
            )
            conv.convert(W)
        finally:
            mod.tqdm = orig

        assert len(losses) >= 2
        assert losses[-1] < losses[0], (
            f"Loss should decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# Early stopping via stall counter
# ---------------------------------------------------------------------------

class TestEarlyStopping:
    def test_stall_triggers_early_stop(self):
        """early_stop_stall=2 should stop well before num_iter=1000."""
        torch.manual_seed(3)
        W = _make_weight(32, 64)

        iters_run = []
        import tqdm as tqdm_mod

        class CountTqdm(tqdm_mod.tqdm):
            def __init__(self, iterable=None, *args, **kwargs):
                self._iter_count = 0
                super().__init__(iterable, *args, **kwargs)

            def __iter__(self):
                for x in super().__iter__():
                    self._iter_count += 1
                    iters_run.append(self._iter_count)
                    yield x

        import convert_to_quant.converters.learned_svdquant_w4a4 as mod
        orig = mod.tqdm
        mod.tqdm = CountTqdm
        try:
            conv = LearnedSVDQuantW4A4Converter(
                rank=0, smooth_mode="ones", optimizer="original",
                num_iter=1000,
                early_stop_stall=2,
                early_stop_loss=0.0,
                early_stop_lr=0.0,
                lr=1e-6,   # tiny LR so loss won't improve
            )
            conv.convert(W)
        finally:
            mod.tqdm = orig

        assert len(iters_run) > 0, "No iterations were recorded"
        assert max(iters_run) < 100, (
            f"Expected early stop well before 1000 iters, ran {max(iters_run)}"
        )


# ---------------------------------------------------------------------------
# Construction validation
# ---------------------------------------------------------------------------

class TestValidation:
    def test_invalid_smooth_mode(self):
        with pytest.raises(ValueError, match="smooth_mode"):
            LearnedSVDQuantW4A4Converter(smooth_mode="bad", **_FAST)

    def test_k_not_divisible_raises(self):
        W = torch.randn(32, 96)   # 96 % 64 != 0
        conv = _make_converter()
        with pytest.raises(ValueError, match="not divisible"):
            conv.convert(W)
