"""Tests for adaptive learned-rounding convergence control."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

from convert_to_quant.converters.base_converter import BaseLearnedConverter
from convert_to_quant.converters.convergence import AdaptiveConvergenceController, TuningReportCollector, convergence_window
from convert_to_quant.converters.learned_mxfp8 import LearnedMXFP8Converter
from convert_to_quant.converters.learned_nvfp4 import LearnedNVFP4Converter
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter


class _RetryConverter(BaseLearnedConverter):
    def convert(self, W_orig, key=None, depth=-1, **kwargs):
        self._active_layer_key = key
        return self._run_selected_optimizer(W_orig)

    def _optimize_original(self, weight):
        best = math.inf
        result = weight.clone()
        for iteration in range(self.num_iter):
            loss = float("nan") if self.lr > 0.2 else 1.0 / (iteration + 1)
            improved = self._check_improvement(loss, best)
            if improved:
                best = loss
                result = torch.full_like(weight, self.lr)
            if self._active_auto_controller and self._active_auto_controller.should_stop:
                break
        return result


@pytest.mark.unit
def test_convergence_window_is_shape_aware_and_bounded():
    square = convergence_window((128, 128), rank=128)
    wide = convergence_window((128, 2048), rank=128)
    large_rank = convergence_window((2048, 2048), rank=1024)

    assert 16 <= square <= 128
    assert square < wide <= 128
    assert square < large_rank <= 128


@pytest.mark.unit
def test_controller_stops_a_quiet_plateau_after_lr_reductions():
    controller = AdaptiveConvergenceController(
        shape=(16, 16), rank=16, optimizer="adamw", initial_lr=0.1, budget=128, kind="selected"
    )
    best = math.inf
    lr = 0.1
    for _ in range(96):
        improved = controller.observe(1.0, best)
        if improved:
            best = 1.0
        lr, _ = controller.update_lr(lr)
        if controller.should_stop:
            break

    assert controller.should_stop
    assert controller.summary.stop_reason == "converged_plateau"
    assert controller.lr_reductions >= 1
    assert controller.summary.iterations < controller.summary.budget


@pytest.mark.unit
def test_controller_marks_nonfinite_loss_for_retry():
    controller = AdaptiveConvergenceController(
        shape=(64, 64), rank=32, optimizer="original", initial_lr=1.0, budget=100, kind="selected"
    )
    controller.observe(1.0, math.inf)
    controller.observe(float("nan"), 1.0)

    assert controller.should_stop
    assert controller.retry_recommended
    assert controller.summary.stop_reason == "nonfinite"
    assert controller.summary.retry_reason == "nonfinite_loss"


@pytest.mark.unit
def test_dispatch_uses_one_bounded_retry_and_keeps_it_inside_budget():
    converter = _RetryConverter(optimizer="original", num_iter=96, lr=1.0, device="cpu", auto_tune=True)

    result = converter.convert(torch.zeros(4, 4), key="retry.weight")
    record = converter.get_tuning_report()["layers"][0]

    assert record["retried"]
    assert record["iterations"] <= record["budget"]
    assert len([attempt for attempt in record["attempts"] if attempt["kind"] == "retry"]) == 1
    assert torch.allclose(result, torch.full_like(result, 0.1))


@pytest.mark.unit
def test_tuning_report_collector_writes_versioned_json():
    report_path = Path("test_auto_tuning_collector.json")
    try:
        collector = TuningReportCollector(str(report_path))
        collector.add({"layer": "layer.weight", "stop_reason": "budget", "retried": False})

        payload = json.loads(report_path.read_text(encoding="utf-8"))
        assert payload["version"] == 1
        assert payload["mode"] == "auto"
        assert payload["summary"]["layers"] == 1
        assert payload["layers"][0]["layer"] == "layer.weight"
    finally:
        report_path.unlink(missing_ok=True)


@pytest.mark.integration
def test_auto_tuning_respects_budget_restores_configuration_and_reports():
    torch.manual_seed(123)
    weight = torch.randn(16, 16)
    report_path = Path("test_auto_tuning_integration.json")
    try:
        converter = LearnedRoundingConverter(
            target_format="fp8",
            scaling_mode="tensor",
            optimizer="original",
            num_iter=96,
            lr=0.1,
            lr_schedule="plateau",
            top_p=0.5,
            min_k=2,
            max_k=4,
            device="cpu",
            auto_tune=True,
            auto_tune_report=str(report_path),
        )

        qdata, scale, dequantized, extra = converter.convert(weight, key="layer.weight", has_bias=False)
        record = converter.get_tuning_report()["layers"][0]

        assert qdata.shape == weight.shape
        assert dequantized.shape == weight.shape
        assert scale.ndim == 0
        assert isinstance(extra, dict)
        assert record["iterations"] <= 96
        assert len([attempt for attempt in record["attempts"] if attempt["kind"] == "probe"]) == 3
        assert record["selected_lr"] in {0.025, 0.1, 0.4}
        assert record["best_loss"] is not None
        assert converter.num_iter == 96
        assert converter.lr == 0.1
        assert converter.lr_schedule == "plateau"
        assert json.loads(report_path.read_text(encoding="utf-8"))["summary"]["layers"] == 1
    finally:
        report_path.unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.parametrize("optimizer", ("original", "adamw", "radam", "prodigy"))
def test_auto_tuning_runs_through_every_optimizer(optimizer):
    torch.manual_seed(456)
    weight = torch.randn(8, 8)
    converter = LearnedRoundingConverter(
        target_format="fp8",
        scaling_mode="tensor",
        optimizer=optimizer,
        num_iter=4,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    qdata, _, dequantized, _ = converter.convert(weight, key=f"{optimizer}.weight", has_bias=False)

    assert qdata.shape == weight.shape
    assert dequantized.shape == weight.shape
    assert converter.get_tuning_report()["summary"]["layers"] == 1


@pytest.mark.integration
@pytest.mark.parametrize(
    ("converter_type", "shape"),
    ((LearnedMXFP8Converter, (32, 32)), (LearnedNVFP4Converter, (16, 16))),
)
def test_auto_tuning_covers_block_float_formats(converter_type, shape):
    torch.manual_seed(789)
    converter = converter_type(
        optimizer="original",
        num_iter=2,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    converter.convert(torch.randn(*shape), key=f"{converter_type.__name__}.weight")
    record = converter.get_tuning_report()["layers"][0]

    assert record["converter"] == converter_type.__name__
    assert record["iterations"] == 2
    assert record["rank"] == 4


@pytest.mark.integration
def test_auto_tuning_covers_int8_convrot_adaround():
    torch.manual_seed(246)
    weight = torch.randn(8, 16)
    calibration = torch.randn(4, 16)
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=16,
        optimizer="adamw",
        num_iter=4,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        auto_tune=True,
    )

    qdata, scale, dequantized, _ = converter.convert(
        weight,
        key="convrot.weight",
        calibration_data=calibration,
        has_bias=False,
    )
    record = converter.get_tuning_report()["layers"][0]

    assert qdata.dtype == torch.int8
    assert scale.shape == (8, 1)
    assert dequantized.shape == weight.shape
    assert record["method"] == "_optimize_int8_adaround"
