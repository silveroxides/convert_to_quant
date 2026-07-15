"""Unit contracts for centralized quantization routing."""

from __future__ import annotations

import re

import pytest

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.routing import resolve_conversion_route, resolve_layer_route


def resolve(key="block.weight", **overrides):
    arguments = {
        "target_format": "fp8",
        "optimizer": "prodigy",
        "primary_simple": False,
        "filter_flags": {},
    }
    arguments.update(overrides)
    return resolve_layer_route(key, **arguments)


@pytest.mark.unit
def test_primary_route_carries_learned_optimizer_selection():
    route = resolve()

    assert route.action == "quantize"
    assert route.source == "primary"
    assert route.target_format == "fp8"
    assert route.mode == "learned"
    assert route.optimizer == "prodigy"


@pytest.mark.unit
def test_primary_simple_mode_is_explicit():
    assert resolve(primary_simple=True).mode == "simple"


@pytest.mark.unit
def test_layer_config_has_priority_over_custom_and_exclusion():
    settings = {"format": "int8_tensorwise", "scaling_mode": "row", "simple": True}
    route = resolve(
        "special.weight",
        layer_config={"special": settings, "_compiled_patterns": {"special": re.compile("special")}},
        custom_pattern=re.compile("special"),
        custom_type="nvfp4",
        exclude_pattern=re.compile("special"),
        fallback="mxfp8",
    )

    assert route.source == "layer_config"
    assert route.target_format == "int8"
    assert route.mode == "simple"
    assert route.layer_settings is settings


@pytest.mark.unit
def test_layer_config_without_simple_inherits_primary_mode():
    config = {
        "block": {"format": "float8_e4m3fn"},
        "_compiled_patterns": {"block": re.compile("block")},
    }

    assert resolve(layer_config=config, primary_simple=False).mode == "learned"
    assert resolve(layer_config=config, primary_simple=True).mode == "simple"


@pytest.mark.unit
def test_layer_config_cannot_disable_primary_simple_mode():
    config = {
        "block": {"format": "float8_e4m3fn", "simple": False},
        "_compiled_patterns": {"block": re.compile("block")},
    }

    assert resolve(layer_config=config, primary_simple=True).mode == "simple"


@pytest.mark.unit
def test_layer_config_skip_is_terminal():
    config = {"block": {"skip": True}, "_compiled_patterns": {"block": re.compile("block")}}
    route = resolve(layer_config=config, custom_pattern=re.compile("block"), custom_type="int8")

    assert route.action == "skip"
    assert route.source == "layer_config"


@pytest.mark.unit
def test_custom_route_overrides_exclusions_and_has_its_own_mode():
    route = resolve(
        "special.weight",
        primary_simple=True,
        custom_pattern=re.compile("special"),
        custom_type="int8",
        custom_simple=False,
        exclude_pattern=re.compile("special"),
        filter_flags={"qwen35": True},
    )

    assert route.source == "custom"
    assert route.target_format == "int8"
    assert route.mode == "learned"


@pytest.mark.unit
def test_exclusion_routes_to_fallback_with_fallback_mode():
    route = resolve(
        exclude_pattern=re.compile("block"),
        fallback="int8",
        fallback_simple=True,
    )

    assert route.action == "quantize"
    assert route.source == "fallback"
    assert route.target_format == "int8"
    assert route.mode == "simple"
    assert route.exclusion_reason == "regex exclusion (--exclude-layers)"


@pytest.mark.unit
def test_exclusion_without_fallback_skips_layer():
    route = resolve(exclude_pattern=re.compile("block"))

    assert route.action == "skip"
    assert route.source == "exclusion"


@pytest.mark.unit
def test_model_filter_records_its_reason():
    route = resolve("model.layers.0.attn.weight", filter_flags={"qwen35": True})

    assert route.action == "skip"
    assert route.source == "model_filter"
    assert route.exclusion_reason == "qwen35 skip"


@pytest.mark.unit
def test_t5_decoder_route_removes_layer_before_other_routing():
    route = resolve(
        "decoder.block.0.attn.weight",
        filter_flags={"t5xxl": True},
        custom_pattern=re.compile("decoder"),
        custom_type="int8",
    )

    assert route.action == "remove"
    assert route.source == "model_filter"


@pytest.mark.unit
def test_existing_non_fp8_layer_config_mapping_is_preserved():
    config = {"block": {"format": "nvfp4"}, "_compiled_patterns": {"block": re.compile("block")}}
    route = resolve(layer_config=config)

    assert route.target_format == "fp8"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("target_format", "scaling_mode", "expected"),
    (
        ("fp8", "tensor", "_convert_fp8"),
        ("fp8", "row", "_convert_fp8_rowwise"),
        ("fp8", "block", "_convert_fp8_block2d"),
        ("fp8", "block3d", "_convert_fp8"),
        ("int8", "tensor", "_convert_int8_tensorwise"),
        ("int8", "row", "_convert_int8_tensorwise"),
        ("int8", "block", "_convert_int8"),
    ),
)
def test_format_and_scaling_resolve_to_existing_method(target_format, scaling_mode, expected):
    route = resolve_conversion_route(
        target_format,
        scaling_mode,
        no_learned_rounding=False,
        optimizer="prodigy",
    )

    assert route.method_name == expected
    assert route.mode == "learned"
    assert route.optimizer == "prodigy"


@pytest.mark.unit
def test_conversion_route_records_simple_mode_without_changing_method():
    route = resolve_conversion_route(
        "int8",
        "row",
        no_learned_rounding=True,
        optimizer="original",
    )

    assert route.method_name == "_convert_int8_tensorwise"
    assert route.mode == "simple"
    assert route.optimizer == "original"


@pytest.mark.unit
@pytest.mark.parametrize("optimizer", ("original", "adamw", "radam", "prodigy"))
def test_learned_optimizer_dispatch_selects_existing_method(monkeypatch, optimizer):
    converter = LearnedRoundingConverter(optimizer=optimizer, no_learned_rounding=True, device="cpu")
    selected = []

    def record(*args, **kwargs):
        selected.append((args, kwargs))
        return optimizer

    monkeypatch.setattr(converter, f"_optimize_{optimizer}", record)
    assert converter._run_selected_optimizer("weight", scale="scale") == optimizer
    assert selected == [(('weight',), {"scale": "scale"})]
