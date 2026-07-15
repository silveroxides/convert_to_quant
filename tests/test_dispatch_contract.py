"""Fast characterization of primary/custom/fallback/layer-config dispatch."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from convert_to_quant.formats import fp8_conversion


class RecordingConverter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.target_format = kwargs.get("target_format", "fp8")
        self.scaling_mode = kwargs.get("scaling_mode", "tensor")
        self.block_size = kwargs.get("block_size") or 8
        self.convrot = kwargs.get("convrot", False)
        self.convrot_group_size = kwargs.get("convrot_group_size", 256)
        self.dynamic_convrot = kwargs.get("dynamic_convrot", False)
        self.converted_keys = []
        self.__class__.instances.append(self)

    def convert(self, tensor, key=None, **kwargs):
        self.converted_keys.append((key, kwargs))
        if self.target_format == "int8":
            qdata = torch.zeros_like(tensor, dtype=torch.int8)
            if self.scaling_mode == "row":
                scale = torch.ones(tensor.shape[0], 1, dtype=torch.float32)
            elif self.scaling_mode == "tensor":
                scale = torch.tensor(1.0, dtype=torch.float32)
            else:
                scale = torch.ones(
                    tensor.shape[0] // self.block_size,
                    tensor.shape[1] // self.block_size,
                    dtype=torch.float32,
                )
        else:
            qdata = tensor.to(torch.float8_e4m3fn)
            scale = torch.tensor(1.0, dtype=torch.float32)
        return qdata, scale, tensor.float(), {}


@pytest.fixture
def recording_converters(monkeypatch):
    RecordingConverter.instances = []
    monkeypatch.setattr(fp8_conversion, "LearnedRoundingConverter", RecordingConverter)
    return RecordingConverter.instances


def run_dispatch_case(**kwargs):
    input_path = Path("test_dispatch_input.safetensors")
    output_path = Path("test_dispatch_output.safetensors")
    try:
        save_file(
            {
                "main.weight": torch.ones(8, 8),
                "special.weight": torch.ones(8, 8) * 2,
                "fallback.weight": torch.ones(8, 8) * 3,
            },
            input_path,
        )
        fp8_conversion.convert_to_fp8_scaled(
            str(input_path),
            str(output_path),
            comfy_quant=True,
            filter_flags={},
            calib_samples=2,
            seed=7,
            no_learned_rounding=True,
            save_quant_metadata=True,
            device="cpu",
            block_size=8,
            **kwargs,
        )
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


@pytest.mark.unit
def test_primary_converter_receives_complete_mode(recording_converters):
    run_dispatch_case(int8=True, scaling_mode="row", convrot=True, convrot_group_size=4)

    assert len(recording_converters) == 1
    primary = recording_converters[0]
    assert primary.target_format == "int8"
    assert primary.scaling_mode == "row"
    assert primary.convrot is True
    assert primary.convrot_group_size == 4
    assert {key for key, _ in primary.converted_keys} == {
        "fallback.weight",
        "main.weight",
        "special.weight",
    }


@pytest.mark.unit
def test_custom_converter_overrides_primary_without_inheriting_simple(recording_converters):
    run_dispatch_case(
        int8=False,
        scaling_mode="tensor",
        custom_layers="special",
        custom_type="int8",
        custom_scaling_mode="row",
        custom_simple=False,
    )

    primary, custom = recording_converters
    assert primary.target_format == "fp8"
    assert custom.target_format == "int8"
    assert custom.scaling_mode == "row"
    assert custom.kwargs["no_learned_rounding"] is False
    assert [key for key, _ in custom.converted_keys] == ["special.weight"]


@pytest.mark.unit
def test_fallback_converter_handles_only_excluded_layers(recording_converters):
    run_dispatch_case(
        int8=False,
        scaling_mode="tensor",
        exclude_layers="fallback",
        fallback="int8",
        fallback_block_size=8,
        fallback_simple=True,
    )

    primary, fallback = recording_converters
    assert fallback.target_format == "int8"
    assert fallback.kwargs["no_learned_rounding"] is True
    assert [key for key, _ in fallback.converted_keys] == ["fallback.weight"]
    assert "fallback.weight" not in {key for key, _ in primary.converted_keys}


@pytest.mark.unit
def test_layer_config_has_priority_and_constructs_its_own_converter(recording_converters):
    layer_config = {
        "special": {"format": "int8_tensorwise", "scaling_mode": "row", "simple": True},
        "_compiled_patterns": {"special": re.compile("special")},
    }
    run_dispatch_case(
        int8=False,
        scaling_mode="tensor",
        custom_layers="special",
        custom_type="fp8",
        layer_config=layer_config,
    )

    configured = [instance for instance in recording_converters if instance.target_format == "int8"]
    assert len(configured) == 1
    assert configured[0].scaling_mode == "row"
    assert configured[0].kwargs["no_learned_rounding"] is True
    assert [key for key, _ in configured[0].converted_keys] == ["special.weight"]
