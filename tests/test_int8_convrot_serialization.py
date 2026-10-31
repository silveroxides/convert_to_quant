import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_convert_to_fp8_scaled_serializes_actual_primary_convrot_groups(tmp_path):
    input_path = tmp_path / "input.safetensors"
    output_path = tmp_path / "output.safetensors"
    tensors = {
        "blocks.0.compatible.weight": torch.randn(2, 512),
        "blocks.0.fallback64.weight": torch.randn(2, 2688),
        "blocks.0.incompatible.weight": torch.randn(2, 96),
    }
    save_file(tensors, input_path)

    convert_to_fp8_scaled(
        str(input_path),
        str(output_path),
        comfy_quant=True,
        filter_flags={},
        calib_samples=2,
        seed=1,
        int8=True,
        convrot=True,
        convrot_group_size=256,
        no_learned_rounding=True,
        scaling_mode="row",
        full_precision_matrix_mult=True,
        save_quant_metadata=True,
        low_memory=True,
        device="cpu",
    )

    converted = load_file(output_path)
    configs = {
        name: tensor_to_dict(converted[f"{name}.comfy_quant"])
        for name in ("blocks.0.compatible", "blocks.0.fallback64", "blocks.0.incompatible")
    }

    assert configs["blocks.0.compatible"] == {
        "format": "int8_tensorwise",
        "orig_dtype": "torch.bfloat16",
        "full_precision_matrix_mult": True,
        "convrot": True,
        "convrot_groupsize": 256,
        "per_row": True,
    }
    assert configs["blocks.0.fallback64"]["convrot_groupsize"] == 64
    assert configs["blocks.0.fallback64"]["full_precision_matrix_mult"] is True
    assert "convrot" not in configs["blocks.0.incompatible"]
    assert "convrot_groupsize" not in configs["blocks.0.incompatible"]

    for name in tensors:
        base = name.removesuffix(".weight")
        assert converted[f"{base}.weight_scale"].dtype == torch.float32
        assert converted[f"{base}.weight_scale"].shape == (2, 1)

    with safe_open(output_path, framework="pt", device="cpu") as handle:
        header = json.loads(handle.metadata()["_quantization_metadata"])
    layers = header["layers"]
    assert layers["blocks.0.compatible"]["convrot_groupsize"] == 256
    assert layers["blocks.0.fallback64"]["convrot_groupsize"] == 64
    assert "convrot" not in layers["blocks.0.incompatible"]
    assert layers["blocks.0.compatible"]["full_precision_matrix_mult"] is True


@pytest.mark.parametrize(
    ("mode", "simple"),
    [("custom", True), ("custom", False), ("fallback", True), ("fallback", False)],
)
def test_custom_and_fallback_convrot_do_not_use_primary_compat(tmp_path, mode, simple):
    input_path = tmp_path / f"{mode}-{simple}-input.safetensors"
    output_path = tmp_path / f"{mode}-{simple}-output.safetensors"
    layer = "blocks.0.target"
    save_file({f"{layer}.weight": torch.randn(2, 2688)}, input_path)
    mode_kwargs = {}
    if mode == "custom":
        mode_kwargs.update(
            custom_layers=r"blocks\.0\.target\.weight",
            custom_type="int8",
            custom_scaling_mode="row",
            custom_simple=simple,
            custom_convrot=True,
            custom_convrot_group_size=256,
        )
    else:
        mode_kwargs.update(
            exclude_layers=r"blocks\.0\.target\.weight",
            fallback="int8",
            fallback_simple=simple,
        )

    convert_to_fp8_scaled(
        str(input_path),
        str(output_path),
        comfy_quant=True,
        filter_flags={},
        calib_samples=2,
        seed=1,
        int8=True,
        convrot=mode == "fallback",
        convrot_group_size=256,
        no_learned_rounding=True,
        scaling_mode="row",
        save_quant_metadata=True,
        low_memory=True,
        device="cpu",
        num_iter=1,
        **mode_kwargs,
    )

    converted = load_file(output_path)
    config = tensor_to_dict(converted[f"{layer}.comfy_quant"])
    assert config["format"] == "int8_tensorwise"
    assert "convrot" not in config
    assert "convrot_groupsize" not in config


def test_primary_convrot_metadata_requires_successful_rotation(tmp_path, monkeypatch):
    input_path = tmp_path / "failed-rotation-input.safetensors"
    output_path = tmp_path / "failed-rotation-output.safetensors"
    layer = "blocks.0.failed_rotation"
    save_file({f"{layer}.weight": torch.randn(2, 512)}, input_path)

    def fail_hadamard(*args, **kwargs):
        raise RuntimeError("synthetic rotation failure")

    monkeypatch.setattr("convert_to_quant.utils.convrot.build_hadamard", fail_hadamard)

    convert_to_fp8_scaled(
        str(input_path),
        str(output_path),
        comfy_quant=True,
        filter_flags={},
        calib_samples=2,
        seed=1,
        int8=True,
        convrot=True,
        convrot_group_size=256,
        no_learned_rounding=True,
        scaling_mode="row",
        save_quant_metadata=True,
        low_memory=True,
        device="cpu",
    )

    converted = load_file(output_path)
    config = tensor_to_dict(converted[f"{layer}.comfy_quant"])
    assert "convrot" not in config
    assert "convrot_groupsize" not in config
    with safe_open(output_path, framework="pt", device="cpu") as handle:
        header = json.loads(handle.metadata()["_quantization_metadata"])
    assert "convrot" not in header["layers"][layer]
