"""Early gathering must omit only weights that cannot reach a converter."""

import pytest
import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.utils.memory_efficient_loader import MemoryEfficientSafeOpen


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize(
    "options,quantized,gathered",
    [
        ({"filter_flags": {}}, True, True),
        ({"filter_flags": {"mistral": False}}, True, True),
        ({"filter_flags": {"generic_text": True}}, True, True),
        ({"filter_flags": {"mistral": True}}, False, False),
        ({"exclude_layers": "norm"}, False, False),
        ({"custom_layers": "norm", "custom_type": "int8", "custom_simple": True}, True, True),
        ({"fallback": "int8", "fallback_simple": True}, True, True),
        ({"layer_config": {"norm": {"format": "int8_tensorwise", "simple": True}}}, True, True),
        ({"layer_config": {"_default": {"format": "int8_tensorwise", "simple": True}}}, True, True),
        ({"layer_config": {"norm": {"skip": True}}, "fallback": "int8", "fallback_simple": True}, False, True),
        ({"layer_config": {"norm": {"skip": True}}, "custom_layers": "norm", "custom_type": "int8", "custom_simple": True}, False, True),
        ({"layer_config": {"other": {"skip": True}}}, False, False),
        ({"layer_config": {"model.norm": {"format": "int8_tensorwise", "simple": True}}, "layer_config_fullmatch": True}, True, True),
        ({"layer_config": {"norm": {"format": "int8_tensorwise", "simple": True}}, "layer_config_fullmatch": True}, False, False),
    ],
)
def test_gathering_respects_exclusions_and_overrides(tmp_path, monkeypatch, low_memory, options, quantized, gathered):
    tensors = {
        "model.norm.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        "model.attn.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        "model.norm.bias": torch.randn(16),
        "model.norm_vector.weight": torch.randn(16),
        "model.conv.weight": torch.randn(2, 2, 2),
        "decoder.attn.weight": torch.randn(16, 16, dtype=torch.bfloat16),
    }
    input_path = str(tmp_path / "input.safetensors")
    output_path = str(tmp_path / "output.safetensors")
    save_file(tensors, input_path, metadata={"source": "regression"})
    ndim_keys = []
    original_get_ndim = MemoryEfficientSafeOpen.get_ndim

    def track_ndim(loader, key):
        ndim_keys.append(key)
        return original_get_ndim(loader, key)

    monkeypatch.setattr(MemoryEfficientSafeOpen, "get_ndim", track_ndim)
    kwargs = dict(
        comfy_quant=True, filter_flags={"mistral": True}, calib_samples=2,
        seed=0, no_learned_rounding=True, save_quant_metadata=False,
        device="cpu", scaling_mode="tensor", low_memory=low_memory,
        output_dtype="bfloat16",
    )
    kwargs.update(options)
    convert_to_fp8_scaled(input_path, output_path, **kwargs)
    output = load_file(output_path)

    assert ("model.norm.weight" in ndim_keys) == gathered
    assert ("model.norm.weight_scale" in output) == quantized
    assert ("model.norm.comfy_quant" in output) == quantized
    if not quantized:
        assert torch.equal(output["model.norm.weight"], tensors["model.norm.weight"])
        assert not any(key.startswith("model.norm.") and key not in tensors for key in output)
    else:
        expected_dtype = torch.int8 if any(k in options for k in ("custom_type", "fallback", "layer_config")) else torch.float8_e4m3fn
        assert output["model.norm.weight"].dtype == expected_dtype
    assert "model.attn.weight_scale" in output
    preserved_keys = ["model.norm_vector.weight", "model.conv.weight"]
    if not quantized:
        preserved_keys.append("model.norm.bias")
    for key in preserved_keys:
        assert torch.equal(output[key], tensors[key])


@pytest.mark.parametrize("filter_name", ["qwen_vlm", "qwen35"])
def test_dynamic_final_layer_exclusion(tmp_path, monkeypatch, filter_name):
    tensors = {f"model.layers.{i}.attn.weight": torch.randn(16, 16) for i in (0, 2, 7)}
    input_path = str(tmp_path / "input.safetensors")
    output_path = str(tmp_path / "output.safetensors")
    save_file(tensors, input_path)
    original_get_ndim = MemoryEfficientSafeOpen.get_ndim

    def check_ndim(loader, key):
        assert key == "model.layers.2.attn.weight"
        return original_get_ndim(loader, key)

    monkeypatch.setattr(MemoryEfficientSafeOpen, "get_ndim", check_ndim)
    convert_to_fp8_scaled(
        input_path, output_path, comfy_quant=True, filter_flags={filter_name: True},
        calib_samples=2, seed=0, no_learned_rounding=True, device="cpu",
        save_quant_metadata=False, output_dtype="bfloat16",
    )
    output = load_file(output_path)
    assert "model.layers.2.attn.weight_scale" in output
    for i in (0, 7):
        key = f"model.layers.{i}.attn.weight"
        assert output[key].dtype == torch.bfloat16  # Qwen's existing preserve policy
        assert torch.equal(output[key], tensors[key].bfloat16())
        assert f"model.layers.{i}.attn.weight_scale" not in output
