"""Tests for the public ``quantize`` module entry point."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_to_quant import quantize


@pytest.mark.integration
def test_quantize_module_creates_a_loadable_model():
    input_path = Path("test_module_usage_input.safetensors")
    output_path = Path("test_module_usage_output.safetensors")
    try:
        generator = torch.Generator(device="cpu").manual_seed(1234)
        save_file({"layer.weight": torch.randn(16, 16, generator=generator)}, input_path)

        quantize(
            input=str(input_path),
            output=str(output_path),
            int8=True,
            scaling_mode="tensor",
            simple=True,
            device="cpu",
            calib_samples=8,
            manual_seed=42,
            save_quant_metadata=True,
        )

        assert output_path.exists()
        with safe_open(output_path, framework="pt", device="cpu") as handle:
            assert set(handle.keys()) == {
                "layer.comfy_quant",
                "layer.weight",
                "layer.weight_scale",
            }
            assert "_quantization_metadata" in (handle.metadata() or {})
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
