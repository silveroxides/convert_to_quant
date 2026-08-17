import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_to_quant.cli.main import get_parser
from convert_to_quant.formats.flux_klein_convrot import (
    convert_flux_klein_convrot_hq,
    inspect_flux_klein_checkpoint,
)


def _write_policy(path, layers, default_mode="int8"):
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "name": "unit-test-policy",
                "default_mode": default_mode,
                "layers": layers,
                "metadata": {"calibration": "synthetic"},
            }
        ),
        encoding="utf-8",
    )


def _decode_config(handle, layer):
    return json.loads(bytes(handle.get_tensor(f"{layer}.comfy_quant").tolist()))


def test_flux_klein_cli_arguments_are_available():
    args = get_parser().parse_args(
        [
            "-i",
            "klein.safetensors",
            "--flux-klein-convrot-hq",
            "--flux-klein-precision",
            "int4",
            "--flux-klein-policy",
            "policy.json",
        ]
    )
    assert args.flux_klein_convrot_hq is True
    assert args.flux_klein_precision == "int4"
    assert args.flux_klein_policy == "policy.json"
    assert args.flux_klein_w4_activation == "int8"
    assert args.flux_klein_int4_layout == "w4a8"
    assert args.flux_klein_int4_group_size == 16


def test_mixed_streaming_conversion_writes_native_comfy_layouts(tmp_path):
    source = tmp_path / "flux-klein-bf16.safetensors"
    output = tmp_path / "flux-klein-mixed.safetensors"
    report = tmp_path / "report.json"
    policy = tmp_path / "policy.json"
    w8 = "double_blocks.0.img_attn.qkv"
    w4 = "double_blocks.0.img_attn.proj"
    bf16 = "single_blocks.0.linear1"
    passthrough = "final_layer.linear.weight"
    tensors = {
        f"{w8}.weight": torch.randn(16, 256, dtype=torch.bfloat16),
        f"{w4}.weight": torch.randn(16, 256, dtype=torch.bfloat16),
        f"{bf16}.weight": torch.randn(16, 256, dtype=torch.bfloat16),
        f"{w8}.bias": torch.randn(16, dtype=torch.bfloat16),
        passthrough: torch.randn(8, 256, dtype=torch.bfloat16),
    }
    save_file(tensors, source, metadata={"license": "test"})
    _write_policy(policy, {w4: "int4", bf16: "bf16"})

    result = convert_flux_klein_convrot_hq(
        str(source),
        str(output),
        str(report),
        policy_path=str(policy),
        default_mode="int8",
        scale_search="absmax",
        w4_activation="int8",
        device="cpu",
        strict_architecture=False,
    )

    with safe_open(output, framework="pt", device="cpu") as converted:
        keys = set(converted.keys())
        assert converted.get_tensor(f"{w8}.weight").shape == (16, 256)
        assert converted.get_tensor(f"{w8}.weight").dtype == torch.int8
        assert converted.get_tensor(f"{w8}.weight_scale").shape == (16, 1)
        assert _decode_config(converted, w8) == {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 256,
        }

        assert converted.get_tensor(f"{w4}.weight").shape == (16, 128)
        assert converted.get_tensor(f"{w4}.weight").dtype == torch.int8
        assert converted.get_tensor(f"{w4}.weight_s_rel").shape == (16, 16)
        assert converted.get_tensor(f"{w4}.weight_s_rel").dtype == torch.float8_e4m3fn
        assert converted.get_tensor(f"{w4}.weight_s_channel").shape == (16,)
        assert converted.get_tensor(f"{w4}.weight_codebook").shape == (16,)
        assert f"{w4}.weight_scale" not in keys
        assert _decode_config(converted, w4) == {
            "format": "asym_w4a8_int8",
            "group_size": 16,
            "convrot_groupsize": 256,
        }

        assert converted.get_tensor(f"{bf16}.weight").dtype == torch.bfloat16
        assert f"{bf16}.comfy_quant" not in keys
        assert f"{bf16}.weight_scale" not in keys
        assert torch.equal(converted.get_tensor(passthrough), tensors[passthrough])
        assert torch.equal(converted.get_tensor(f"{w8}.bias"), tensors[f"{w8}.bias"])
        metadata = converted.metadata()
        assert metadata["license"] == "test"
        assert metadata["flux_klein_convrot_policy"] == "unit-test-policy"
        assert metadata["flux_klein_convrot_int8_layers"] == "1"
        assert metadata["flux_klein_convrot_int4_layers"] == "1"
        assert metadata["flux_klein_convrot_bf16_layers"] == "1"
        assert metadata["flux_klein_convrot_int4_layout"] == "w4a8"
        assert metadata["flux_klein_convrot_int4_group_size"] == "16"

    assert result["counts"] == {"int8": 1, "int4": 1, "bf16": 1}
    saved_report = json.loads(report.read_text(encoding="utf-8"))
    assert saved_report["policy"] == "unit-test-policy"
    assert saved_report["int4_layout"] == "w4a8"
    assert saved_report["int4_group_size"] == 16
    assert saved_report["output_bytes"] == output.stat().st_size


def test_policy_rejects_unknown_layers(tmp_path):
    source = tmp_path / "flux-klein-bf16.safetensors"
    policy = tmp_path / "policy.json"
    save_file(
        {"double_blocks.0.img_attn.qkv.weight": torch.randn(8, 256, dtype=torch.bfloat16)},
        source,
    )
    _write_policy(policy, {"double_blocks.99.img_attn.qkv": "bf16"})

    with pytest.raises(ValueError, match="unknown layers"):
        inspect_flux_klein_checkpoint(
            str(source),
            str(policy),
            strict_architecture=False,
        )


def test_strict_mode_rejects_non_9b_fixture(tmp_path):
    source = tmp_path / "not-klein-9b.safetensors"
    save_file(
        {"double_blocks.0.img_attn.qkv.weight": torch.randn(8, 256, dtype=torch.bfloat16)},
        source,
    )

    with pytest.raises(ValueError, match="112"):
        inspect_flux_klein_checkpoint(str(source), strict_architecture=True)
