import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from convert_to_quant.cli.main import get_parser
from convert_to_quant.formats.ltx25_convrot import (
    convert_ltx25_convrot_hq,
    inspect_ltx25_checkpoint,
    preserve_ltx25_bf16,
)


PREFIX = "model.diffusion_model.transformer_blocks"


def test_video_recipe_preserves_the_validated_dev_and_distilled_projections():
    audio_out = f"{PREFIX}.0.audio_attn1.to_out.0"
    video_out = f"{PREFIX}.12.attn1.to_out.0"
    video_to_audio_v = f"{PREFIX}.4.video_to_audio_attn.to_v"
    dev_only_q = f"{PREFIX}.20.video_to_audio_attn.to_q"
    ordinary = f"{PREFIX}.0.attn1.to_q"

    for variant in ("dev", "distilled"):
        assert preserve_ltx25_bf16(audio_out, variant, "video")
        assert preserve_ltx25_bf16(video_out, variant, "video")
        assert preserve_ltx25_bf16(video_to_audio_v, variant, "video")
        assert not preserve_ltx25_bf16(ordinary, variant, "video")

    assert preserve_ltx25_bf16(dev_only_q, "dev", "video")
    assert not preserve_ltx25_bf16(dev_only_q, "distilled", "video")
    assert not preserve_ltx25_bf16(video_to_audio_v, "dev", "compact")
    assert not preserve_ltx25_bf16(audio_out, "dev", "full")


def test_ltx25_cli_arguments_are_available():
    args = get_parser().parse_args(
        [
            "-i",
            "model.safetensors",
            "--ltx25-convrot-hq",
            "--ltx25-variant",
            "distilled",
            "--ltx25-recipe",
            "compact",
        ]
    )

    assert args.ltx25_convrot_hq is True
    assert args.ltx25_variant == "distilled"
    assert args.ltx25_recipe == "compact"


def test_streaming_ltx25_conversion_keeps_selected_bf16_and_writes_native_convrot(tmp_path):
    source = tmp_path / "ltx-2.5-22b-dev-transformer-bf16.safetensors"
    output = tmp_path / "ltx-2.5-dev-int8.safetensors"
    report = tmp_path / "report.json"
    quantized = f"{PREFIX}.0.attn1.to_q"
    preserved = f"{PREFIX}.0.audio_attn1.to_out.0"
    passthrough = "model.diffusion_model.proj_out.weight"
    tensors = {
        f"{quantized}.weight": torch.linspace(-1, 1, 256, dtype=torch.bfloat16).reshape(16, 16),
        f"{preserved}.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        passthrough: torch.randn(16, 16, dtype=torch.bfloat16),
        f"{quantized}.bias": torch.randn(16, dtype=torch.bfloat16),
    }
    save_file(tensors, source, metadata={"model_version": "2.5.0", "license": "test"})

    plan = inspect_ltx25_checkpoint(
        str(source), variant="auto", recipe="video", min_gemm=8, strict_architecture=False
    )
    assert plan["variant"] == "dev"
    assert set(plan["quantized"]) == {quantized}
    assert plan["preserved"] == {preserved}

    result = convert_ltx25_convrot_hq(
        str(source),
        str(output),
        str(report),
        variant="auto",
        recipe="video",
        min_gemm=8,
        device="cpu",
        strict_architecture=False,
    )

    with safe_open(output, framework="pt", device="cpu") as converted:
        keys = set(converted.keys())
        assert converted.get_tensor(f"{quantized}.weight").dtype == torch.int8
        assert converted.get_tensor(f"{quantized}.weight_scale").shape == (16, 1)
        config = bytes(converted.get_tensor(f"{quantized}.comfy_quant").tolist()).decode().strip()
        assert json.loads(config) == {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": 16,
        }
        assert converted.get_tensor(f"{preserved}.weight").dtype == torch.bfloat16
        assert f"{preserved}.weight_scale" not in keys
        assert f"{preserved}.comfy_quant" not in keys
        assert torch.equal(converted.get_tensor(passthrough), tensors[passthrough])
        metadata = converted.metadata()
        assert metadata["license"] == "test"
        assert metadata["convrot_hq_int8_layers"] == "1"
        assert metadata["convrot_hq_restored_bf16_layers"] == "1"

    assert result["int8_layers"] == 1
    assert result["bf16_layers"] == 1
    assert json.loads(report.read_text(encoding="utf-8"))["variant"] == "dev"
