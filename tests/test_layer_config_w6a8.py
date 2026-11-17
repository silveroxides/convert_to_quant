import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.config.layer_config import load_layer_config
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled, resolve_layer_config_format
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_resolve_layer_config_format_w6a8():
    assert resolve_layer_config_format("w6a8_int8") == "w6a8"
    assert resolve_layer_config_format("w6a8") == "w6a8"


def test_layer_config_w6a8_mixed_routing():
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "in.safetensors")
        out_file = os.path.join(tmp_dir, "out.safetensors")
        cfg_file = os.path.join(tmp_dir, "cfg.json")

        tensors = {
            "model.layer1.weight": torch.randn(16, 256, dtype=torch.float32) * 0.05,
            "model.layer2.weight": torch.randn(16, 64, dtype=torch.float32) * 0.05,
            "model.layer3.weight": torch.randn(16, 64, dtype=torch.float32) * 0.05,
        }
        save_file(tensors, in_file)

        cfg = {
            "model.layer1": {"format": "w6a8_int8", "group_size": 32, "convrot_group_size": 256, "simple": True},
            "model.layer2": {"format": "int4_convrot_w4a4", "simple": True},
            "model.layer3": {"format": "float8_e4m3fn", "simple": True},
        }
        with open(cfg_file, "w") as f:
            json.dump(cfg, f)

        layer_config = load_layer_config(cfg_file)
        convert_to_fp8_scaled(
            in_file,
            out_file,
            comfy_quant=True,
            filter_flags={},
            calib_samples=10,
            seed=42,
            layer_config=layer_config,
            device="cpu",
            no_learned_rounding=True,
        )

        out = load_file(out_file)

        # Layer 1: W6A8 packed [16, 3*256//4 = 192]
        assert out["model.layer1.weight"].dtype == torch.int8
        assert out["model.layer1.weight"].shape == (16, 192)
        assert out["model.layer1.weight_s_rel"].dtype == torch.float8_e4m3fn
        assert out["model.layer1.weight_s_rel"].shape == (16, 8)
        assert out["model.layer1.weight_s_channel"].dtype == torch.float32
        assert out["model.layer1.weight_s_channel"].shape == (16,)
        meta1 = tensor_to_dict(out["model.layer1.comfy_quant"])
        assert meta1["format"] == "w6a8_int8"
        assert meta1["group_size"] == 32
        assert meta1["convrot_groupsize"] == 256

        # Layer 2: INT4 W4A4 packed [16, 32]
        assert out["model.layer2.weight"].dtype == torch.int8
        assert out["model.layer2.weight"].shape == (16, 32)
        assert tensor_to_dict(out["model.layer2.comfy_quant"])["format"] == "convrot_w4a4"

        # Layer 3: FP8 [16, 64]
        assert out["model.layer3.weight"].dtype == torch.float8_e4m3fn
        assert out["model.layer3.weight"].shape == (16, 64)

        with safe_open(out_file, framework="pt") as f:
            header_meta = json.loads(f.metadata()["_quantization_metadata"])
        assert header_meta["layers"]["model.layer1"]["format"] == "w6a8_int8"
        assert header_meta["layers"]["model.layer1"]["group_size"] == 32
        assert header_meta["layers"]["model.layer1"]["convrot_groupsize"] == 256


def test_cli_w6a8_standalone_and_mixed_fallback():
    from convert_to_quant.cli.main import get_parser, run_conversion

    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "in.safetensors")
        out_standalone = os.path.join(tmp_dir, "out_standalone.safetensors")
        out_mixed = os.path.join(tmp_dir, "out_mixed.safetensors")

        tensors = {
            "model.block.weight": torch.randn(16, 256, dtype=torch.float32) * 0.05,
            "model.proj_out.weight": torch.randn(16, 64, dtype=torch.float32) * 0.05,
        }
        save_file(tensors, in_file)

        parser = get_parser()
        args_standalone = parser.parse_args(
            ["-i", in_file, "-o", out_standalone, "--w6a8", "--simple", "--comfy_quant", "--group-size", "32", "--device", "cpu", "--manual_seed", "42"]
        )
        run_conversion(args_standalone)
        out_s = load_file(out_standalone)
        assert out_s["model.block.weight"].shape == (16, 192)
        assert out_s["model.block.weight_s_rel"].shape == (16, 8)
        assert tensor_to_dict(out_s["model.block.comfy_quant"])["group_size"] == 32

        args_mixed = parser.parse_args(
            [
                "-i", in_file, "-o", out_mixed,
                "--w6a8", "--simple", "--comfy_quant",
                "--exclude-layers", "proj_out", "--fallback", "fp8", "--fallback-simple",
                "--device", "cpu", "--manual_seed", "42",
            ]
        )
        run_conversion(args_mixed)
        out_m = load_file(out_mixed)
        assert out_m["model.block.weight"].dtype == torch.int8
        assert out_m["model.block.weight"].shape == (16, 192)
        assert tensor_to_dict(out_m["model.block.comfy_quant"])["format"] == "w6a8_int8"
        assert out_m["model.proj_out.weight"].dtype == torch.float8_e4m3fn
        assert tensor_to_dict(out_m["model.proj_out.comfy_quant"])["format"] == "float8_e4m3fn"


def test_custom_type_w6a8_in_fp8_primary():
    from convert_to_quant.cli.main import get_parser, run_conversion

    torch.manual_seed(11)
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "in.safetensors")
        out_file = os.path.join(tmp_dir, "out_custom.safetensors")

        tensors = {
            "model.block.weight": torch.randn(16, 256, dtype=torch.float32) * 0.05,
            "model.other.weight": torch.randn(16, 64, dtype=torch.float32) * 0.05,
        }
        save_file(tensors, in_file)

        parser = get_parser()
        args = parser.parse_args(
            [
                "-i", in_file, "-o", out_file,
                "--simple", "--comfy_quant",
                "--custom-layers", "block", "--custom-type", "w6a8", "--custom-simple",
                "--device", "cpu", "--manual_seed", "42",
            ]
        )
        run_conversion(args)
        out = load_file(out_file)
        assert out["model.block.weight"].dtype == torch.int8
        assert out["model.block.weight"].shape == (16, 192)
        assert tensor_to_dict(out["model.block.comfy_quant"])["format"] == "w6a8_int8"
        assert out["model.other.weight"].dtype == torch.float8_e4m3fn
