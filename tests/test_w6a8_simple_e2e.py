import json
import os
import sys
import tempfile
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.converters.w6a8_converter import W6A8Converter
from convert_to_quant.formats.w6a8_conversion import convert_to_w6a8
from convert_to_quant.utils.convrot import unpack_int6_w6a8
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_w6a8_converter_simple_contract():
    torch.manual_seed(42)
    out_features, in_features = 32, 256
    W = torch.randn(out_features, in_features, dtype=torch.float32) * 0.05

    converter = W6A8Converter(
        convrot=True,
        convrot_group_size=256,
        group_size=16,
        dynamic_convrot=False,
        device="cpu",
        no_learned_rounding=True,
        has_bias=False,
    )
    packed, s_rel, s_channel, dequant, extra = converter.convert(W)

    assert packed.dtype == torch.int8
    assert packed.shape == (out_features, (3 * in_features) // 4)
    assert s_rel.dtype == torch.float8_e4m3fn
    assert s_rel.shape == (out_features, in_features // 16)
    assert s_channel.dtype == torch.float32
    assert s_channel.shape == (out_features,)
    assert dequant.shape == (out_features, in_features)
    assert extra == {}

    codes = unpack_int6_w6a8(packed, in_features)
    assert codes.min().item() >= 1
    assert codes.max().item() <= 63

    rel_err = (dequant - W).norm() / W.norm()
    assert rel_err < 0.05


def test_w6a8_converter_zero_tensor():
    W = torch.zeros(16, 64, dtype=torch.float32)
    converter = W6A8Converter(convrot_group_size=64, group_size=16, dynamic_convrot=False, device="cpu", no_learned_rounding=True)
    packed, s_rel, s_channel, dequant, _ = converter.convert(W)
    assert packed.shape == (16, 48) and torch.all(packed == 0)
    assert s_rel.shape == (16, 4) and s_rel.dtype == torch.float8_e4m3fn
    assert s_channel.shape == (16,) and s_channel.dtype == torch.float32
    assert torch.all(dequant == 0)


def test_convert_to_w6a8_simple_e2e():
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "model_in.safetensors")
        out_file = os.path.join(tmp_dir, "model_out.safetensors")

        tensors = {
            "model.layers.0.attn.qkv.weight": torch.randn(32, 256, dtype=torch.bfloat16) * 0.05,
            "model.layers.0.attn.qkv.bias": torch.randn(32, dtype=torch.bfloat16),
            "model.layers.0.incompatible.weight": torch.randn(16, 48, dtype=torch.bfloat16),
            "model.layers.0.norm.weight": torch.randn(256, dtype=torch.bfloat16),
        }
        save_file(tensors, in_file)

        convert_to_w6a8(
            input_file=in_file,
            output_file=out_file,
            simple=True,
            convrot=True,
            group_size=16,
            convrot_group_size=256,
            dynamic_convrot=True,
            device="cpu",
        )

        out = load_file(out_file)

        # Packed weight [32, 3*256//4 = 192]
        assert "model.layers.0.attn.qkv.weight" in out
        packed_w = out["model.layers.0.attn.qkv.weight"]
        assert packed_w.dtype == torch.int8
        assert packed_w.shape == (32, 192)

        # W6A8 dual scales (weight_s_rel + weight_s_channel), no weight_scale
        assert "model.layers.0.attn.qkv.weight_s_rel" in out
        assert "model.layers.0.attn.qkv.weight_s_channel" in out
        assert "model.layers.0.attn.qkv.weight_scale" not in out
        s_rel = out["model.layers.0.attn.qkv.weight_s_rel"]
        s_channel = out["model.layers.0.attn.qkv.weight_s_channel"]
        assert s_rel.dtype == torch.float8_e4m3fn and s_rel.shape == (32, 16)
        assert s_channel.dtype == torch.float32 and s_channel.shape == (32,)

        # comfy_quant metadata matches ComfyUI w6a8_int8 contract
        assert "model.layers.0.attn.qkv.comfy_quant" in out
        meta_dict = tensor_to_dict(out["model.layers.0.attn.qkv.comfy_quant"])
        assert meta_dict["format"] == "w6a8_int8"
        assert meta_dict["group_size"] == 16
        assert meta_dict["convrot_groupsize"] == 256
        assert meta_dict["orig_dtype"] == "torch.bfloat16"

        # Header _quantization_metadata follows format_version 1.0 schema
        with safe_open(out_file, framework="pt") as f:
            header_meta = f.metadata()
            assert "_quantization_metadata" in header_meta
            quant_meta = json.loads(header_meta["_quantization_metadata"])
            assert quant_meta.get("format_version") == "1.0"
            layer_entry = quant_meta["layers"]["model.layers.0.attn.qkv"]
            assert layer_entry["format"] == "w6a8_int8"
            assert layer_entry["group_size"] == 16
            assert layer_entry["convrot_groupsize"] == 256
            assert layer_entry["orig_dtype"] == "torch.bfloat16"

        # Incompatible K=48 weight skipped unquantized
        assert out["model.layers.0.incompatible.weight"].dtype == torch.bfloat16
        assert "model.layers.0.incompatible.comfy_quant" not in out

        # Norm weight (1D) preserved unquantized
        assert torch.equal(out["model.layers.0.norm.weight"], tensors["model.layers.0.norm.weight"])

        # Verify Comfy Kitchen AsymW4A8Int8Layout can directly reconstruct & dequantize saved tensors
        kitchen_path = Path(__file__).resolve().parent.parent / "references" / "comfy-kitchen"
        if (kitchen_path / "comfy_kitchen").exists():
            saved_modules = {k: v for k, v in sys.modules.items() if k == "comfy_kitchen" or k.startswith("comfy_kitchen.")}
            for k in saved_modules:
                del sys.modules[k]
            sys.path.insert(0, str(kitchen_path))
            try:
                from comfy_kitchen.tensor import AsymW4A8Int8Layout, QuantizedTensor

                params = AsymW4A8Int8Layout.Params(
                    scale=s_rel,
                    s_channel=s_channel,
                    codebook=None,
                    orig_dtype=torch.bfloat16,
                    orig_shape=(32, 256),
                    group_size=meta_dict["group_size"],
                    convrot_groupsize=meta_dict["convrot_groupsize"],
                )
                qt = QuantizedTensor(packed_w, "AsymW4A8Int8Layout", params)
                assert AsymW4A8Int8Layout.bits(qt) == 6
                dq = qt.dequantize()
                rel_l2 = (
                    (dq.float() - tensors["model.layers.0.attn.qkv.weight"].float()).norm()
                    / tensors["model.layers.0.attn.qkv.weight"].float().norm()
                ).item()
                assert rel_l2 < 0.05
            finally:
                if str(kitchen_path) in sys.path:
                    sys.path.remove(str(kitchen_path))
                for k in [k for k in sys.modules if k == "comfy_kitchen" or k.startswith("comfy_kitchen.")]:
                    del sys.modules[k]
                sys.modules.update(saved_modules)
