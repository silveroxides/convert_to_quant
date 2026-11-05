import os
import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.converters.int4_converter import INT4Converter
from convert_to_quant.formats.int4_conversion import convert_to_int4
from convert_to_quant.utils.convrot import unpack_int4_row_major
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_int4_converter_simple_contract():
    torch.manual_seed(42)
    out_features, in_features = 32, 64
    W = torch.randn(out_features, in_features, dtype=torch.float32)

    converter = INT4Converter(
        convrot=True,
        convrot_group_size=64,
        dynamic_convrot=False,
        device="cpu",
        no_learned_rounding=True,
        has_bias=False,
    )
    packed, scale, dequant, extra = converter.convert(W)

    assert packed.dtype == torch.int8
    assert packed.shape == (out_features, in_features // 2)
    assert scale.shape == (out_features,)
    assert scale.dtype == torch.float32
    assert dequant.shape == (out_features, in_features)

    # Check unpacked nibbles are within [-7, 7]
    unpacked = unpack_int4_row_major(packed)
    assert unpacked.min().item() >= -7
    assert unpacked.max().item() <= 7

    # Reconstruction relative error should be reasonable for 4-bit
    rel_err = (dequant - W).norm() / W.norm()
    assert rel_err < 0.45


def test_convert_to_int4_simple_e2e():
    import tempfile
    torch.manual_seed(42)
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "model_in.safetensors")
        out_file = os.path.join(tmp_dir, "model_out.safetensors")

        tensors = {
            "model.layers.0.attn.qkv.weight": torch.randn(32, 64, dtype=torch.bfloat16),
            "model.layers.0.attn.qkv.bias": torch.randn(32, dtype=torch.bfloat16),
            "model.layers.0.norm.weight": torch.randn(64, dtype=torch.bfloat16),
        }
        save_file(tensors, in_file)

        convert_to_int4(
            input_file=in_file,
            output_file=out_file,
            simple=True,
            convrot=True,
            convrot_group_size=64,
            dynamic_convrot=True,
            device="cpu",
        )

        out = load_file(out_file)

        # Weight tensor must be packed int8 with half the columns
        assert "model.layers.0.attn.qkv.weight" in out
        packed_w = out["model.layers.0.attn.qkv.weight"]
        assert packed_w.dtype == torch.int8
        assert packed_w.shape == (32, 32)

        # Weight scale must be present
        assert "model.layers.0.attn.qkv.weight_scale" in out
        scale = out["model.layers.0.attn.qkv.weight_scale"]
        assert scale.shape == (32,)

        # comfy_quant metadata tensor must exist and match ComfyUI convrot_w4a4 contract
        assert "model.layers.0.attn.qkv.comfy_quant" in out
        meta_dict = tensor_to_dict(out["model.layers.0.attn.qkv.comfy_quant"])
        assert meta_dict["format"] == "convrot_w4a4"
        assert meta_dict["convrot"] is True
        assert meta_dict["convrot_groupsize"] == 64
        assert meta_dict["orig_dtype"] == "torch.bfloat16"

        # Check safetensors header _quantization_metadata follows format_version 1.0 schema with "layers"
        from safetensors import safe_open
        import json
        with safe_open(out_file, framework="pt") as f:
            header_meta = f.metadata()
            assert "_quantization_metadata" in header_meta
            quant_meta = json.loads(header_meta["_quantization_metadata"])
            assert quant_meta.get("format_version") == "1.0"
            assert "layers" in quant_meta
            assert "model.layers.0.attn.qkv" in quant_meta["layers"]
            layer_entry = quant_meta["layers"]["model.layers.0.attn.qkv"]
            assert layer_entry["format"] == "convrot_w4a4"
            assert layer_entry["orig_dtype"] == "torch.bfloat16"
            assert layer_entry["convrot"] is True
            assert layer_entry["convrot_groupsize"] == 64

        # Norm weight (1D) preserved unquantized
        assert "model.layers.0.norm.weight" in out
        assert out["model.layers.0.norm.weight"].dtype == torch.bfloat16
        assert torch.equal(out["model.layers.0.norm.weight"], tensors["model.layers.0.norm.weight"])
