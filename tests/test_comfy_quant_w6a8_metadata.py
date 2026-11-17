import json
import os
import tempfile

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.utils.comfy_quant import create_comfy_quant_tensor, edit_comfy_quant
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def test_create_comfy_quant_tensor_w6a8_int8():
    t = create_comfy_quant_tensor(
        "w6a8_int8",
        block_size=16,
        convrot_groupsize=256,
        orig_dtype="torch.bfloat16",
    )
    cfg = tensor_to_dict(t)
    assert cfg == {
        "format": "w6a8_int8",
        "orig_dtype": "torch.bfloat16",
        "group_size": 16,
        "convrot_groupsize": 256,
    }


def test_edit_comfy_quant_preserves_w6a8_metadata():
    with tempfile.TemporaryDirectory(dir=r"C:\Users\ishim\AppData\Local\Temp\opencode") as tmp_dir:
        in_file = os.path.join(tmp_dir, "in.safetensors")
        out_file = os.path.join(tmp_dir, "out.safetensors")

        tensors = {
            "layer.weight": torch.zeros((16, 192), dtype=torch.int8),
            "layer.weight_s_rel": torch.ones((16, 16), dtype=torch.float8_e4m3fn),
            "layer.weight_s_channel": torch.ones((16,), dtype=torch.float32),
            "layer.comfy_quant": create_comfy_quant_tensor(
                "w6a8_int8",
                block_size=16,
                convrot_groupsize=256,
                orig_dtype="torch.bfloat16",
            ),
        }
        save_file(tensors, in_file)

        edit_comfy_quant(
            in_file,
            out_file,
            add_keys_str="'full_precision_matrix_mult': true",
            save_quant_metadata=True,
        )

        out = load_file(out_file)
        cfg = tensor_to_dict(out["layer.comfy_quant"])
        assert cfg["format"] == "w6a8_int8"
        assert cfg["group_size"] == 16
        assert cfg["convrot_groupsize"] == 256
        assert cfg["full_precision_matrix_mult"] is True

        with safe_open(out_file, framework="pt") as f:
            meta = json.loads(f.metadata()["_quantization_metadata"])
        assert meta["layers"]["layer"]["group_size"] == 16
        assert meta["layers"]["layer"]["convrot_groupsize"] == 256
        assert meta["layers"]["layer"]["full_precision_matrix_mult"] is True
