import os
import unittest
import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.config.layer_config import load_layer_config
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.utils.tensor_utils import tensor_to_dict


class TestLayerConfigINT4(unittest.TestCase):

    def setUp(self):
        self.input_file = "test_int4_layer_config_in.safetensors"
        self.output_file = "test_int4_layer_config_out.safetensors"
        self.config_file = "test_int4_layer_config.json"

        tensors = {
            "model.layer1.weight": torch.randn(16, 64, dtype=torch.float32),
            "model.layer2.weight": torch.randn(16, 64, dtype=torch.float32),
        }
        save_file(tensors, self.input_file)

    def tearDown(self):
        for f in [self.input_file, self.output_file, self.config_file]:
            if os.path.exists(f):
                os.remove(f)

    def test_layer_config_int4_convrot_routes_correctly(self):
        import json
        cfg = {
            "model.layer1": {"format": "int4_convrot_w4a4", "simple": True},
            "model.layer2": {"format": "float8_e4m3fn", "simple": True},
        }
        with open(self.config_file, "w") as f:
            json.dump(cfg, f)

        layer_config = load_layer_config(self.config_file)

        convert_to_fp8_scaled(
            self.input_file,
            self.output_file,
            comfy_quant=True,
            filter_flags={},
            calib_samples=10,
            seed=42,
            layer_config=layer_config,
            device="cpu",
            no_learned_rounding=True,
        )

        out = load_file(self.output_file)

        # Layer 1 must be packed INT4 (int8 container, 32 columns)
        assert out["model.layer1.weight"].dtype == torch.int8
        assert out["model.layer1.weight"].shape == (16, 32)
        meta1 = tensor_to_dict(out["model.layer1.comfy_quant"])
        assert meta1["format"] == "convrot_w4a4"
        assert meta1["convrot"] is True

        # Layer 2 must be FP8 (float8_e4m3fn container, 64 columns)
        assert out["model.layer2.weight"].dtype == torch.float8_e4m3fn
        assert out["model.layer2.weight"].shape == (16, 64)
