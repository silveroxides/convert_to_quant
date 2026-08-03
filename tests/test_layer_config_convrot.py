"""
Unit tests for --layer-config support of ConvRot options (convrot, convrot_group_size,
dynamic_convrot), added so per-layer JSON configs can enable Hadamard-rotated INT8
row-wise quantization the same way --custom-convrot does for --custom-layers.
"""

import os
import unittest

import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.config.layer_config import load_layer_config
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.utils.comfy_quant import tensor_to_dict


class TestLayerConfigConvrot(unittest.TestCase):

    def setUp(self):
        self.input_path = "test_layer_config_convrot_input.safetensors"
        self.output_path = "test_layer_config_convrot_output.safetensors"
        self.config_path = "test_layer_config_convrot.json"

        # in_features=64 is divisible by 4, 16, and 64 (all valid ConvRot group sizes)
        self.tensors = {
            "blocks.0.attn.qkv.weight": torch.randn(64, 64, dtype=torch.float16),
            "blocks.0.attn.out.weight": torch.randn(64, 64, dtype=torch.float16),
        }
        save_file(self.tensors, self.input_path)

    def tearDown(self):
        for path in [self.input_path, self.output_path, self.config_path]:
            if os.path.exists(path):
                os.remove(path)

    def _run(self, config):
        import json
        with open(self.config_path, "w") as f:
            json.dump(config, f)
        layer_config = load_layer_config(self.config_path)

        convert_to_fp8_scaled(
            input_file=self.input_path,
            output_file=self.output_path,
            comfy_quant=True,
            filter_flags={},
            calib_samples=100,
            seed=42,
            no_learned_rounding=True,  # simple mode, fast
            layer_config=layer_config,
            device="cpu",
        )
        return load_file(self.output_path)

    def test_fixed_convrot_group_size(self):
        out = self._run({
            "attn\\.qkv": {
                "format": "int8_tensorwise",
                "scaling_mode": "row",
                "convrot": True,
                "convrot_group_size": 16,
            },
            "attn\\.out": {
                "format": "int8_tensorwise",
                "scaling_mode": "row",
            },
        })

        self.assertIn("blocks.0.attn.qkv.comfy_quant", out)
        qkv_meta = tensor_to_dict(out["blocks.0.attn.qkv.comfy_quant"])
        self.assertTrue(qkv_meta.get("convrot", False))
        self.assertEqual(qkv_meta.get("convrot_groupsize"), 16)

        # Layer without convrot in its config must not have it applied
        self.assertIn("blocks.0.attn.out.comfy_quant", out)
        out_meta = tensor_to_dict(out["blocks.0.attn.out.comfy_quant"])
        self.assertFalse(out_meta.get("convrot", False))

    def test_dynamic_convrot_picks_compatible_group_size(self):
        out = self._run({
            "attn\\.qkv": {
                "format": "int8_tensorwise",
                "scaling_mode": "row",
                "dynamic_convrot": True,
                "convrot_group_size": 16,  # used as the minimum for dynamic search
            },
        })

        self.assertIn("blocks.0.attn.qkv.comfy_quant", out)
        qkv_meta = tensor_to_dict(out["blocks.0.attn.qkv.comfy_quant"])
        self.assertTrue(qkv_meta.get("convrot", False))
        # in_features=64, min_group_size=16 -> largest compatible power-of-4 is 64
        self.assertEqual(qkv_meta.get("convrot_groupsize"), 64)

    def test_convrot_ignored_without_row_scaling(self):
        # ConvRot is only valid for int8 + row scaling; requesting it with tensor
        # scaling should not blow up, and should simply not apply rotation.
        out = self._run({
            "attn\\.qkv": {
                "format": "int8_tensorwise",
                "scaling_mode": "tensor",
                "convrot": True,
                "convrot_group_size": 16,
            },
        })

        self.assertIn("blocks.0.attn.qkv.comfy_quant", out)
        qkv_meta = tensor_to_dict(out["blocks.0.attn.qkv.comfy_quant"])
        self.assertFalse(qkv_meta.get("convrot", False))


if __name__ == "__main__":
    unittest.main()
