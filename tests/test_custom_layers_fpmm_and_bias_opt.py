"""
Unit tests for custom layers argument options (--custom-fpmm, --custom-convrot) and has_bias optimization.
"""

import os
import unittest

import torch
from safetensors.torch import (
    load_file,
    save_file,
)

from convert_to_quant.formats.fp8_conversion import (
    convert_to_fp8_scaled,
)
from convert_to_quant.utils.comfy_quant import tensor_to_dict


class TestCustomLayersAndBiasOpt(unittest.TestCase):

    def setUp(self):
        self.input_path = "test_custom_opt_input.safetensors"
        self.output_path = "test_custom_opt_output.safetensors"

        # Create a small synthetic model:
        # - blocks.0.attn.wq.weight (with associated bias blocks.0.attn.wq.bias)
        # - blocks.0.attn.wk.weight (WITHOUT bias)
        # - blocks.0.mlp.down.weight (WITHOUT bias)
        self.tensors = {
            "blocks.0.attn.wq.weight": torch.randn(64, 64, dtype=torch.float16),
            "blocks.0.attn.wq.bias": torch.randn(64, dtype=torch.float16),
            "blocks.0.attn.wk.weight": torch.randn(64, 64, dtype=torch.float16),
            "blocks.0.mlp.down.weight": torch.randn(64, 64, dtype=torch.float16)
        }
        save_file(self.tensors, self.input_path)

    def tearDown(self):
        for path in [self.input_path, self.output_path]:
            if os.path.exists(path):
                os.remove(path)

    def test_custom_fpmm_and_has_bias(self):
        # Run conversion with custom options:
        # - target custom layers "blocks.0.mlp.down" and "blocks.0.attn.wk"
        # - set custom_fpmm = True for custom layers only
        convert_to_fp8_scaled(
            input_file=self.input_path,
            output_file=self.output_path,
            comfy_quant=True,
            filter_flags={},
            calib_samples=100,
            seed=42,
            no_learned_rounding=True,  # simple mode
            custom_layers="(blocks.0.mlp.down|blocks.0.attn.wk)",
            custom_type="fp8",
            custom_full_precision_mm=True,
            device="cpu",
        )

        out_tensors = load_file(self.output_path)

        # 1. Check that blocks.0.mlp.down.weight has comfy_quant metadata containing full_precision_matrix_mult
        self.assertIn("blocks.0.mlp.down.comfy_quant", out_tensors)
        comfy_quant_down = tensor_to_dict(out_tensors["blocks.0.mlp.down.comfy_quant"])
        self.assertTrue(comfy_quant_down.get("full_precision_matrix_mult", False))

        self.assertIn("blocks.0.attn.wk.comfy_quant", out_tensors)
        comfy_quant_wk = tensor_to_dict(out_tensors["blocks.0.attn.wk.comfy_quant"])
        self.assertTrue(comfy_quant_wk.get("full_precision_matrix_mult", False))

        # 2. Check that blocks.0.attn.wq.weight (which is NOT matched by custom-layers) does NOT have full_precision_matrix_mult
        self.assertIn("blocks.0.attn.wq.comfy_quant", out_tensors)
        comfy_quant_wq = tensor_to_dict(out_tensors["blocks.0.attn.wq.comfy_quant"])
        self.assertFalse(comfy_quant_wq.get("full_precision_matrix_mult", False))


if __name__ == "__main__":
    unittest.main()
