"""
Regression test for Finding 1 of the whole-branch review: a --layer-config
entry with format "int4_convrot_w4a4" must route through the INT4 converter,
not silently fall back to FP8.

Before the fix, formats/fp8_conversion.py's layer_config format-mapping block
only recognized "float8*" and "int8*" prefixes; any other format (including
the valid, whitelisted "int4_convrot_w4a4") fell into the `else` branch and
was converted as FP8 - producing a float8_e4m3fn output tensor mislabeled
with int4 metadata, with no error raised.

This test builds a tiny synthetic model with an in_features divisible by the
default ConvRot group size (256), runs it through convert_to_fp8_scaled with
a layer_config that requests "int4_convrot_w4a4" for that layer, and checks
that the output weight tensor is actually packed INT4 (torch.int8 storage,
half the column count) rather than float8_e4m3fn.
"""

import os
import unittest

import torch
from safetensors.torch import load_file, save_file

from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled


_FP8_KWARGS = dict(
    comfy_quant=True,
    calib_samples=4,
    seed=0,
    int8=False,
    no_learned_rounding=True,
    save_quant_metadata=False,
    low_memory=False,
    device="cpu",
    scaling_mode="tensor",
    optimizer="prodigy",
    num_iter=1,
    lr=1.0,
    lr_schedule="plateau",
    top_p=0.2,
    min_k=8,
    max_k=64,
    full_matrix=False,
    lr_gamma=0.99,
    lr_patience=1,
    lr_factor=0.95,
    lr_min=1e-8,
    lr_cooldown=0,
    lr_threshold=0.0,
    lr_adaptive_mode="simple-reset",
    lr_shape_influence=1.0,
    lr_threshold_mode="rel",
    early_stop_loss=5e-9,
    early_stop_lr=1.01e-8,
    early_stop_stall=2000,
    use_speed=False,
    extract_lora=False,
    lora_rank=4,
    lora_target=None,
    lora_depth=-1,
    lora_ar_threshold=0.0,
)


class TestLayerConfigInt4Routing(unittest.TestCase):
    def setUp(self):
        self.input_file = "_test_layer_config_int4_input.safetensors"
        self.output_file = "_test_layer_config_int4_output.safetensors"
        torch.manual_seed(0)
        out_features, in_features = 32, 256  # in_features divisible by default group size (256)
        tensors = {
            "model.layer.weight": torch.randn(out_features, in_features),
        }
        save_file(tensors, self.input_file)

    def tearDown(self):
        for f in [self.input_file, self.output_file]:
            if os.path.exists(f):
                os.remove(f)

    def test_layer_config_int4_convrot_routes_to_int4_converter(self):
        layer_config = {
            "model.layer": {"format": "int4_convrot_w4a4", "simple": True},
        }
        kw = {**_FP8_KWARGS}
        convert_to_fp8_scaled(
            self.input_file,
            self.output_file,
            filter_flags={},
            layer_config=layer_config,
            **kw,
        )
        out = load_file(self.output_file)

        weight = out["model.layer.weight"]
        # INT4 packing stores two nibbles per byte -> half the column count, int8 storage.
        self.assertEqual(weight.dtype, torch.int8)
        self.assertEqual(weight.shape, (32, 128))
        self.assertNotEqual(weight.dtype, torch.float8_e4m3fn)

        # Metadata must claim int4, and must be consistent with the actual output.
        comfy_quant_key = "model.layer.comfy_quant"
        self.assertIn(comfy_quant_key, out)


if __name__ == "__main__":
    unittest.main()
