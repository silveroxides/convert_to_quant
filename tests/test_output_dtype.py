import json
import os
import tempfile
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.cli.main import get_parser
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.formats.mxfp8_conversion import convert_to_mxfp8
from convert_to_quant.formats.nvfp4_conversion import convert_to_nvfp4
from convert_to_quant.utils.output_dtype import cast_unquantized_weights
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def _model():
    return {
        "blocks.0.attn.wq.weight": torch.randn(64, 64, dtype=torch.float32),
        "txtfusion.layerwise_blocks.0.attn.wq.weight": torch.randn(64, 64, dtype=torch.float32),
        "txtfusion.projector.weight": torch.randn(64, 64, dtype=torch.float32),
        "first.weight": torch.randn(64, 64, dtype=torch.bfloat16),
        "conv.weight": torch.randn(8, 8, 3, 3, dtype=torch.float32),
        "vector.weight": torch.randn(64, dtype=torch.float32),
        "blocks.0.attn.qknorm.qnorm.scale": torch.randn(64, dtype=torch.float32),
    }


class TestOutputDtype(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_file = os.path.join(self.temp_dir.name, "input.safetensors")
        save_file(_model(), self.input_file)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _assert_common_bfloat16_policy(self, output_file):
        tensors = load_file(output_file)
        quant_base = "blocks.0.attn.wq"

        self.assertEqual(tensors["txtfusion.layerwise_blocks.0.attn.wq.weight"].dtype, torch.bfloat16)
        self.assertEqual(tensors["txtfusion.projector.weight"].dtype, torch.float32)
        self.assertEqual(tensors["first.weight"].dtype, torch.bfloat16)
        self.assertEqual(tensors["conv.weight"].dtype, torch.float32)
        self.assertEqual(tensors["vector.weight"].dtype, torch.float32)
        self.assertEqual(tensors["blocks.0.attn.qknorm.qnorm.scale"].dtype, torch.float32)

        config = tensor_to_dict(tensors[f"{quant_base}.comfy_quant"])
        self.assertEqual(config["orig_dtype"], "torch.bfloat16")

        with safe_open(output_file, framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
        quant_metadata = json.loads(metadata["_quantization_metadata"])
        self.assertEqual(quant_metadata["layers"][quant_base]["orig_dtype"], "torch.bfloat16")

    def test_unified_path_applies_krea2_policy_and_metadata(self):
        output_file = os.path.join(self.temp_dir.name, "fp8.safetensors")
        convert_to_fp8_scaled(
            self.input_file,
            output_file,
            comfy_quant=True,
            filter_flags={"krea2": True},
            calib_samples=4,
            seed=0,
            no_learned_rounding=True,
            save_quant_metadata=True,
            device="cpu",
            scaling_mode="tensor",
        )
        self._assert_common_bfloat16_policy(output_file)
        tensors = load_file(output_file)
        self.assertEqual(tensors["blocks.0.attn.wq.weight_scale"].dtype, torch.float32)

    def test_unified_path_float16_and_manual_preservation(self):
        output_file = os.path.join(self.temp_dir.name, "fp16.safetensors")
        preserved_key = "txtfusion.layerwise_blocks.0.attn.wq.weight"
        convert_to_fp8_scaled(
            self.input_file,
            output_file,
            comfy_quant=True,
            filter_flags={"krea2": True},
            calib_samples=4,
            seed=0,
            no_learned_rounding=True,
            save_quant_metadata=True,
            device="cpu",
            scaling_mode="tensor",
            output_dtype="float16",
            preserve_layers=r"^txtfusion\.layerwise_blocks\.0\.attn\.wq\.weight$",
        )

        tensors = load_file(output_file)
        self.assertEqual(tensors[preserved_key].dtype, torch.float32)
        config = tensor_to_dict(tensors["blocks.0.attn.wq.comfy_quant"])
        self.assertEqual(config["orig_dtype"], "torch.float16")

    def test_mxfp8_path_applies_krea2_policy_and_metadata(self):
        output_file = os.path.join(self.temp_dir.name, "mxfp8.safetensors")
        convert_to_mxfp8(
            self.input_file,
            output_file,
            filter_flags={"krea2": True},
            simple=True,
            calib_samples=4,
            seed=0,
        )
        self._assert_common_bfloat16_policy(output_file)
        tensors = load_file(output_file)
        self.assertEqual(tensors["blocks.0.attn.wq.weight_scale"].dtype, torch.uint8)

    def test_nvfp4_path_applies_krea2_policy_and_metadata(self):
        output_file = os.path.join(self.temp_dir.name, "nvfp4.safetensors")
        convert_to_nvfp4(
            self.input_file,
            output_file,
            filter_flags={"krea2": True},
            simple=True,
            calib_samples=4,
            seed=0,
        )
        self._assert_common_bfloat16_policy(output_file)
        tensors = load_file(output_file)
        self.assertEqual(tensors["blocks.0.attn.wq.weight_scale_2"].dtype, torch.float32)

    def test_cli_defaults_and_rejects_fp32_output_dtype(self):
        parser = get_parser()
        args = parser.parse_args(["-i", "input.safetensors"])
        self.assertEqual(args.output_dtype, "bfloat16")
        self.assertIsNone(args.preserve_layers)

        with self.assertRaises(SystemExit):
            parser.parse_args(["-i", "input.safetensors", "--output-dtype", "float32"])

    def test_invalid_preserve_regex_stops_before_output(self):
        output_file = os.path.join(self.temp_dir.name, "invalid.safetensors")
        convert_to_fp8_scaled(
            self.input_file,
            output_file,
            comfy_quant=True,
            filter_flags={},
            calib_samples=4,
            seed=0,
            no_learned_rounding=True,
            device="cpu",
            preserve_layers="(",
        )
        self.assertFalse(os.path.exists(output_file))

    def test_krea2_preservation_map_matches_reference_fp32_weights(self):
        preserved_keys = {
            "first.weight",
            "last.linear.weight",
            "tmlp.0.weight",
            "tmlp.2.weight",
            "tproj.1.weight",
            "txtfusion.projector.weight",
            "txtmlp.1.weight",
            "txtmlp.3.weight",
        }
        converted_key = "txtfusion.layerwise_blocks.0.attn.wq.weight"
        tensors = {key: torch.randn(2, 2, dtype=torch.float32) for key in preserved_keys | {converted_key}}

        cast_unquantized_weights(
            tensors,
            quantized_weight_keys=set(),
            output_dtype="bfloat16",
            preserve_pattern=None,
            filter_flags={"krea2": True},
        )

        for key in preserved_keys:
            self.assertEqual(tensors[key].dtype, torch.float32)
        self.assertEqual(tensors[converted_key].dtype, torch.bfloat16)

    def test_qwen_vlm_preserves_excluded_2d_weights_as_bfloat16(self):
        tensors = {
            "model.language_model.layers.34.attn.weight": torch.randn(2, 2, dtype=torch.float32),
            "model.language_model.layers.35.attn.weight": torch.randn(2, 2, dtype=torch.float32),
            "model.visual.blocks.23.attn.weight": torch.randn(2, 2, dtype=torch.float32),
            "model.mtp.layers.2.attn.weight": torch.randn(2, 2, dtype=torch.float32),
            "model.visual.patch_embed.proj.weight": torch.randn(2, 2, 2, 2, dtype=torch.float32),
        }

        cast_unquantized_weights(
            tensors,
            quantized_weight_keys=set(),
            output_dtype="float16",
            preserve_pattern=None,
            filter_flags={"qwen_vlm": True},
        )

        self.assertEqual(tensors["model.language_model.layers.34.attn.weight"].dtype, torch.float16)
        self.assertEqual(tensors["model.language_model.layers.35.attn.weight"].dtype, torch.bfloat16)
        self.assertEqual(tensors["model.visual.blocks.23.attn.weight"].dtype, torch.bfloat16)
        self.assertEqual(tensors["model.mtp.layers.2.attn.weight"].dtype, torch.bfloat16)
        self.assertEqual(tensors["model.visual.patch_embed.proj.weight"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
