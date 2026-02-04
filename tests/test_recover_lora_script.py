import os
import torch
import unittest
import tempfile
import shutil
from safetensors.torch import save_file, load_file
from convert_to_quant.converters.mxfp8_converter import quantize_mxfp8
from convert_to_quant.converters.nvfp4_converter import quantize_nvfp4
from unittest.mock import patch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recover_lora import main as recover_main

class TestRecoverLoraScript(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_path = os.path.join(self.test_dir, "orig.safetensors")
        self.quant_mxfp8_path = os.path.join(self.test_dir, "quant_mxfp8.safetensors")
        self.quant_nvfp4_path = os.path.join(self.test_dir, "quant_nvfp4.safetensors")
        self.output_path = os.path.join(self.test_dir, "lora.safetensors")
        
        # Create dummy original model
        self.layer_name = "double_blocks.0.img_attn.qkv.weight"
        self.shape = (128, 128)
        self.W_orig = torch.randn(self.shape, dtype=torch.bfloat16)
        save_file({self.layer_name: self.W_orig}, self.orig_path)
        
        # Create dummy MXFP8 quantized model
        qdata, block_scales = quantize_mxfp8(self.W_orig)
        
        # Safetensors doesn't support e8m0, so cast to uint8
        if block_scales.dtype == torch.float8_e8m0fnu:
            block_scales = block_scales.view(torch.uint8)
        
        save_file({
            self.layer_name: qdata,
            f"double_blocks.0.img_attn.qkv.weight_scale": block_scales
        }, self.quant_mxfp8_path)
        
        # Create dummy NVFP4 quantized model
        qdata_nv, block_scales_nv, per_tensor_nv = quantize_nvfp4(self.W_orig)
        save_file({
            self.layer_name: qdata_nv,
            f"double_blocks.0.img_attn.qkv.weight_scale": block_scales_nv,
            f"double_blocks.0.img_attn.qkv.weight_scale_2": per_tensor_nv
        }, self.quant_nvfp4_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_recover_mxfp8(self):
        with patch.object(sys, 'argv', [
            "recover_lora.py",
            "--original", self.orig_path,
            "--quantized", self.quant_mxfp8_path,
            "--output", self.output_path,
            "--rank", "16",
            "--lora-depth", "1",
            "--device", "cpu"
        ]):
            recover_main()
            
        self.assertTrue(os.path.exists(self.output_path))
        lora_weights = load_file(self.output_path)
        self.assertIn("diffusion_model.double_blocks.0.img_attn.qkv.lora_up", lora_weights)
        self.assertIn("diffusion_model.double_blocks.0.img_attn.qkv.lora_down", lora_weights)
        self.assertEqual(lora_weights["diffusion_model.double_blocks.0.img_attn.qkv.lora_up"].shape, (128, 16))

    def test_recover_nvfp4(self):
        with patch.object(sys, 'argv', [
            "recover_lora.py",
            "--original", self.orig_path,
            "--quantized", self.quant_nvfp4_path,
            "--output", self.output_path,
            "--rank", "16",
            "--lora-depth", "1",
            "--device", "cpu"
        ]):
            recover_main()
            
        self.assertTrue(os.path.exists(self.output_path))
        lora_weights = load_file(self.output_path)
        self.assertIn("diffusion_model.double_blocks.0.img_attn.qkv.lora_up", lora_weights)
        
    def test_skip_logic(self):
        # Test that deep blocks are skipped if lora_depth is small
        # Use a skewed MLP layer to fail the attention heuristic as well
        deep_layer = "double_blocks.5.img_mlp.0.weight"
        # Skewed shape to fail aspect ratio check (128x512 = AR 4.0 > 1.5)
        skewed_shape = (128, 512)
        W_skewed = torch.randn(skewed_shape, dtype=torch.bfloat16)
        
        save_file({deep_layer: W_skewed}, self.orig_path)
        
        qdata, block_scales = quantize_mxfp8(W_skewed)
        if block_scales.dtype == torch.float8_e8m0fnu:
            block_scales = block_scales.view(torch.uint8)
        save_file({
            deep_layer: qdata,
            f"double_blocks.5.img_mlp.0.weight_scale": block_scales
        }, self.quant_mxfp8_path)
        
        with patch.object(sys, 'argv', [
            "recover_lora.py",
            "--original", self.orig_path,
            "--quantized", self.quant_mxfp8_path,
            "--output", self.output_path,
            "--lora-depth", "1", # Should skip block 5
            "--device", "cpu"
        ]):
            recover_main()
            
        # File might not exist if nothing saved, or exist but empty?
        # The script prints "No LoRA tensors extracted" and doesn't save if empty.
        self.assertFalse(os.path.exists(self.output_path))

if __name__ == "__main__":
    unittest.main()
