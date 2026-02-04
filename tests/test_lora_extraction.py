import torch
import unittest
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.converters.learned_nvfp4 import LearnedNVFP4Converter

class TestLoraExtraction(unittest.TestCase):
    def test_heuristics(self):
        # Setup converter with LoRA enabled
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_depth=1,
            lora_rank=16
        )
        
        # 1. Block 0 should be targeted
        self.assertTrue(converter._should_extract_lora("double_blocks.0.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=0))
        
        # 2. Block 1 Attention (Square) should NOT be targeted (Depth 1 means only Block 0)
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_attn.qkv.weight", torch.Size([4096, 4096]), depth=1))
        
        # 3. Block 1 MLP (Elongated) should NOT be targeted
        self.assertFalse(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))
        
        # 4. Explicit Regex Target
        converter.lora_target_regex = __import__('re').compile("mlp")
        self.assertTrue(converter._should_extract_lora("double_blocks.1.img_mlp.0.weight", torch.Size([16384, 4096]), depth=1))

    def test_extraction_learned_rounding(self):
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_rank=4,
            no_learned_rounding=True # Use simple quant for fast test
        )
        
        # Create dummy weight
        W = torch.randn(128, 128)
        
        # Run conversion
        q, s, dq, extra = converter.convert(W, key="double_blocks.0.img_attn.qkv.weight")
        
        self.assertIn("lora_up", extra)
        self.assertIn("lora_down", extra)
        self.assertEqual(extra["lora_up"].shape, (128, 4))
        self.assertEqual(extra["lora_down"].shape, (4, 128))
        
        # Verify reconstruction
        error_approx = extra["lora_up"].float() @ extra["lora_down"].float()
        original_error = W - dq.cpu()
        
        # The approximation should capture some variance
        # (Since it's random, we just check it's not zero and has correct shape)
        self.assertGreater(torch.norm(error_approx), 0)

    def test_extraction_nvfp4(self):
        converter = LearnedNVFP4Converter(
            extract_lora=True,
            lora_rank=4,
            no_learned_rounding=True
        )
        
        W = torch.randn(64, 64)
        q, s, ps, dq, extra = converter.convert(W, key="double_blocks.0.weight")
        
        self.assertIn("lora_up", extra)
        self.assertEqual(extra["lora_up"].shape, (64, 4))

    def test_contiguous_output(self):
        converter = LearnedRoundingConverter(
            extract_lora=True,
            lora_rank=4,
            no_learned_rounding=True
        )
        W = torch.randn(64, 64)
        _, _, _, extra = converter.convert(W, key="double_blocks.0.weight")
        
        self.assertTrue(extra["lora_up"].is_contiguous(), "lora_up should be contiguous")
        self.assertTrue(extra["lora_down"].is_contiguous(), "lora_down should be contiguous")

if __name__ == "__main__":
    unittest.main()
