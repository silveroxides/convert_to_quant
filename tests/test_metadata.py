import torch
import sys
import os
import unittest
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from convert_to_quant.converters.metadata import pack_metadata, unpack_metadata
from convert_to_quant.converters.sdnq_math import sdnq_quantize_layer_weight
from convert_to_quant.converters.sdnq_transform import convert_state_dict

class TestMetadata(unittest.TestCase):
    def test_pack_unpack(self):
        settings = {
            "weights_dtype": "int8",
            "group_size": 32,
            "use_svd": True,
            "svd_rank": 16,
            "random_extra_key": "should_be_ignored"
        }
        
        packed = pack_metadata(settings)
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(packed.ndim, 1)
        
        unpacked = unpack_metadata(packed)
        
        # Check that standardized keys are present and correct
        self.assertEqual(unpacked["weights_dtype"], "int8")
        self.assertEqual(unpacked["group_size"], 32)
        self.assertEqual(unpacked["use_svd"], True)
        self.assertEqual(unpacked["svd_rank"], 16)
        
        # Check that extra keys were filtered out
        self.assertNotIn("random_extra_key", unpacked)

    def test_integration(self):
        # Create a dummy weight
        weight = torch.randn(64, 64)
        state_dict = {"layer.weight": weight}
        config = {
            "weights_dtype": "int8",
            "group_size": 32
        }
        
        new_state_dict = convert_state_dict(state_dict, config)
        
        # Check if metadata tensor exists
        self.assertIn("layer.weight_metadata", new_state_dict)
        
        # Unpack and verify
        metadata = new_state_dict["layer.weight_metadata"]
        info = unpack_metadata(metadata)
        
        self.assertEqual(info["weights_dtype"], "int8")
        # group_size might be adjusted if invalid, but here 32 divides 64 so it should be kept?
        # sdnq_math logic: if group_size > 0: ...
        # If channel_size (64) // group_size (32) = 2. It works.
        self.assertEqual(info["group_size"], 32)

if __name__ == "__main__":
    unittest.main()
