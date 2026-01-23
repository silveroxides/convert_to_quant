
import os
import torch
import unittest
from safetensors.torch import save_file, load_file
from safetensors import safe_open
from convert_to_quant.formats.hybrid_mxfp8_conversion import convert_to_hybrid_mxfp8
from convert_to_quant.utils.comfy_quant import edit_comfy_quant, create_comfy_quant_tensor, tensor_to_dict

class TestHybridMXFP8(unittest.TestCase):
    def setUp(self):
        self.input_file = "test_input_mxfp8.safetensors"
        self.output_file = "test_output_hybrid.safetensors"
        self.tensor_scales_file = "test_tensor_scales.safetensors"
        
        # Create dummy MXFP8 model
        tensors = {}
        # Weight (FP8)
        tensors["layer1.weight"] = torch.zeros((32, 32), dtype=torch.float8_e4m3fn)
        # Block scales (E8M0 -> uint8) - randomized
        tensors["layer1.weight_scale"] = torch.randint(0, 255, (1, 32), dtype=torch.uint8) # Flattened block scales
        # Comfy quant
        tensors["layer1.comfy_quant"] = create_comfy_quant_tensor("mxfp8")
        
        save_file(tensors, self.input_file, metadata={"_quantization_metadata": '{"layers": {"layer1": {"format": "mxfp8"}}}'})

        # Create dummy tensor scales model
        scale_tensors = {}
        scale_tensors["layer1.weight_scale"] = torch.tensor(2.5, dtype=torch.float32)
        save_file(scale_tensors, self.tensor_scales_file)

    def tearDown(self):
        for f in [self.input_file, self.output_file, self.tensor_scales_file]:
            if os.path.exists(f):
                os.remove(f)

    def test_conversion_computed(self):
        # Convert without external scales
        convert_to_hybrid_mxfp8(self.input_file, self.output_file)
        
        self.assertTrue(os.path.exists(self.output_file))
        
        with safe_open(self.output_file, framework="pt") as f:
            keys = f.keys()
            self.assertIn("layer1.weight_scalar", keys)
            
            # Check comfy_quant updated
            cq = f.get_tensor("layer1.comfy_quant")
            config = tensor_to_dict(cq)
            self.assertEqual(config["format"], "hybrid_mxfp8")
            
            # Check metadata updated
            meta = f.metadata()
            self.assertIn("hybrid_mxfp8", meta["_quantization_metadata"])

    def test_conversion_external(self):
        # Convert with external scales
        convert_to_hybrid_mxfp8(self.input_file, self.output_file, tensor_scales_path=self.tensor_scales_file)
        
        with safe_open(self.output_file, framework="pt") as f:
            scalar = f.get_tensor("layer1.weight_scalar")
            self.assertEqual(scalar.item(), 2.5)

    def test_edit_quant(self):
        # Test editing existing key
        convert_to_hybrid_mxfp8(self.input_file, self.output_file)
        
        # Now edit it to add a key
        edit_output = "test_output_edited.safetensors"
        edit_comfy_quant(
            self.output_file,
            edit_output,
            add_keys_str="'my_key': 'my_value', 'group_size': 64"
        )
        
        with safe_open(edit_output, framework="pt") as f:
            cq = f.get_tensor("layer1.comfy_quant")
            config = tensor_to_dict(cq)
            self.assertEqual(config["my_key"], "my_value")
            self.assertEqual(config["group_size"], 64)
            self.assertEqual(config["format"], "hybrid_mxfp8")
            
        if os.path.exists(edit_output):
            os.remove(edit_output)

if __name__ == "__main__":
    unittest.main()
