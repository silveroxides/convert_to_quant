import torch
import os
import sys
import subprocess
import json
from safetensors.torch import save_file, load_file
import shutil

def create_dummy_model(path: str):
    """Creates a dummy model with Linear and Conv2d layers."""
    state_dict = {}
    
    # Linear layer
    state_dict["layer1.weight"] = torch.randn(64, 128, dtype=torch.float32)
    state_dict["layer1.bias"] = torch.randn(64, dtype=torch.float32)
    
    # Conv2d layer
    state_dict["layer2.weight"] = torch.randn(32, 16, 3, 3, dtype=torch.float32)
    state_dict["layer2.bias"] = torch.randn(32, dtype=torch.float32)
    
    # Layer to exclude (by pattern in test config)
    state_dict["exclude_me.weight"] = torch.randn(10, 10, dtype=torch.float32)
    
    save_file(state_dict, path)
    print(f"Created dummy model at {path}")

def verify_output(path: str):
    """Verifies the quantized output."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Output file {path} not found")
        
    state_dict = load_file(path)
    print(f"Loaded output model from {path}")
    print(f"Keys: {list(state_dict.keys())}")
    
    # Check Linear layer quantization
    assert "layer1.weight" in state_dict
    assert "layer1.weight_scale" in state_dict
    assert "layer1.weight_metadata" in state_dict
    assert state_dict["layer1.weight"].dtype == torch.int8
    
    # Check Conv2d layer quantization
    assert "layer2.weight" in state_dict
    assert "layer2.weight_scale" in state_dict
    assert "layer2.weight_metadata" in state_dict
    assert state_dict["layer2.weight"].dtype == torch.int8

    # Check excluded layer
    assert "exclude_me.weight" in state_dict
    assert "exclude_me.weight_scale" not in state_dict
    assert state_dict["exclude_me.weight"].dtype == torch.float32

    print("Verification passed!")

def main():
    test_dir = "tests/temp_sdnq_verification"
    os.makedirs(test_dir, exist_ok=True)
    
    input_path = os.path.join(test_dir, "input.safetensors")
    output_path = os.path.join(test_dir, "output.safetensors")
    
    try:
        # 1. Create dummy model
        create_dummy_model(input_path)
        
        # 2. Run CLI
        cmd = [
            sys.executable, "-m", "convert_to_quant.cli.run_sdnq",
            "--input", input_path,
            "--output", output_path,
            "--dtype", "int8",
            "--verbose"
        ]
        
        # We also want to test config file support for exclusion
        config_path = os.path.join(test_dir, "config.json")
        config = {
            "weights_dtype": "int8",
            "modules_to_not_convert": ["exclude_me"]
        }
        with open(config_path, "w") as f:
            json.dump(config, f)
            
        cmd += ["--config", config_path]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        
        if result.returncode != 0:
            raise RuntimeError("CLI execution failed")
            
        # 3. Verify
        verify_output(output_path)
        
    finally:
        # Cleanup
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned up {test_dir}")

if __name__ == "__main__":
    main()
