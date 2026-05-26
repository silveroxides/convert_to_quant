import os
from convert_to_quant import quantize

def main():
    # Make sure we have a dummy input safetensors
    import torch
    from safetensors.torch import save_file

    input_path = "dummy_model.safetensors"
    output_path = "dummy_model_out.safetensors"

    # Create a dummy model
    tensors = {
        "layer1.weight": torch.randn(128, 128)
    }
    save_file(tensors, input_path)

    # Try calling quantize programmatically
    quantize(
        input=input_path,
        output=output_path,
        int8=True,
        block_size=64,
        simple=True
    )

    print("Quantize successful!")

    if os.path.exists(input_path):
        os.remove(input_path)
    if os.path.exists(output_path):
        os.remove(output_path)

if __name__ == "__main__":
    main()
