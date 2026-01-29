import argparse
import os
import torch
import json
import logging
from safetensors.torch import load_file, save_file
from ..converters.sdnq_transform import convert_state_dict

def setup_logging(verbose: bool):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

def main():
    parser = argparse.ArgumentParser(description="SDNQ Quantization CLI (Stateless)")
    parser.add_argument("-i", "--input", type=str, required=True, help="Input safetensors file")
    parser.add_argument("-o", "--output", type=str, help="Output safetensors file (auto-generated if not provided)")
    parser.add_argument("--config", type=str, help="Path to JSON config file")
    
    # Common SDNQ settings as CLI flags (used if no config file or as overrides)
    parser.add_argument("--dtype", type=str, default="int8", help="Target weight dtype (e.g., int8, int4, fp8, float8_e4m3fn)")
    parser.add_argument("--group_size", type=int, default=0, help="Group size for quantization (0 for auto)")
    parser.add_argument("--use_svd", action="store_true", help="Enable SVD-based quantization")
    parser.add_argument("--svd_rank", type=int, default=32, help="SVD rank")
    parser.add_argument("--svd_steps", type=int, default=8, help="SVD iterations")
    parser.add_argument("--use_quantized_matmul", action="store_true", help="Optimize for quantized matmul")
    parser.add_argument("--use_stochastic_rounding", action="store_true", help="Use stochastic rounding")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Load config
    config = {}
    if args.config:
        print(f"Loading config from {args.config}...")
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default config from flags
        config = {
            "weights_dtype": args.dtype,
            "group_size": args.group_size,
            "use_svd": args.use_svd,
            "svd_rank": args.svd_rank,
            "svd_steps": args.svd_steps,
            "use_quantized_matmul": args.use_quantized_matmul,
            "use_stochastic_rounding": args.use_stochastic_rounding,
            "verbose": args.verbose,
            "modules_to_not_convert": [],
            "modules_dtype_dict": {}
        }
    
    if not args.output:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_sdnq_{config.get('weights_dtype', 'quant')}{ext}"

    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        return

    print(f"Loading model: {args.input}")
    state_dict = load_file(args.input)
    
    print(f"Quantizing model with SDNQ (dtype={config.get('weights_dtype')}, svd={config.get('use_svd')})...")
    new_state_dict = convert_state_dict(state_dict, config)
    
    print(f"Saving quantized model to: {args.output}")
    # Ensure directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    save_file(new_state_dict, args.output)
    print("Conversion complete!")

if __name__ == "__main__":
    main()
