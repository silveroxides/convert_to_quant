#!/usr/bin/env python
"""
Utility script to modify the 'format' field in .comfy_quant metadata tensors.
Allows renaming quantization formats in already-quantized models.

Usage:
    python rename_quant_format.py -i model.safetensors -o patched.safetensors --from float8_e4m3fn --to float8_e4m3fn_block3d
    python rename_quant_format.py -i model.safetensors -o patched.safetensors --from float8_e4m3fn --to float8_e4m3fn_block3d --layers "double_blocks.*"
"""
import argparse
import os
import re
import json
import torch
from safetensors import safe_open
from safetensors.torch import save_file


def tensor_to_dict(tensor_data: torch.Tensor) -> dict:
    """Decode metadata from tensor bytes to dict."""
    raw_bytes = tensor_data.numpy().tobytes()
    if raw_bytes:
        try:
            return json.loads(raw_bytes.rstrip(b'\x00').decode('utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}
    return {}


def dict_to_tensor(data_dict: dict) -> torch.Tensor:
    """Encode metadata dict to tensor bytes."""
    json_bytes = json.dumps(data_dict).encode('utf-8')
    return torch.tensor(list(json_bytes), dtype=torch.uint8)


def rename_quant_format(input_file: str, output_file: str, from_format: str, to_format: str, 
                        layer_pattern: str = None, dry_run: bool = False):
    """
    Rename the 'format' field in .comfy_quant metadata tensors.
    
    Args:
        input_file: Input safetensors file
        output_file: Output safetensors file
        from_format: Format to match and replace
        to_format: New format value
        layer_pattern: Optional regex to filter layers (applied to base layer name)
        dry_run: If True, only print what would be changed without saving
    """
    print(f"Loading: {input_file}")
    print(f"Renaming format: '{from_format}' -> '{to_format}'")
    if layer_pattern:
        print(f"Layer filter: {layer_pattern}")
    print("-" * 60)
    
    layer_regex = re.compile(layer_pattern) if layer_pattern else None
    
    tensors = {}
    with safe_open(input_file, framework="pt", device='cpu') as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    
    modified_count = 0
    skipped_count = 0
    
    comfy_quant_keys = [k for k in tensors.keys() if k.endswith('.comfy_quant')]
    print(f"Found {len(comfy_quant_keys)} .comfy_quant tensors")
    
    for key in comfy_quant_keys:
        base_name = key[:-11]  # Remove '.comfy_quant'
        
        # Check layer filter
        if layer_regex and not layer_regex.search(base_name):
            continue
        
        # Decode metadata
        meta = tensor_to_dict(tensors[key])
        if not meta:
            print(f"  Warning: Could not decode metadata for {key}")
            continue
        
        current_format = meta.get('format')
        if current_format != from_format:
            skipped_count += 1
            continue
        
        # Modify format
        meta['format'] = to_format
        print(f"  {base_name}: '{from_format}' -> '{to_format}'")
        
        if not dry_run:
            tensors[key] = dict_to_tensor(meta)
        modified_count += 1
    
    print("-" * 60)
    print(f"Modified: {modified_count} layers")
    print(f"Skipped (different format): {skipped_count} layers")
    
    if dry_run:
        print("\n[DRY RUN] No changes saved.")
        return
    
    if modified_count > 0:
        print(f"\nSaving to: {output_file}")
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        save_file(tensors, output_file)
        print("Done!")
    else:
        print("\nNo changes to save.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename quantization format in .comfy_quant metadata tensors.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True, help="Input safetensors file")
    parser.add_argument("-o", "--output", required=True, help="Output safetensors file")
    parser.add_argument("--from", dest="from_format", required=True, 
                        help="Format value to match (e.g., 'float8_e4m3fn')")
    parser.add_argument("--to", dest="to_format", required=True,
                        help="New format value (e.g., 'float8_e4m3fn_block3d')")
    parser.add_argument("--layers", type=str, default=None,
                        help="Regex pattern to filter layers (optional, matches base layer name)")
    parser.add_argument("--dry-run", action='store_true',
                        help="Show what would be changed without saving")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    if os.path.abspath(args.input) == os.path.abspath(args.output) and not args.dry_run:
        print("Error: Output file cannot be same as input.")
        return
    
    rename_quant_format(
        args.input, args.output, 
        args.from_format, args.to_format,
        args.layers, args.dry_run
    )


if __name__ == "__main__":
    main()
