#!/usr/bin/env python3
"""
Repair NVFP4 metadata nesting for models quantized with broken format.

Fixes: Missing "format_version" and "layers" wrapper in _quantization_metadata.
"""
import argparse
import json
import os
import sys
from safetensors import safe_open
from safetensors.torch import save_file


def repair_nvfp4_metadata(input_file: str, output_file: str = None, dry_run: bool = False) -> bool:
    """
    Repair NVFP4 metadata nesting.
    
    Returns True if repair was needed, False if already correct.
    """
    if output_file is None:
        output_file = input_file  # In-place repair
    
    with safe_open(input_file, framework="pt") as f:
        metadata = f.metadata() or {}
        
        if "_quantization_metadata" not in metadata:
            print(f"No _quantization_metadata found in {input_file}")
            return False
        
        try:
            quant_meta = json.loads(metadata["_quantization_metadata"])
        except json.JSONDecodeError:
            print(f"Invalid JSON in _quantization_metadata")
            return False
        
        # Check if already has correct structure
        if "format_version" in quant_meta and "layers" in quant_meta:
            print(f"Metadata already has correct structure (format_version + layers)")
            return False
        
        # Check if this looks like broken NVFP4 metadata (flat layer dict)
        sample_key = next(iter(quant_meta.keys()), None)
        if sample_key and isinstance(quant_meta.get(sample_key), dict):
            sample_val = quant_meta[sample_key]
            if "format" in sample_val and sample_val.get("format") == "nvfp4":
                print(f"Found broken NVFP4 metadata - needs repair")
            else:
                print(f"Unknown metadata format, skipping")
                return False
        else:
            print(f"Metadata structure not recognized")
            return False
        
        if dry_run:
            print(f"[DRY RUN] Would wrap {len(quant_meta)} layers in format_version/layers structure")
            return True
        
        # Fix: wrap in proper structure
        fixed_metadata = {"format_version": "1.0", "layers": quant_meta}
        new_file_metadata = dict(metadata)
        new_file_metadata["_quantization_metadata"] = json.dumps(fixed_metadata)
        
        # Load all tensors
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    
    # Save with fixed metadata (atomic write for in-place safety)
    print(f"Saving repaired file to: {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    # Write to temp file first, then rename (atomic on same filesystem)
    temp_file = output_file + ".tmp"
    try:
        save_file(tensors, temp_file, metadata=new_file_metadata)
        # Atomic rename (overwrites destination on Windows/Linux)
        if os.path.exists(output_file) and output_file != input_file:
            os.remove(output_file)
        os.replace(temp_file, output_file)
    except Exception as e:
        # Clean up temp file on failure
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise RuntimeError(f"Failed to save repaired file: {e}") from e
    
    print(f"Done! Repaired metadata for {len(quant_meta)} layers.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Repair NVFP4 metadata nesting for models with broken format"
    )
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("-o", "--output", help="Output file (default: overwrite input)")
    parser.add_argument("--dry-run", action="store_true", help="Check without modifying")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    repaired = repair_nvfp4_metadata(args.input, args.output, args.dry_run)
    sys.exit(0 if repaired else 1)


if __name__ == "__main__":
    main()
