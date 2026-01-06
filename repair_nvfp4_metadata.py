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


def repair_nvfp4_metadata(
    input_file: str, 
    output_file: str = None, 
    dry_run: bool = False,
    replace_original: bool = False,
) -> bool:
    """
    Repair NVFP4 metadata nesting.
    
    Returns True if repair was needed, False if already correct.
    """
    # Determine output path
    if replace_original:
        output_file = input_file  # Will rename original to .bak first
    elif output_file is None:
        # Default: add _repaired suffix
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_repaired{ext}"
    
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
            print(f"[DRY RUN] Output would be: {output_file}")
            return True
        
        # Fix: wrap in proper structure
        fixed_metadata = {"format_version": "1.0", "layers": quant_meta}
        new_file_metadata = dict(metadata)
        new_file_metadata["_quantization_metadata"] = json.dumps(fixed_metadata)
        
        # Load all tensors and fix issues:
        # 1. Rename old scale names to match ComfyUI convention
        # 2. Fix 1D weight_scale_2 tensors ([1] -> []) to match NVIDIA scalar format
        # NOTE: weight_scale dtype fix (uint8->float8) is NOT feasible via repair.
        #       If weight_scale is uint8, the model must be re-quantized.
        tensors = {}
        renamed_scales = 0
        scalar_scales_fixed = 0
        dtype_warning_shown = False
        
        all_keys = list(f.keys())
        for key in all_keys:
            tensor = f.get_tensor(key)
            new_key = key
            
            # Rename old naming convention to new:
            #   Old: .block_scale -> New: .weight_scale (should be float8)
            #   Old: .weight_scale (per_tensor) -> New: .weight_scale_2
            if key.endswith(".block_scale"):
                # Old block_scale becomes weight_scale
                new_key = key.replace(".block_scale", ".weight_scale")
                renamed_scales += 1
            elif key.endswith(".weight_scale"):
                # Check if this is actually per_tensor_scale (old convention)
                # by checking if there's also a block_scale for this layer
                base = key.rsplit(".weight_scale", 1)[0]
                if f"{base}.block_scale" in all_keys:
                    # Old per_tensor_scale becomes weight_scale_2
                    new_key = key.replace(".weight_scale", ".weight_scale_2")
                    renamed_scales += 1
            
            # Warn about weight_scale dtype issue (uint8 instead of float8)
            if new_key.endswith(".weight_scale") and not dtype_warning_shown:
                import torch
                if tensor.dtype == torch.uint8:
                    print("WARNING: weight_scale is uint8 (should be float8_e4m3fn).")
                    print("         This cannot be fixed by repair. Re-quantize the model instead.")
                    dtype_warning_shown = True
            
            # Fix weight_scale_2 tensors: squeeze [1] to scalar [] (NVIDIA format)
            if new_key.endswith(".weight_scale_2") and tensor.dim() == 1 and tensor.numel() == 1:
                tensor = tensor.squeeze(0)  # [1] -> []
                scalar_scales_fixed += 1
            
            tensors[new_key] = tensor
        
        if renamed_scales > 0:
            print(f"Renamed {renamed_scales} scale tensors to ComfyUI convention")
        if scalar_scales_fixed > 0:
            print(f"Fixed {scalar_scales_fixed} weight_scale_2 tensors: [1] -> [] (scalar)")
    
    # If replacing original, rename faulty file to .bak first
    backup_file = None
    if replace_original:
        backup_file = input_file + ".bak"
        print(f"Backing up original to: {backup_file}")
        os.rename(input_file, backup_file)
    
    # Save repaired file
    print(f"Saving repaired file to: {output_file}")
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    
    try:
        save_file(tensors, output_file, metadata=new_file_metadata)
    except Exception as e:
        # Restore backup on failure
        if backup_file and os.path.exists(backup_file):
            print(f"Error occurred, restoring backup...")
            os.rename(backup_file, input_file)
        raise RuntimeError(f"Failed to save repaired file: {e}") from e
    
    print(f"Done! Repaired metadata for {len(quant_meta)} layers.")
    if backup_file:
        print(f"Original backed up to: {backup_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Repair NVFP4 metadata nesting for models with broken format"
    )
    parser.add_argument("input", help="Input safetensors file")
    parser.add_argument("-o", "--output", help="Output file (default: adds _repaired suffix)")
    parser.add_argument("--replace-original", action="store_true", 
                        help="Replace original file (backs up to .bak first)")
    parser.add_argument("--dry-run", action="store_true", help="Check without modifying")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    repaired = repair_nvfp4_metadata(
        args.input, 
        args.output, 
        args.dry_run,
        args.replace_original,
    )
    sys.exit(0 if repaired else 1)


if __name__ == "__main__":
    main()
