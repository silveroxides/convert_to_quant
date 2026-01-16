#!/usr/bin/env python3
"""
Calibrate FP8 Activation Scales

Computes input_scale values for FP8 quantized models using simulated PTQ
(Post-Training Quantization) calibration.

Supports two modes:
1. Random calibration: Generate random inputs (fast, approximate)
2. LoRA-informed calibration: Use RL-extracted LoRA weights as input directions
   (more accurate for models commonly used with that LoRA)

Usage:
    python calibrate_activation_scales.py model.safetensors -o calibrated.safetensors
    python calibrate_activation_scales.py model.safetensors --lora lora.safetensors -o calibrated.safetensors
"""

import argparse
import gc
import json
import os
import sys
from collections import OrderedDict

import torch
from safetensors.torch import load_file, save_file

# FP8 maximum value
FP8_E4M3_MAX = 448.0

def infer_block_size(weight_shape: tuple, scale_shape: tuple) -> int:
    """
    Infer block_size from weight and scale tensor shapes.

    For 2D block scales: scale_shape = (M // block_size, N // block_size)

    Args:
        weight_shape: Shape of weight tensor (M, N)
        scale_shape: Shape of scale tensor

    Returns:
        Inferred block_size, or None if not determinable
    """
    if len(scale_shape) != 2 or len(weight_shape) != 2:
        return None

    M, N = weight_shape
    scale_M, scale_N = scale_shape

    if scale_M == 0 or scale_N == 0:
        return None

    # Block size along M dimension
    if M % scale_M == 0:
        block_size_m = M // scale_M
    else:
        return None

    # Block size along N dimension (should match M)
    if N % scale_N == 0:
        block_size_n = N // scale_N
    else:
        return None

    # For square blocks, both should be equal
    if block_size_m == block_size_n:
        return block_size_m

    # Non-square blocks - use M dimension
    return block_size_m

def dequantize_fp8_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_size: int = None,
    target_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Dequantize FP8 weight tensor using scale.

    Handles:
    - TensorCoreFP8Layout: scalar scale
    - RowWiseFP8Layout: scale shape (M,)
    - BlockWiseFP8Layout: scale shape (M//block_size, N//block_size)

    Ported from ComfyUI-QuantOps/fp8_ops.py _dequantize_weight().

    Args:
        weight: FP8 weight tensor (M, N)
        scale: Scale tensor for dequantization
        block_size: Block size (required for 2D block scales)
        target_dtype: Target dtype for output

    Returns:
        Dequantized weight tensor in target_dtype
    """
    M, N = weight.shape
    scale = scale.to(torch.float32)

    # Scalar scale (tensor-wise)
    if scale.ndim == 0 or (scale.ndim == 1 and scale.numel() == 1):
        return weight.to(target_dtype) * scale.item()

    # Row-wise scale: shape (M,)
    if scale.ndim == 1 and scale.shape[0] == M:
        scale_broadcast = scale.unsqueeze(1).to(dtype=target_dtype)
        return weight.to(target_dtype) * scale_broadcast

    # Block-wise scale: shape (M//block_size, N//block_size)
    if scale.ndim == 2:
        # Infer block_size if not provided
        if block_size is None:
            block_size = infer_block_size((M, N), scale.shape)

        if block_size is not None and M % block_size == 0 and N % block_size == 0:
            # Reshape to blocks: (M//bs, bs, N//bs, bs) -> (M//bs, N//bs, bs, bs)
            qdata_blocked = weight.reshape(
                M // block_size, block_size, N // block_size, block_size
            )
            qdata_blocked = qdata_blocked.permute(0, 2, 1, 3)

            # Broadcast scale: (M//bs, N//bs) -> (M//bs, N//bs, 1, 1)
            scale_broadcast = scale.unsqueeze(-1).unsqueeze(-1).to(dtype=target_dtype)

            # Dequantize per-block
            dequant = qdata_blocked.to(target_dtype) * scale_broadcast

            # Reshape back: (M//bs, N//bs, bs, bs) -> (M, N)
            return dequant.permute(0, 2, 1, 3).reshape(M, N)

    # Fallback: try simple broadcast
    return weight.to(target_dtype) * scale.to(dtype=target_dtype)

def compute_activation_scale(
    weight: torch.Tensor,
    in_features: int,
    calib_samples: int = 64,
    seed: int = 42,
    percentile: float = 99.9,
    lora_A: torch.Tensor = None,
    weight_scale: torch.Tensor = None,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Compute input_scale for a layer using calibration data.

    Follows ComfyUI QUANTIZATION.md PTQ approach:
    1. Generate calibration data (random or LoRA-informed)
    2. Compute activation outputs: X @ W.T
    3. Get absmax (or percentile) of activations
    4. Derive input_scale = absmax / fp8_max

    Args:
        weight: Layer weight tensor (out_features, in_features)
        in_features: Input dimension
        calib_samples: Number of calibration samples (used for random mode)
        seed: Random seed for reproducibility
        percentile: Use this percentile of absmax (99.9 avoids outliers)
        lora_A: Optional LoRA_A tensor (rank, in_features) for informed calibration
        weight_scale: Weight scale tensor for dequantizing FP8 weights
        device: Device to run computation on (default: CPU, use 'cuda' for GPU)

    Returns:
        input_scale as scalar tensor (float32) on CPU
    """
    # Determine device
    if device is None:
        device = torch.device('cpu')

    # For reproducibility with CUDA, we need to handle the generator appropriately
    if device.type == 'cuda':
        generator = torch.Generator(device='cpu').manual_seed(seed)
    else:
        generator = torch.Generator().manual_seed(seed)

    if lora_A is not None:
        # LoRA-informed calibration: use LoRA_A rows as input directions
        # These represent the input patterns that were important during RL training
        x_base = lora_A.to(dtype=torch.float32, device=device)

        # Expand with noise to cover nearby directions
        # Generate noise on CPU then move (for reproducibility)
        noise1 = torch.randn(x_base.shape, generator=generator, device='cpu').to(device)
        noise2 = torch.randn(x_base.shape, generator=generator, device='cpu').to(device)
        x_calib = torch.cat([
            x_base,
            x_base + 0.1 * noise1,
            x_base + 0.2 * noise2,
            x_base * -1,  # Also test negative directions
        ])

        # Normalize to unit variance (like random inputs would have)
        x_calib = x_calib / x_calib.std().clamp(min=1e-6)
    else:
        # Random calibration (original approach)
        # Generate on CPU for reproducibility, then move to target device
        x_calib = torch.randn(
            calib_samples,
            in_features,
            dtype=torch.float32,
            generator=generator,
        ).to(device)

    # Move weight to target device
    weight = weight.to(device)
    if weight_scale is not None:
        weight_scale = weight_scale.to(device)

    # Dequantize weight properly using the new function
    if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        if weight_scale is not None:
            w_float = dequantize_fp8_weight(weight, weight_scale, target_dtype=torch.float32)
        else:
            w_float = weight.to(torch.float32)
    else:
        w_float = weight.to(torch.float32)

    # Simulate forward pass: activation = X @ W.T
    with torch.no_grad():
        activations = x_calib @ w_float.t()

    # Get absmax (or percentile to avoid outliers)
    abs_vals = activations.abs().flatten()
    if percentile < 100.0:
        k = int(len(abs_vals) * percentile / 100.0)
        k = max(1, min(k, len(abs_vals)))
        absmax = abs_vals.kthvalue(k).values
    else:
        absmax = abs_vals.max()

    # Compute input_scale = absmax / fp8_max
    input_scale = (absmax / FP8_E4M3_MAX).clamp(min=1e-12)

    # Return on CPU (output should be CPU tensor for saving)
    return input_scale.to(torch.float32).cpu()


# Known prefixes to strip when normalizing layer names
MODEL_PREFIXES = [
    "model.diffusion_model.",  # ComfyUI-saved full models
    "diffusion_model.",        # Standard diffusion model prefix
    "transformer.",            # Transformer models
    "base_model.model.",       # HuggingFace PEFT format
    "unet.",                   # Diffusers UNet
]

# Known LoRA key prefixes
LORA_PREFIXES = [
    "lora_unet_",              # Kohya/A1111 UNet format
    "lora_transformer_",       # OneTrainer transformer format
    "lora_te_",                # Text encoder
    "lora_te1_",               # SDXL TE1
    "lora_te2_",               # SDXL TE2
    "lycoris_",                # SimpleTuner LyCORIS
]

def normalize_layer_name(name: str) -> str:
    """
    Normalize a layer name for matching.

    Strips known prefixes, converts between dots and underscores,
    and returns a canonical form for comparison.

    Args:
        name: Original layer name (model or LoRA key)

    Returns:
        Normalized layer name suitable for matching
    """
    # Strip model prefixes first
    for prefix in MODEL_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Strip LoRA prefixes
    for prefix in LORA_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    # Normalize: convert underscores to dots for consistent matching
    normalized = name.replace("_", ".")

    return normalized.lower()

def build_lora_key_map(model_keys: list, lora_keys: dict) -> dict:
    """
    Build a mapping from model layer names to LoRA layer data.

    Uses normalized layer names for matching, supporting various
    prefix and naming conventions (Kohya, diffusers, OneTrainer, etc.).

    Args:
        model_keys: List of model layer base names (e.g., "double_blocks.0.img_attn.proj")
        lora_keys: Dict from load_lora_tensors() mapping LoRA base names to data

    Returns:
        Dict mapping model layer base names to LoRA data
    """
    # Build normalized -> original mapping for LoRA keys
    lora_normalized = {}
    for lora_key, lora_data in lora_keys.items():
        norm = normalize_layer_name(lora_key)
        lora_normalized[norm] = (lora_key, lora_data)

    # Match model keys to LoRA keys
    key_map = {}
    for model_key in model_keys:
        norm_model = normalize_layer_name(model_key)

        # Exact match on normalized name
        if norm_model in lora_normalized:
            key_map[model_key] = lora_normalized[norm_model][1]
            continue

        # Try without trailing components (e.g., "out_proj" vs "out.proj")
        # This handles minor naming variations
        for norm_lora, (orig_lora, lora_data) in lora_normalized.items():
            # Check if one is a suffix of the other (handles prefix differences)
            if norm_model.endswith(norm_lora) or norm_lora.endswith(norm_model):
                key_map[model_key] = lora_data
                break

    return key_map

def load_lora_tensors(lora_path: str) -> dict:
    """
    Load LoRA tensors and organize by base layer name.

    Handles various naming conventions:
    - lora_A / lora_B (standard)
    - lora_down / lora_up (Kohya)
    - lora.down / lora.up
    - .down / .up (with alpha)

    Returns dict mapping base_name -> {'lora_A': tensor, 'lora_B': tensor, 'alpha': tensor}
    """
    lora_tensors = load_file(lora_path)
    lora_by_layer = {}

    # Suffixes for "down" tensor (input projection, A matrix)
    down_suffixes = [
        '.lora_A.weight', '.lora_down.weight', '.lora.down.weight',
        '.lora_A', '.lora_down', '.lora.down',
        '.down.weight', '.down',
    ]

    # Suffixes for "up" tensor (output projection, B matrix)
    up_suffixes = [
        '.lora_B.weight', '.lora_up.weight', '.lora.up.weight',
        '.lora_B', '.lora_up', '.lora.up',
        '.up.weight', '.up',
    ]

    # Alpha suffixes
    alpha_suffixes = ['.alpha', '.lora_alpha']

    for key, tensor in lora_tensors.items():
        # Check for down/A tensor
        for suffix in down_suffixes:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base not in lora_by_layer:
                    lora_by_layer[base] = {}
                lora_by_layer[base]['lora_A'] = tensor
                break

        # Check for up/B tensor
        for suffix in up_suffixes:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base not in lora_by_layer:
                    lora_by_layer[base] = {}
                lora_by_layer[base]['lora_B'] = tensor
                break

        # Check for alpha
        for suffix in alpha_suffixes:
            if key.endswith(suffix):
                base = key[:-len(suffix)]
                if base not in lora_by_layer:
                    lora_by_layer[base] = {}
                lora_by_layer[base]['alpha'] = tensor
                break

    return lora_by_layer

def calibrate_model(
    tensors: dict,
    calib_samples: int = 64,
    seed: int = 42,
    percentile: float = 99.9,
    verbose: bool = True,
    lora_tensors: dict = None,
    device: str = None,
) -> dict:
    """
    Compute calibrated input_scale for all FP8 layers in a model.

    Args:
        tensors: Model tensors from safetensors
        calib_samples: Number of calibration samples per layer
        seed: Random seed
        percentile: Percentile for absmax (99.9 avoids outliers)
        verbose: Print progress
        lora_tensors: Optional LoRA tensors organized by layer (from load_lora_tensors)
        device: Device to run on ('cpu', 'cuda', 'cuda:0', etc.). Default: auto-detect CUDA if available.

    Returns:
        Dict mapping layer base names to computed input_scale values
    """
    # Auto-detect device if not specified
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    if verbose:
        print(f"  Using device: {device}")

    scales = {}
    lora_used = 0
    random_used = 0

    # Find all weight layers
    weight_keys = [k for k in tensors.keys() if k.endswith(".weight")]

    # Build list of base names for FP8 layers
    fp8_base_names = []
    for key in weight_keys:
        weight = tensors[key]
        if weight.ndim != 2:
            continue
        base_name = key[:-7]  # Remove ".weight"
        is_fp8_weight = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        has_weight_scale = f"{base_name}.weight_scale" in tensors or f"{base_name}.scale_weight" in tensors
        if is_fp8_weight or has_weight_scale:
            fp8_base_names.append(base_name)

    # Build LoRA key map if LoRA tensors are provided
    lora_key_map = {}
    if lora_tensors:
        lora_key_map = build_lora_key_map(fp8_base_names, lora_tensors)
        if verbose:
            print(f"  LoRA key map: matched {len(lora_key_map)} of {len(fp8_base_names)} layers")

    for key in weight_keys:
        weight = tensors[key]

        # Only process 2D weights (linear layers)
        if weight.ndim != 2:
            continue

        out_features, in_features = weight.shape
        base_name = key[:-7]  # Remove ".weight"

        # Check if this layer has FP8 weights or weight_scale (indicating quantization)
        is_fp8_weight = weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        has_weight_scale = f"{base_name}.weight_scale" in tensors or f"{base_name}.scale_weight" in tensors

        if not (is_fp8_weight or has_weight_scale):
            continue

        # Check if we have LoRA for this layer (using the key map)
        lora_A = None
        if base_name in lora_key_map:
            lora_data = lora_key_map[base_name]
            if 'lora_A' in lora_data:
                lora_A = lora_data['lora_A']

        mode = "LoRA" if lora_A is not None else "random"
        if lora_A is not None:
            lora_used += 1
        else:
            random_used += 1

        if verbose:
            print(f"  Calibrating: {base_name} ({out_features}x{in_features}) [{mode}]")

        # Find weight_scale for dequantization
        weight_scale = None
        if f"{base_name}.weight_scale" in tensors:
            weight_scale = tensors[f"{base_name}.weight_scale"]
        elif f"{base_name}.scale_weight" in tensors:
            weight_scale = tensors[f"{base_name}.scale_weight"]

        # Compute calibrated input_scale
        input_scale = compute_activation_scale(
            weight, in_features, calib_samples, seed, percentile, lora_A, weight_scale, device
        )

        scales[base_name] = input_scale

        if verbose:
            print(f"    -> input_scale = {input_scale.item():.6f}")

        # Clean up GPU memory after each layer (following convert_to_quant pattern)
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if verbose and lora_tensors:
        print(f"\n  LoRA-informed: {lora_used} layers, Random: {random_used} layers")

    return scales

def patch_model_with_scales(
    tensors: dict,
    scales: dict,
) -> dict:
    """
    Create new model tensors with calibrated input_scale values.

    Handles both legacy (scale_input) and comfy_quant (.input_scale + .comfy_quant) formats.
    For comfy_quant models, also updates the layer metadata to indicate calibration.

    Args:
        tensors: Original model tensors
        scales: Dict of base_name -> input_scale from calibrate_model()

    Returns:
        New tensors dict with input_scale values replaced/added
    """
    output = OrderedDict()

    # First pass: copy all tensors
    for key, tensor in tensors.items():
        output[key] = tensor

    # Second pass: add/replace input_scale and update comfy_quant metadata
    for base_name, scale in scales.items():
        # Key names for both formats
        new_key = f"{base_name}.input_scale"
        legacy_key = f"{base_name}.scale_input"
        comfy_quant_key = f"{base_name}.comfy_quant"

        # Remove legacy key if present (migrate to new format)
        if legacy_key in output:
            del output[legacy_key]

        # Set calibrated input_scale as scalar float32
        output[new_key] = scale.to(torch.float32).detach().clone()

        # If this layer has comfy_quant metadata, update it to indicate calibration
        if comfy_quant_key in output:
            try:
                # Decode existing comfy_quant metadata
                cq_tensor = output[comfy_quant_key]
                cq_bytes = cq_tensor.numpy().tobytes()
                cq_data = json.loads(cq_bytes.decode('utf-8'))

                # Add calibration flag
                cq_data["input_scale_calibrated"] = True

                # Re-encode and store
                new_bytes = json.dumps(cq_data).encode('utf-8')
                output[comfy_quant_key] = torch.tensor(
                    list(new_bytes), dtype=torch.uint8
                )
            except Exception:
                # If parsing fails, just leave comfy_quant unchanged
                pass

    return output

def main():
    parser = argparse.ArgumentParser(
        description="Calibrate FP8 activation scales using simulated PTQ",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch model with calibrated scales (random mode)
  python calibrate_activation_scales.py model_fp8.safetensors -o model_calibrated.safetensors

  # LoRA-informed calibration (uses RL-extracted LoRA as input directions)
  python calibrate_activation_scales.py model_fp8.safetensors --lora rl_lora.safetensors -o calibrated.safetensors

  # Export scales to JSON (for inspection or manual editing)
  python calibrate_activation_scales.py model_fp8.safetensors --json scales.json

  # Use more calibration samples for better accuracy
  python calibrate_activation_scales.py model_fp8.safetensors -o calibrated.safetensors --samples 256
""",
    )
    parser.add_argument(
        "input",
        help="Input safetensors model file",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output safetensors file with calibrated input_scale values",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        help="Export computed scales to JSON file",
    )
    parser.add_argument(
        "--lora",
        dest="lora_path",
        help="LoRA file for informed calibration (uses LoRA_A as input directions)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=64,
        help="Number of calibration samples per layer for random mode (default: 64)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=99.9,
        help="Percentile for absmax computation (default: 99.9, use 100 for true max)",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    if not args.output and not args.json_output:
        parser.error("Either --output or --json must be specified")

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    verbose = not args.quiet

    if verbose:
        print(f"Loading model: {args.input}")

    tensors = load_file(args.input)

    if verbose:
        print(f"  Total tensors: {len(tensors)}")

    # Load LoRA if specified
    lora_tensors = None
    if args.lora_path:
        if not os.path.exists(args.lora_path):
            print(f"Error: LoRA file not found: {args.lora_path}")
            sys.exit(1)

        if verbose:
            print(f"\nLoading LoRA: {args.lora_path}")

        lora_tensors = load_lora_tensors(args.lora_path)

        if verbose:
            print(f"  LoRA layers found: {len(lora_tensors)}")
            if lora_tensors:
                sample_layer = next(iter(lora_tensors.values()))
                if 'lora_A' in sample_layer:
                    rank = sample_layer['lora_A'].shape[0]
                    print(f"  LoRA rank: {rank}")

    if verbose:
        mode = "LoRA-informed" if lora_tensors else "random"
        print(f"\nCalibrating ({mode} mode, {args.samples} samples, seed={args.seed})...")

    scales = calibrate_model(
        tensors,
        calib_samples=args.samples,
        seed=args.seed,
        percentile=args.percentile,
        verbose=verbose,
        lora_tensors=lora_tensors,
    )

    if verbose:
        print(f"\nCalibrated {len(scales)} layers")

    # Export to JSON if requested
    if args.json_output:
        json_data = {k: v.item() for k, v in scales.items()}
        with open(args.json_output, "w") as f:
            json.dump(json_data, f, indent=2)
        if verbose:
            print(f"Exported scales to: {args.json_output}")

    # Patch model if output specified
    if args.output:
        if verbose:
            print(f"\nPatching model with calibrated scales...")

        patched = patch_model_with_scales(tensors, scales)

        if verbose:
            print(f"Saving to: {args.output}")

        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        save_file(patched, args.output)

        if verbose:
            print("Done!")

if __name__ == "__main__":
    main()
