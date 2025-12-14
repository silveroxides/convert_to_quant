import argparse
import os
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from typing import Dict, Tuple
from tqdm import tqdm
import gc
import math

# --- Constants and Configuration ---
AVOID_KEY_NAMES = ["norm", "bias", "embed_tokens", "shared", "patch_embedding", "audio_model.patch_embedding", "ref_conv", "control_adapter", "motion_encoder.enc.net_app", "face_encoder.conv", "pose_patch_embedding", "motion_encoder.enc.fc", "img_emb.proj", "q_norm", "motion_encoder.dec", "head.modulation", "casual_audio_encoder", "cond_encoder", "frame_packer", "norm_k", "norm_q"]
T5XXL_REMOVE_KEY_NAMES = ["decoder", "lm_head"]
QWEN_AVOID_KEY_NAMES = ["norm_added_k", "norm_added_q", "norm_k", "norm_q", "txt_norm"]
DISTILL_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "final_layer", "img_in", "txt_in"]
DISTILL_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer"]
NERF_LAYER_KEYNAMES_LARGE = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder", "txt_in"]
NERF_LAYER_KEYNAMES_SMALL = ["distilled_guidance_layer", "nerf_blocks", "nerf_image_embedder"]
RADIANCE_LAYER_KEYNAMES = ["img_in_patch", "nerf_final_layer_conv"]
WAN_LAYER_KEYNAMES = ["text_embedding", "time_embedding", "audio_model.text_embedding", "audio_model.time_embedding", "time_projection", "video_model.time_projection", "head.head", "face_encoder.out_proj", "face_adapter"]
QWEN_LAYER_KEYNAMES = ["time_text_embed", "img_in", "norm_out", "proj_out", "txt_in"]
TARGET_FP8_DTYPE = torch.float8_e4m3fn
COMPUTE_DTYPE = torch.float32
SCALE_DTYPE = torch.float32

class LearnedRoundingConverter:
    """
    Implements advanced quantization using learned adaptive rounding.
    Provides a highly effective optimization strategy.
    """
    def __init__(self, num_iter=500, top_p=0.01, min_k=1, max_k=16, scaling_mode='vector', block_size=64, **kwargs):
        self.num_iter = num_iter
        self.top_p = top_p
        self.min_k = min_k
        self.max_k = max_k
        self.scaling_mode = scaling_mode
        self.block_size = block_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.optimizer_kwargs = kwargs
        self.f8_max_val = torch.finfo(TARGET_FP8_DTYPE).max

        print(f"LearnedRoundingConverter initialized on device: {self.device}")
        print(f"  - Using optimizer: 'original'")
        print(f"  - Scaling mode: {self.scaling_mode}")
        if self.scaling_mode == 'block':
            print(f"    - Block size: {self.block_size}")

    def _optimize_original(self, W_float32: torch.Tensor, scale: torch.Tensor, U_k: torch.Tensor, Vh_k: torch.Tensor) -> torch.Tensor:
        W_rounded = (W_float32 * scale).to(TARGET_FP8_DTYPE).to(COMPUTE_DTYPE)
        W_q_refined = W_rounded.clone()
        best_loss = float('inf')
        best_tensor = None
        worse_loss_counter = 0
        curr_lr = self.optimizer_kwargs.get('lr', 0.5)

        pbar = tqdm(range(self.num_iter), desc="    Optimizing (Original)", leave=False, dynamic_ncols=True)
        for i in pbar:
            with torch.no_grad():
                current_dq = W_q_refined / scale
                error = current_dq - W_float32
                projected_error = U_k.T @ error @ Vh_k.T
                loss = torch.linalg.norm(projected_error)

            if loss.item() < best_loss and worse_loss_counter < 25:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter = 0
                curr_lr = min(curr_lr * 1.50, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 24 and worse_loss_counter < 50:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 5
                curr_lr = min(curr_lr * 1.60, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 49 and worse_loss_counter < 100:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 4
                curr_lr = min(curr_lr * 1.70, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 99 and worse_loss_counter < 150:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 2
                curr_lr = min(curr_lr * 1.80, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 149 and worse_loss_counter < 200:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 1
                curr_lr = min(curr_lr * 1.85, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 199 and worse_loss_counter < 400:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 2
                curr_lr = min(curr_lr * 1.90, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 399 and worse_loss_counter < 500:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 3
                curr_lr = min(curr_lr * 1.95, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 499 and worse_loss_counter < 600:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 4
                curr_lr = min(curr_lr * 2.0, 100.0)
            elif loss.item() < best_loss and worse_loss_counter > 599:
                best_loss = loss.item()
                best_tensor = W_q_refined.clone()
                worse_loss_counter -= 6
                curr_lr = min(curr_lr * 2.25, 100.0)
            elif loss.item() > best_loss and worse_loss_counter < 26:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.95, 1e-9)
            elif worse_loss_counter > 25 and worse_loss_counter < 76:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.925, 1e-9)
            elif worse_loss_counter > 75 and worse_loss_counter < 101:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.90, 1e-9)
            elif worse_loss_counter > 100 and worse_loss_counter < 126:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.875, 1e-9)
            elif worse_loss_counter > 125 and worse_loss_counter < 151:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.85, 1e-9)
            elif worse_loss_counter > 150 and worse_loss_counter < 201:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.875, 1e-9)
            elif worse_loss_counter > 200 and worse_loss_counter < 301:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.8875, 1e-9)
            elif worse_loss_counter > 300 and worse_loss_counter < 401:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.90, 1e-9)
            elif worse_loss_counter > 400 and worse_loss_counter < 501:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.9125, 1e-9)
            elif worse_loss_counter > 500 and worse_loss_counter < 601:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.925, 1e-9)
            elif worse_loss_counter > 600:
                worse_loss_counter += 1
                curr_lr = max(curr_lr * 0.95, 1e-9)


            pbar.set_postfix({"loss": f"{loss.item():.3e}", "best": f"{best_loss:.3e}", "lr": f"{curr_lr:.2e}", "worse_count": f"{worse_loss_counter}"})

            if loss.item() < 1e-9 or curr_lr < 2e-9 or worse_loss_counter > 1500:
                if worse_loss_counter > 1500:
                    print("      - Loss has stalled. Stopping.")
                elif curr_lr < 2e-9:
                    print("      - Learning Rate has bottomed out. Stopping.")
                else:
                    print("      - Loss is negligible. Stopping.")
                break

            with torch.no_grad():
                grad_direction = U_k @ (projected_error / loss.clamp_min(1e-20)) @ Vh_k
                W_q_refined -= curr_lr * (grad_direction * scale)

        pbar.close()
        return best_tensor if best_tensor is not None else W_q_refined

    def convert(self, W_orig: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        W_float32 = W_orig.to(self.device, dtype=COMPUTE_DTYPE)

        if torch.all(W_float32 == 0):
            print("  - Tensor is all zeros, skipping optimization.")
            quantized_tensor = torch.zeros_like(W_float32, dtype=TARGET_FP8_DTYPE)
            dequant_scale = None
            if self.scaling_mode == 'vector':
                dequant_scale = torch.ones(W_float32.shape[0], device=self.device, dtype=SCALE_DTYPE)
            elif self.scaling_mode == 'block' and W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                dequant_scale = torch.ones(out_features, num_blocks, 1, device=self.device, dtype=SCALE_DTYPE)
            else:
                dequant_scale = torch.ones(1, device=self.device, dtype=SCALE_DTYPE)
            return quantized_tensor, dequant_scale, torch.zeros_like(W_float32)

        scale = None
        compact_scale = None
        current_scaling_mode = self.scaling_mode

        if current_scaling_mode == 'block':
            if W_float32.ndim == 2 and W_float32.shape[1] > 0 and W_float32.shape[1] % self.block_size == 0:
                print(f"    - Using block scaling with block size {self.block_size}.")
                out_features, in_features = W_float32.shape
                num_blocks = in_features // self.block_size
                W_reshaped = W_float32.view(out_features, num_blocks, self.block_size)
                w_max = W_reshaped.abs().max(dim=2, keepdim=True)[0]
                compact_scale = self.f8_max_val / w_max.clamp_min_(1e-12)
                scale = compact_scale.repeat_interleave(self.block_size, dim=2).view(out_features, in_features)
            else:
                print(f"    - WARNING: Tensor shape {list(W_float32.shape)} not suitable for block size {self.block_size}. Falling back to 'tensor' scaling.")
                current_scaling_mode = 'tensor'

        if current_scaling_mode == 'vector':
            w_max = W_float32.abs().max(dim=1, keepdim=True)[0]
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale
        elif current_scaling_mode == 'tensor':
            w_max = W_float32.abs().max()
            scale = self.f8_max_val / w_max.clamp_min_(1e-12)
            compact_scale = scale

        max_rank = min(W_float32.shape)
        k = min(self.max_k, max(self.min_k, int(math.floor(self.top_p * max_rank))))
        k = min(k, max_rank)

        print(f"    - Tensor shape: {list(W_float32.shape)}, Max rank: {max_rank}. Using k={k} components.")

        if self.top_p == 1.0:
            U, _, Vh = torch.linalg.svd(W_float32, full_matrices=True, driver='gesvd')
        else:
            try:
                U, _, Vh = torch.svd_lowrank(W_float32, q=min(k + 10, max_rank), niter=4)
                Vh = Vh.T
            except torch.linalg.LinAlgError:
                print("    - svd_lowrank failed, falling back to full SVD.")
                U, _, Vh = torch.linalg.svd(W_float32, full_matrices=False)

        U_k, Vh_k = U[:, :k], Vh[:k, :]

        final_tensor_scaled = self._optimize_original(W_float32, scale, U_k, Vh_k)
        final_tensor_scaled.clamp_(-self.f8_max_val, self.f8_max_val)

        with torch.no_grad():
            W_f8 = final_tensor_scaled.to(TARGET_FP8_DTYPE)
            dequant_scale = None
            if current_scaling_mode == 'vector':
                dequant_scale = compact_scale.reciprocal().squeeze()
            elif current_scaling_mode == 'block':
                dequant_scale = compact_scale.reciprocal()
            else:
                dequant_scale = compact_scale.reciprocal().reshape(1)

            dequantized_weight_tensor = (W_f8.to(self.device, dtype=COMPUTE_DTYPE) / scale)

        del W_float32, scale, U, Vh, U_k, Vh_k, final_tensor_scaled, compact_scale
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()

        return W_f8, dequant_scale.to(device=self.device, dtype=SCALE_DTYPE), dequantized_weight_tensor

# --- Main script execution functions ---
def get_fp8_constants(fp8_dtype: torch.dtype) -> Tuple[float, float, float]:
    finfo = torch.finfo(fp8_dtype)
    return float(finfo.min), float(finfo.max), float(finfo.tiny)

FP8_MIN, FP8_MAX, FP8_MIN_POS = get_fp8_constants(TARGET_FP8_DTYPE)

def convert_to_fp8_scaled(
    input_file: str, output_file: str, t5xxl: bool, keep_distillation_large: bool,
    keep_distillation_small: bool, keep_nerf_large: bool, keep_nerf_small: bool,
    radiance: bool, wan: bool, qwen: bool, calib_samples: int, seed: int,
    **converter_kwargs
):
    print(f"Processing: {input_file}\nOutput will be saved to: {output_file}")
    print("-" * 60)
    print(f"Target FP8 format: {TARGET_FP8_DTYPE}\nFP8 Range: [{FP8_MIN}, {FP8_MAX}]")
    print("-" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed_device = device
    seed_generator = torch.Generator(device=seed_device)
    seed_generator.manual_seed(seed)

    tensors: Dict[str, torch.Tensor] = {}
    try:
        with safe_open(input_file, framework="pt", device=device) as f:
            print(f"Loading {len(f.keys())} tensors from source file...")
            for key in tqdm(f.keys(), desc="Loading tensors"):
                tensors[key] = f.get_tensor(key)
    except Exception as e:
        print(f"FATAL: Error loading '{input_file}': {e}")
        return

    converter = LearnedRoundingConverter(**converter_kwargs)

    print("\nScanning model and generating simulated calibration data...")
    calibration_data_cache = {}
    for key, tensor in tensors.items():
        if key.endswith('.weight') and tensor.ndim == 2:
            in_features = tensor.shape[1]
            if in_features not in calibration_data_cache:
                print(f"  - Found new input dimension: {in_features}.")
                calibration_data_cache[in_features] = torch.randn(calib_samples, in_features, dtype=COMPUTE_DTYPE, generator=seed_generator, device=seed_device)
    print("Simulated calibration data generated.\n")

    new_tensors: Dict[str, torch.Tensor] = {}
    weight_keys = sorted([key for key in tensors.keys() if key.endswith('.weight') and tensors[key].ndim == 2])
    total_weights = len(weight_keys)
    skipped_count = 0
    processed_count = 0

    print(f"Found {total_weights} weight tensors to potentially process.")
    print("-" * 60)

    for i, key in enumerate(weight_keys):
        process_this_key = True
        create_scale_key = True
        skip_reason = ""

        if t5xxl and any(n in key for n in T5XXL_REMOVE_KEY_NAMES):
            print(f"({i+1}/{total_weights}) Removing T5XXL decoder tensor: {key}")
            skipped_count += 1
            continue
        if t5xxl and any(n in key for n in AVOID_KEY_NAMES):
            skip_reason = "T5XXL exclusion"
            create_scale_key = False
            process_this_key = False
        if radiance and any(n in key for n in RADIANCE_LAYER_KEYNAMES):
            skip_reason = "Radiance exclusion"
            create_scale_key = False
            process_this_key = False
        if wan and any(n in key for n in AVOID_KEY_NAMES):
            skip_reason = "WAN exclusion"
            create_scale_key = False
            process_this_key = False
        if qwen and any(n in key for n in QWEN_AVOID_KEY_NAMES):
            skip_reason = "Qwen Image exclusion"
            create_scale_key = False
            process_this_key = False
        if keep_distillation_large and any(n in key for n in DISTILL_LAYER_KEYNAMES_LARGE):
            skip_reason = "Distillation layer and Flux1 keep"
            create_scale_key = True
            process_this_key = False
        if keep_distillation_small and any(n in key for n in DISTILL_LAYER_KEYNAMES_SMALL):
            skip_reason = "Distillation layer only"
            create_scale_key = True
            process_this_key = False
        if keep_nerf_large and any(n in key for n in NERF_LAYER_KEYNAMES_LARGE):
            skip_reason = "Distillation layer, NeRF layer and txt_in"
            create_scale_key = True
            process_this_key = False
        if keep_nerf_small and any(n in key for n in NERF_LAYER_KEYNAMES_SMALL):
            skip_reason = "Distillation layer and NeRF layer"
            create_scale_key = True
            process_this_key = False
        if wan and any(n in key for n in WAN_LAYER_KEYNAMES):
            skip_reason = "WAN layer keep in high"
            create_scale_key = True
            process_this_key = False
        if qwen and any(n in key for n in QWEN_LAYER_KEYNAMES):
            skip_reason = "Qwen Image layer keep in high"
            create_scale_key = True
            process_this_key = False

        if not process_this_key:
            if not create_scale_key:
                print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                new_tensors[key] = tensors[key]
                skipped_count += 1
                continue
            else:
                print(f"({i+1}/{total_weights}) Skipping tensor: {key} (Reason: {skip_reason})")
                new_tensors[key] = tensors[key]
                base_name = key[:key.rfind('.weight')]
                new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
                skipped_count += 1
                continue

        print(f"({i+1}/{total_weights}) Processing tensor: {key}")
        processed_count += 1
        original_tensor = tensors[key]

        if original_tensor.numel() == 0 or original_tensor.ndim != 2:
            print(f"  - Skipping empty or non-2D tensor: {key}")
            new_tensors[key] = original_tensor.to(TARGET_FP8_DTYPE)
            base_name = key[:key.rfind('.weight')]
            new_tensors[f"{base_name}.scale_weight"] = torch.tensor([1.0], dtype=SCALE_DTYPE)
            continue

        q_tensor, dequant_s, dequant_w = converter.convert(original_tensor)
        new_tensors[key] = q_tensor
        base_name = key[:key.rfind('.weight')]
        bias_key = f"{base_name}.bias"
        new_tensors[f"{base_name}.scale_weight"] = dequant_s.detach().clone()

        if bias_key in tensors:
            print(f"  - Adjusting corresponding bias: {bias_key}")
            with torch.no_grad():
                original_bias = tensors[bias_key]
                in_features = original_tensor.shape[1]
                if in_features not in calibration_data_cache:
                    print(f"  - WARNING: No calibration data for bias correction.")
                    new_tensors[bias_key] = original_bias
                else:
                    X_calib_dev = calibration_data_cache[in_features].to(device=device)
                    W_orig_dev = original_tensor.to(device=device, dtype=COMPUTE_DTYPE)
                    W_dequant_dev = dequant_w.to(device=device, dtype=COMPUTE_DTYPE)
                    b_orig_dev = original_bias.to(device=device, dtype=COMPUTE_DTYPE)
                    weight_error = W_orig_dev - W_dequant_dev
                    output_error = X_calib_dev @ weight_error.T
                    bias_correction = output_error.mean(dim=0)
                    b_new = b_orig_dev - bias_correction
                    new_tensors[bias_key] = b_new.to(device=device, dtype=original_bias.dtype)
                    print(f"    - Original bias mean : {original_bias.mean().item():.6f}\n    - Corrected bias mean: {new_tensors[bias_key].mean().item():.6f}")
                    del W_orig_dev, W_dequant_dev, X_calib_dev, b_orig_dev, weight_error, output_error, bias_correction, b_new
                    if device == 'cuda': torch.cuda.empty_cache()

        if t5xxl:
            new_tensors[f"{base_name}.scale_input"] = dequant_s.detach().clone()

        print(f"    - Final Dequant Scale shape: {dequant_s.shape}\n    - Final Weight shape       : {q_tensor.shape}")
        print("-" * 60)

    for key, tensor in tensors.items():
        if (any(n in key for n in T5XXL_REMOVE_KEY_NAMES) and t5xxl):
            continue
        if key not in new_tensors:
            new_tensors[key] = tensor

    new_tensors["scaled_fp8"] = torch.empty((0), dtype=TARGET_FP8_DTYPE) if t5xxl else torch.empty((2), dtype=TARGET_FP8_DTYPE)

    print(f"Saving {len(new_tensors)} tensors to {output_file}")
    try:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        save_file(new_tensors, output_file)
        print("Conversion complete!")
    except Exception as e:
        print(f"FATAL: Error saving file '{output_file}': {e}")
        return

    print("-" * 60)
    print("Summary:")
    print(f"  - Original tensor count : {len(tensors)}\n  - Weights processed     : {processed_count}\n  - Weights skipped       : {skipped_count}\n  - Final tensor count    : {len(new_tensors)}")
    print("-" * 60)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description=f"Convert safetensors weights to Scaled {TARGET_FP8_DTYPE} format.")
    parser.add_argument("--input", type=str, required=True, help="Input safetensors file path.")
    parser.add_argument("--output", type=str, help="Output safetensors file path. Auto-generated if not provided.")
    parser.add_argument("--t5xxl", action='store_true', help="Apply exclusions for T5XXL Text Encoder models.")
    parser.add_argument("--keep_distillation_large", action='store_true', help="Exclude known distillation layers and other sensitive.")
    parser.add_argument("--keep_distillation_small", action='store_true', help="Exclude known distillation layers.")
    parser.add_argument("--keep_nerf_large", action='store_true', help="Exclude known NeRF layers, distillation layers and txt_in.")
    parser.add_argument("--keep_nerf_small", action='store_true', help="Exclude known NeRF layers and distillation layers.")
    parser.add_argument("--radiance", action='store_true', help="Exclude known Radiance Field layers.")
    parser.add_argument("--wan", action='store_true', help="Exclude known WAN layers.")
    parser.add_argument("--qwen", action='store_true', help="Exclude known Qwen Image layers.")
    parser.add_argument("--scaling_mode", type=str, default="vector", choices=["vector", "tensor", "block"], help="Quantization scaling mode.")
    parser.add_argument("--block_size", type=int, default=64, help="Block size for 'block' scaling mode.")
    parser.add_argument("--calib_samples", type=int, default=3072, help="Number of random samples for bias correction.")
    parser.add_argument("--manual_seed", type=int, default=42, help="Set a manual seed for reproducibility. Use -1 for random.")
    parser.add_argument("--num_iter", type=int, default=500, help="Total optimization iterations per tensor.")
    parser.add_argument("--lr", type=float, default=1e-2, help="[Original] Initial learning rate.")
    parser.add_argument("--top_p", type=float, default=0.01, help="Proportion of principal components (SVD) to use.")
    parser.add_argument("--min_k", type=int, default=1, help="Minimum number of principal components.")
    parser.add_argument("--max_k", type=int, default=16, help="Maximum number of principal components.")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return

    try:
        _ = torch.zeros(1, dtype=TARGET_FP8_DTYPE, device='cuda' if torch.cuda.is_available() else 'cpu')
    except (RuntimeError, TypeError):
        print("Error: This hardware/PyTorch version does not support the target FP8 dtype.")
        return

    if not args.output:
        base = os.path.splitext(args.input)[0]
        fp8_str = TARGET_FP8_DTYPE.__str__().split('.')[-1]
        flags = "".join(["_t5" if args.t5xxl else "", "_nodist_l" if args.keep_distillation_large else "", "_nodist_s" if args.keep_distillation_small else "", "_nonerf_l" if args.keep_nerf_large else "", "_nonerf_s" if args.keep_nerf_small else "", "_norad" if args.radiance else ""])
        output_file = f"{base}_{fp8_str}_{args.scaling_mode}{flags}_k{args.min_k}-{args.max_k}_p{args.top_p}_lr{args.lr}.safetensors"
    else:
        output_file = args.output

    if os.path.abspath(args.input) == os.path.abspath(output_file):
        print("Error: Output file cannot be same as input.")
        return

    seed = int(torch.randint(0, 2**32 - 1, ()).item()) if args.manual_seed == -1 else args.manual_seed
    print(f"Using seed: {seed}")

    converter_kwargs = {k: v for k, v in vars(args).items() if k not in ['input', 'output', 't5xxl', 'keep_distillation_large', 'keep_distillation_small', 'keep_nerf_large', 'keep_nerf_small', 'radiance', 'wan', 'qwen', 'calib_samples', 'manual_seed']}

    convert_to_fp8_scaled(
        args.input, output_file, args.t5xxl, args.keep_distillation_large,
        args.keep_distillation_small, args.keep_nerf_large, args.keep_nerf_small,
        args.radiance, args.wan, args.qwen, args.calib_samples, seed,
        **converter_kwargs
    )

if __name__ == "__main__":
    main()
