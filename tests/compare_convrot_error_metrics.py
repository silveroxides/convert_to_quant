import torch
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight, rotate_activation, find_max_compatible_group_size

def compute_detailed_errors(y_ref, y_quant):
    # Metric calculations
    mae = torch.mean(torch.abs(y_ref - y_quant)).item()
    mse = torch.mean((y_ref - y_quant) ** 2).item()
    rmse = np.sqrt(mse)

    ref_norm = torch.norm(y_ref).item()
    err_norm = torch.norm(y_ref - y_quant).item()
    snr_db = 20 * np.log10(ref_norm / max(err_norm, 1e-12)) if err_norm > 0 else float("inf")

    # Cosine Similarity
    cos_sim = torch.nn.functional.cosine_similarity(y_ref.flatten(), y_quant.flatten(), dim=0).item()

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "SNR (dB)": snr_db,
        "Cosine Similarity": cos_sim
    }

def evaluate_case(name, out_features, in_features, num_samples=128):
    print("=" * 80)
    print(f"CASE: {name} (Shape: [{out_features}, {in_features}])")
    print("=" * 80)

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthesize weights with structured outliers to simulate sensitive diffusion channels
    W_orig = torch.randn(out_features, in_features, device=device) * 0.1
    # Add strong outlier channels to simulate real-world outlier clustering
    W_orig[:, 5] *= 50.0
    W_orig[:, 12] *= 50.0

    X = torch.randn(num_samples, in_features, device=device)
    bias = torch.randn(out_features, device=device)

    # Reference high-precision output
    Y_ref = X @ W_orig.T + bias

    # 1. Determine dynamic and static group sizes
    static_group_size = 256
    dynamic_group_size = find_max_compatible_group_size(in_features, min_group_size=256)

    print(f"  - Static ConvRot Group Size:  {static_group_size}")
    print(f"  - Dynamic ConvRot Group Size: {dynamic_group_size}")
    print("-" * 80)

    # -------------------------------------------------------------------------
    # System A: Standard Symmetric Row-wise Quantization (No ConvRot, No AdaRound)
    # -------------------------------------------------------------------------
    print("  Running System A: Standard Row-wise Quantization...")
    conv_standard = LearnedRoundingConverter(
        target_format="int8", scaling_mode="row", no_learned_rounding=True, device=device
    )
    qdata_a, scale_a, dequant_w_a, extra_a = conv_standard.convert(W_orig, calibration_data=X)
    Y_a = X @ dequant_w_a.T + bias
    metrics_a = compute_detailed_errors(Y_ref, Y_a)

    # -------------------------------------------------------------------------
    # System B: Static ConvRot (Group Size = 256) + Simple Rounding
    # -------------------------------------------------------------------------
    print(f"  Running System B: Static ConvRot (group_size={static_group_size}) + Simple Rounding...")
    conv_static = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=static_group_size,
        no_learned_rounding=True,
        device=device
    )
    qdata_b, scale_b, dequant_w_b, extra_b = conv_static.convert(W_orig, calibration_data=X)

    # Re-apply static rotation during evaluation if applied
    static_applied = (in_features % static_group_size == 0)
    if static_applied:
        H_static = build_hadamard(static_group_size, device=device, dtype=torch.float32)
        X_rot_static = rotate_activation(X, H_static, static_group_size)
        Y_b = X_rot_static @ dequant_w_b.T + bias
    else:
        Y_b = X @ dequant_w_b.T + bias

    if "bias_correction" in extra_b:
        Y_b += extra_b["bias_correction"].to(device=device)
    metrics_b = compute_detailed_errors(Y_ref, Y_b)

    # -------------------------------------------------------------------------
    # System C: Dynamic ConvRot (Determined Group Size) + Simple Rounding
    # -------------------------------------------------------------------------
    print(f"  Running System C: Dynamic ConvRot (group_size={dynamic_group_size}) + Simple Rounding...")
    conv_dynamic = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        dynamic_convrot=True,
        convrot_group_size=256, # min group size
        no_learned_rounding=True,
        device=device
    )
    qdata_c, scale_c, dequant_w_c, extra_c = conv_dynamic.convert(W_orig, calibration_data=X)

    # Re-apply dynamic rotation during evaluation if applied
    dynamic_applied = (dynamic_group_size is not None and in_features % dynamic_group_size == 0)
    if dynamic_applied:
        H_dynamic = build_hadamard(dynamic_group_size, device=device, dtype=torch.float32)
        X_rot_dynamic = rotate_activation(X, H_dynamic, dynamic_group_size)
        Y_c = X_rot_dynamic @ dequant_w_c.T + bias
    else:
        Y_c = X @ dequant_w_c.T + bias

    if "bias_correction" in extra_c:
        Y_c += extra_c["bias_correction"].to(device=device)
    metrics_c = compute_detailed_errors(Y_ref, Y_c)

    # -------------------------------------------------------------------------
    # System D: Unified Dynamic ConvRot + SVD-guided AdaRound + Bias Calibration
    # -------------------------------------------------------------------------
    print(f"  Running System D: Unified Dynamic ConvRot (group_size={dynamic_group_size}) + AdaRound + Bias Calibration...")
    conv_pipeline = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        dynamic_convrot=True,
        convrot_group_size=256, # min group size
        num_iter=2000,           # Fast but robust optimization
        optimizer="prodigy",
        lr_schedule="adaptive",
        lr_factor=0.965,
        lr_cooldown=1,
        lr=1.0,
        device=device
    )
    qdata_d, scale_d, dequant_w_d, extra_d = conv_pipeline.convert(W_orig, calibration_data=X)

    if dynamic_applied:
        Y_d = X_rot_dynamic @ dequant_w_d.T + bias
    else:
        Y_d = X @ dequant_w_d.T + bias

    if "bias_correction" in extra_d:
        Y_d += extra_d["bias_correction"].to(device=device)
    metrics_d = compute_detailed_errors(Y_ref, Y_d)

    # -------------------------------------------------------------------------
    # Print Comparative Results Table
    # -------------------------------------------------------------------------
    print("\n" + f"COMPARISON REPORT FOR {name}:")
    print("-" * 140)
    print(f"{'Metric':<25} | {'System A (Standard Row)':<23} | {f'System B (Static Rot-{static_group_size})':<25} | {f'System C (Dynamic Rot-{dynamic_group_size})':<25} | {f'System D (Dynamic + AdaRound)':<28}")
    print("-" * 140)

    metric_labels = {
        "MAE": "MAE (↓)",
        "MSE": "MSE (v)",  # Wait, let's use ↓ for lower is better
        "RMSE": "RMSE (↓)",
        "SNR (dB)": "SNR (dB) (↑)",
        "Cosine Similarity": "Cosine Similarity (↑)"
    }
    # Let's fix MSE label as well
    metric_labels["MSE"] = "MSE (↓)"

    for metric in ["MAE", "MSE", "RMSE", "SNR (dB)", "Cosine Similarity"]:
        label = metric_labels[metric]
        v1 = metrics_a[metric]
        v2 = metrics_b[metric]
        v3 = metrics_c[metric]
        v4 = metrics_d[metric]
        print(f"{label:<25} | {v1:.6e}             | {v2:.6e}             | {v3:.6e}             | {v4:.6e}")
    print("-" * 140)

    # Mathematical Proof - Rotation Invariance Assertions
    if dynamic_applied:
        W_rot_dynamic = rotate_weight(W_orig, H_dynamic, dynamic_group_size)
        rot_err = torch.norm((X_rot_dynamic @ W_rot_dynamic.T) - (X @ W_orig.T)).item()
        print(f"Mathematical Proof - Orthogonal Invariance Error: {rot_err:.6e} (must be near 0)")
        assert rot_err < 1e-2, "Orthogonal Invariance is broken!"

    # System C (Dynamic) should perform strictly better than or equal to standard/static rounding
    if dynamic_applied:
        # If we successfully resolved a larger group size than 256 (e.g. 1024 or 4096), we expect even better or equal SNR
        if dynamic_group_size > static_group_size:
            print(f"Dynamic group size {dynamic_group_size} is LARGER than static {static_group_size}. Verifying metric improvements...")
            assert metrics_c["SNR (dB)"] >= metrics_a["SNR (dB)"], "Dynamic ConvRot failed to match or beat standard baseline SNR!"
    else:
        print("Dynamic ConvRot safely skipped because features are not divisible by any power of 4 >= 256.")
        assert metrics_c["MSE"] == metrics_a["MSE"], "Dynamic ConvRot was supposed to skip and match baseline exactly!"

    print("\n")
    return metrics_a, metrics_b, metrics_c, metrics_d

def evaluate_attention_special_case(out_features=512, in_features=2048, num_samples=128):
    print("=" * 80)
    print(f"SPECIAL CASE: Attention Block Group-Size Comparison (Shape: [{out_features}, {in_features}])")
    print("=" * 80)

    torch.manual_seed(1337)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Synthesize weights with structured outliers to simulate sensitive diffusion channels
    W_orig = torch.randn(out_features, in_features, device=device) * 0.1
    # Add strong outlier channels to simulate real-world outlier clustering
    W_orig[:, 5] *= 50.0
    W_orig[:, 12] *= 50.0

    X = torch.randn(num_samples, in_features, device=device)
    bias = torch.randn(out_features, device=device)

    # Reference high-precision output
    Y_ref = X @ W_orig.T + bias

    # 1. System A: Standard Symmetric Row-wise Quantization (No ConvRot, No AdaRound)
    print("  Running System A: Standard Row-wise Quantization...")
    conv_standard = LearnedRoundingConverter(
        target_format="int8", scaling_mode="row", no_learned_rounding=True, device=device
    )
    qdata_a, scale_a, dequant_w_a, extra_a = conv_standard.convert(W_orig, calibration_data=X)
    Y_a = X @ dequant_w_a.T + bias
    metrics_a = compute_detailed_errors(Y_ref, Y_a)

    # 2. System B: Static ConvRot (group_size=256) + Simple Rounding
    print("  Running System B: Static ConvRot (group_size=256) + Simple Rounding...")
    conv_256 = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        no_learned_rounding=True,
        device=device
    )
    qdata_256, scale_256, dequant_w_256, extra_256 = conv_256.convert(W_orig, calibration_data=X)
    H_256 = build_hadamard(256, device=device, dtype=torch.float32)
    X_rot_256 = rotate_activation(X, H_256, 256)
    Y_256 = X_rot_256 @ dequant_w_256.T + bias
    if "bias_correction" in extra_256:
        Y_256 += extra_256["bias_correction"].to(device=device)
    metrics_256 = compute_detailed_errors(Y_ref, Y_256)

    # 3. System C: Static ConvRot (group_size=512) + Simple Rounding
    print("  Running System C: Static ConvRot (group_size=512) + Simple Rounding...")
    conv_512 = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=512,
        no_learned_rounding=True,
        device=device
    )
    qdata_512, scale_512, dequant_w_512, extra_512 = conv_512.convert(W_orig, calibration_data=X)
    H_512 = build_hadamard(512, device=device, dtype=torch.float32)
    X_rot_512 = rotate_activation(X, H_512, 512)
    Y_512 = X_rot_512 @ dequant_w_512.T + bias
    if "bias_correction" in extra_512:
        Y_512 += extra_512["bias_correction"].to(device=device)
    metrics_512 = compute_detailed_errors(Y_ref, Y_512)

    # 4. System D: Static ConvRot (group_size=1024) + Simple Rounding
    print("  Running System D: Static ConvRot (group_size=1024) + Simple Rounding...")
    conv_1024 = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=1024,
        no_learned_rounding=True,
        device=device
    )
    qdata_1024, scale_1024, dequant_w_1024, extra_1024 = conv_1024.convert(W_orig, calibration_data=X)
    H_1024 = build_hadamard(1024, device=device, dtype=torch.float32)
    X_rot_1024 = rotate_activation(X, H_1024, 1024)
    Y_1024 = X_rot_1024 @ dequant_w_1024.T + bias
    if "bias_correction" in extra_1024:
        Y_1024 += extra_1024["bias_correction"].to(device=device)
    metrics_1024 = compute_detailed_errors(Y_ref, Y_1024)

    # 5. System E: Static ConvRot (group_size=256) + AdaRound + Bias Calibration
    print("  Running System E: Static ConvRot (group_size=256) + AdaRound + Bias Calibration...")
    conv_pipeline = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=256,
        num_iter=2000,
        optimizer="prodigy",
        lr_schedule="adaptive",
        lr_factor=0.965,
        lr_cooldown=1,
        lr=1.0,
        device=device
    )
    qdata_ada, scale_ada, dequant_w_ada, extra_ada = conv_pipeline.convert(W_orig, calibration_data=X)
    Y_ada = X_rot_256 @ dequant_w_ada.T + bias
    if "bias_correction" in extra_ada:
        Y_ada += extra_ada["bias_correction"].to(device=device)
    metrics_ada = compute_detailed_errors(Y_ref, Y_ada)

    # Print Comparative Results Table
    print("\n" + "COMPARISON REPORT FOR Standard Attention Projection Block:")
    print("-" * 170)
    print(f"{'Metric':<25} | {'System A (Standard Row)':<23} | {'System B (Rot-256)':<18} | {'System C (Rot-512)':<18} | {'System D (Rot-1024)':<19} | {'System E (Rot-256 + AdaRound)':<28}")
    print("-" * 170)

    metric_labels = {
        "MAE": "MAE (↓)",
        "MSE": "MSE (↓)",
        "RMSE": "RMSE (↓)",
        "SNR (dB)": "SNR (dB) (↑)",
        "Cosine Similarity": "Cosine Similarity (↑)"
    }

    for metric in ["MAE", "MSE", "RMSE", "SNR (dB)", "Cosine Similarity"]:
        label = metric_labels[metric]
        v1 = metrics_a[metric]
        v2 = metrics_256[metric]
        v3 = metrics_512[metric]
        v4 = metrics_1024[metric]
        v5 = metrics_ada[metric]
        print(f"{label:<25} | {v1:.6e}             | {v2:.6e}       | {v3:.6e}       | {v4:.6e}        | {v5:.6e}")
    print("-" * 170)

    # Verification assertions
    assert metrics_ada["SNR (dB)"] > metrics_1024["SNR (dB)"], "AdaRound on group size 256 failed to beat simple rounded larger group sizes!"
    assert metrics_1024["SNR (dB)"] >= metrics_512["SNR (dB)"], "Larger group size 1024 failed to beat 512!"
    assert metrics_512["SNR (dB)"] >= metrics_256["SNR (dB)"], "Larger group size 512 failed to beat 256!"

    print("\n")
    return metrics_a, metrics_256, metrics_512, metrics_1024, metrics_ada

def run_production_comparison():
    print("=" * 80)
    print("PRODUCTION QUANTIZATION ERROR COMPARISON & VALIDATION SUITE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Executing mathematical audits on device: {device}\n")

    # Evaluate multiple DiT configurations
    # Case 1: 4096 (standard projection layer, power of 4, supports max 4096)
    evaluate_case("Flux/SD3 Projection Block", out_features=1024, in_features=4096)

    # Case 2: 3072 (non-power-of-4 but divisible by 1024, supports max 1024)
    evaluate_case("Hunyuan / DiT MLP Expansion", out_features=1024, in_features=3072)

    # Case 3: 1152 (non-divisible by any power of 4 >= 256, safely skips)
    evaluate_case("Low-rank or Odd Projection Block", out_features=512, in_features=1152)

    # Case 4: 2048 Special Attention Block Group Size comparison
    evaluate_attention_special_case(out_features=512, in_features=2048)

    print("=" * 80)
    print("All production validation test cases completed successfully with zero regressions!")
    print("=" * 80)

if __name__ == "__main__":
    run_production_comparison()
