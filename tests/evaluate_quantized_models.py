"""
Comprehensive Quantization Error Evaluation Tool

Measures quantization error (MSE, SNR, Cosine Similarity) between a ground-truth
unquantized base model and one or more pre-quantized models.

Uses UnifiedSafetensorsLoader in low-memory mode to stream tensors layer-by-layer,
preventing OOM even with large models.

Usage:
    python tests/evaluate_quantized_models.py \
        --base path/to/base.safetensors \
        --quantized path/to/quant1.safetensors path/to/quant2.safetensors \
        --output evaluation_report.txt \
        --device cuda
"""

import argparse
import gc
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert_to_quant.comfy.quant_ops import LAYOUTS, BlockWiseINT8Layout, TensorWiseINT8Layout
from convert_to_quant.utils.logging import setup_logging, info, error, warning, verbose
from convert_to_quant.utils.memory_efficient_loader import UnifiedSafetensorsLoader
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def compute_metrics(ref: torch.Tensor, quant: torch.Tensor) -> Dict[str, float]:
    """Compute error metrics between reference and quantized tensors."""
    # Ensure they are on the same device and float32 for computation
    ref = ref.to(torch.float32)
    quant = quant.to(torch.float32)

    diff = ref - quant
    mse = torch.mean(diff**2).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    mae = torch.mean(torch.abs(diff)).item()

    ref_norm = torch.norm(ref).item()
    err_norm = torch.norm(diff).item()

    # Signal-to-Noise Ratio (dB)
    if err_norm < 1e-12:
        snr_db = float("inf")
    elif ref_norm < 1e-12:
        snr_db = 0.0
    else:
        snr_db = 20 * torch.log10(torch.tensor(ref_norm / err_norm)).item()

    # Cosine Similarity
    if ref_norm > 1e-10 and torch.norm(quant).item() > 1e-10:
        cos_sim = torch.nn.functional.cosine_similarity(ref.flatten(), quant.flatten(), dim=0).item()
    else:
        cos_sim = 1.0 if (ref_norm < 1e-10 and torch.norm(quant).item() < 1e-10) else 0.0

    return {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "SNR": snr_db,
        "CosSim": cos_sim,
        "Elements": ref.numel()
    }


class ModelEvaluator:
    def __init__(self, base_path: str, device: str = "cpu"):
        self.base_path = base_path
        self.device = device
        self.base_loader = UnifiedSafetensorsLoader(base_path, low_memory=True)
        # Identify weight keys (skipping metadata, scales, biases)
        self.weight_keys = [
            k for k in self.base_loader.keys()
            if not k.endswith((".comfy_quant", "_scale", ".weight_scale", ".bias", "bias"))
        ]
        info(f"Base model loaded: {len(self.weight_keys)} layers identified.")

    def dequantize_layer(self, loader: UnifiedSafetensorsLoader, key: str) -> Optional[torch.Tensor]:
        """Load and dequantize a layer from the quantized model loader."""
        try:
            qdata = loader.get_tensor(key).to(self.device)

            # 1. Check for .comfy_quant metadata tensor
            comfy_key = f"{key}.comfy_quant"
            if comfy_key in loader.keys():
                config = tensor_to_dict(loader.get_tensor(comfy_key))
                fmt = config.get("format")
                layout_name = config.get("comfy_tensor_layout")

                # Retrieve scale(s)
                scale = None
                for scale_name in [f"{key}.weight_scale", f"{key}_scale", f"{key}.scale"]:
                    if scale_name in loader.keys():
                        scale = loader.get_tensor(scale_name).to(self.device)
                        break

                if layout_name in LAYOUTS:
                    layout = LAYOUTS[layout_name]
                    # Prepare dequant args
                    dq_args = {
                        "qdata": qdata,
                        "scale": scale,
                        "orig_dtype": torch.float32 # Dequant to float32 for comparison
                    }
                    if "block_size" in config:
                        dq_args["block_size"] = config["block_size"]
                    if "group_size" in config:
                        dq_args["block_size"] = config["group_size"] # alias

                    # Handle INT8 specific flags
                    if layout_name == "BlockWiseINT8Layout":
                        dq_args["is_weight"] = True
                    elif layout_name == "TensorWiseINT8Layout":
                        dq_args["is_weight"] = True

                    return layout.dequantize(**dq_args)
                else:
                    warning(f"  - Unknown layout '{layout_name}' for {key}, attempting simple scaling.")
                    if scale is not None:
                        return qdata.to(torch.float32) * scale.to(torch.float32)
                    return qdata.to(torch.float32)

            # 2. Heuristic fallback for non-ComfyQuant metadata
            # Check for scale anyway
            scale = None
            for scale_name in [f"{key}.weight_scale", f"{key}_scale", f"{key}.scale"]:
                if scale_name in loader.keys():
                    scale = loader.get_tensor(scale_name).to(self.device)
                    break

            if scale is not None:
                # Try simple broadcasting
                try:
                    return qdata.to(torch.float32) * scale.to(torch.float32)
                except Exception:
                    # Try row-wise
                    if scale.dim() == 1 and scale.shape[0] == qdata.shape[0]:
                        return qdata.to(torch.float32) * scale.unsqueeze(1).to(torch.float32)

            return qdata.to(torch.float32)
        except Exception as e:
            error(f"  - Failed to dequantize {key}: {e}")
            return None

    def evaluate_model(self, quant_path: str, layer_filter: Optional[str] = None) -> Dict[str, Any]:
        """Evaluate a single quantized model against the base model."""
        info(f"\nEvaluating model: {quant_path}")

        results = []
        with UnifiedSafetensorsLoader(quant_path, low_memory=True) as quant_loader:
            # Filter keys
            target_keys = self.weight_keys
            if layer_filter:
                regex = re.compile(layer_filter)
                target_keys = [k for k in target_keys if regex.search(k)]

            pbar = tqdm(target_keys, desc="  Layers", leave=False)
            for key in pbar:
                if key not in quant_loader.keys():
                    verbose(f"  - Skipping {key}: not in quantized model")
                    continue

                base_w = self.base_loader.get_tensor(key).to(self.device)
                quant_w = self.dequantize_layer(quant_loader, key)

                if quant_w is not None:
                    # Check shape alignment
                    if base_w.shape != quant_w.shape:
                        warning(f"  - Shape mismatch for {key}: Base {list(base_w.shape)} != Quant {list(quant_w.shape)}")
                    else:
                        metrics = compute_metrics(base_w, quant_w)
                        metrics["Name"] = key
                        results.append(metrics)

                # Cleanup per layer
                del base_w, quant_w
                self.base_loader.mark_processed(key)
                quant_loader.mark_processed(key)
                if self.device == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

        if not results:
            return None

        # Aggregate results
        total_elements = sum(r["Elements"] for r in results)
        avg_mse = sum(r["MSE"] * r["Elements"] for r in results) / total_elements
        avg_mae = sum(r["MAE"] * r["Elements"] for r in results) / total_elements
        avg_cos = sum(r["CosSim"] * r["Elements"] for r in results) / total_elements

        # Aggregate SNR requires weighted log-sum? Actually, simpler to just use
        # Total MSE and Total Variance if we had it. Let's just do simple average SNR.
        avg_snr = sum(r["SNR"] for r in results) / len(results)

        return {
            "Path": quant_path,
            "Layers": len(results),
            "MSE": avg_mse,
            "MAE": avg_mae,
            "SNR": avg_snr,
            "CosSim": avg_cos,
            "PerLayer": results
        }

    def close(self):
        self.base_loader.close()


def generate_report(all_results: List[Dict], output_file: str):
    """Write detailed comparison report to a text file."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 100 + "\n")
        f.write(f"QUANTIZATION ERROR EVALUATION REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 100 + "\n\n")

        # Summary Table
        f.write("SUMMARY COMPARISON\n")
        f.write("-" * 100 + "\n")
        header = f"{'Model Path':<40} | {'Layers':<6} | {'MSE':<10} | {'SNR (dB)':<8} | {'CosSim':<8}\n"
        f.write(header)
        f.write("-" * 100 + "\n")

        for res in all_results:
            name = Path(res["Path"]).name
            if len(name) > 37: name = "..." + name[-34:]
            line = f"{name:<40} | {res['Layers']:<6} | {res['MSE']:<10.3e} | {res['SNR']:<8.2f} | {res['CosSim']:<8.6f}\n"
            f.write(line)
        f.write("-" * 100 + "\n\n")

        # Detailed Breakdown per model
        for res in all_results:
            f.write(f"\nDETAILED BREAKDOWN: {res['Path']}\n")
            f.write("-" * 100 + "\n")
            f.write(f"{'Layer Name':<60} | {'MSE':<10} | {'SNR':<8} | {'CosSim':<8}\n")
            f.write("-" * 100 + "\n")
            # Sort by worst MSE
            sorted_layers = sorted(res["PerLayer"], key=lambda x: x["MSE"], reverse=True)
            for l in sorted_layers:
                lname = l["Name"]
                if len(lname) > 57: lname = "..." + lname[-54:]
                f.write(f"{lname:<60} | {l['MSE']:<10.3e} | {l['SNR']:<8.2f} | {l['CosSim']:<8.6f}\n")
            f.write("-" * 100 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Measure quantization error across multiple models.")
    parser.add_argument("--base", type=str, required=True, help="Path to base (unquantized) model")
    parser.add_argument("--quantized", type=str, nargs="+", required=True, help="Path(s) to quantized model(s)")
    parser.add_argument("--output", type=str, default="quantization_report.txt", help="Output report file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--filter", type=str, default=None, help="Regex to filter layers (e.g. 'attention|mlp')")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()
    setup_logging("VERBOSE" if args.verbose else "INFO")

    evaluator = ModelEvaluator(args.base, device=args.device)

    all_model_results = []
    try:
        for q_path in args.quantized:
            res = evaluator.evaluate_model(q_path, layer_filter=args.filter)
            if res:
                all_model_results.append(res)
                # Print summary for this model
                info(f"  -> Aggregate MSE: {res['MSE']:.4e} | SNR: {res['SNR']:.2f} dB | CosSim: {res['CosSim']:.6f}")
            else:
                error(f"  -> Evaluation failed for {q_path}")
    finally:
        evaluator.close()

    if all_model_results:
        generate_report(all_model_results, args.output)
        info(f"\nFinal comparative report written to: {args.output}")
    else:
        error("No models were successfully evaluated.")

if __name__ == "__main__":
    main()
