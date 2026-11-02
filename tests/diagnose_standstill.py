import torch
import torch.nn as nn
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.utils.logging import setup_logging
import sys

def diagnose():
    setup_logging("VERBOSE")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    M, N = 512, 512
    group_size = 256
    W = torch.randn(M, N, device=device)

    # Initialize converter
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=group_size,
        optimizer="prodigy",
        lr_schedule="plateau",
        num_iter=100, # Increased to see if it starts optimizing after warm-up
        lr=1.0,
        scale_optimization="dualround",
        device=device
    )

    print("\n--- Starting Diagnosis with fix (Plateau + Prodigy) ---")

    # Mocking what the converter does internally
    qdata, scale, dequantized, extra = converter.convert(W)

    print("\n--- Diagnosis Complete ---")

if __name__ == "__main__":
    diagnose()
