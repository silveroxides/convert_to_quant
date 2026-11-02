import torch
import math
from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter

def test_int8_rowwise():
    M, N = 64, 64
    W_float32 = torch.randn(M, N, dtype=torch.float32)
    w_max = W_float32.abs().max()
    scale = torch.ones(M, 1, dtype=torch.float32) * (w_max / 127.0)
    # mock quant
    qdata = (W_float32 / scale).clamp(-127, 127).round().to(torch.int8)

    U_k = torch.randn(M, 16)
    Vh_k = torch.randn(16, N)

    converter = LearnedRoundingConverter(
        scaling_mode="row",
        target_format="int8",
        optimizer_choice="original",
        num_iter=10,
        lr=1.0,
        lr_schedule="adaptive"
    )

    # Should not crash and should return updated int8 tensor
    q_out = converter._optimize_int8_original(W_float32, qdata, scale, U_k, Vh_k, scaling_mode="row")

    assert q_out.dtype == torch.int8
    assert q_out.shape == (M, N)
    print("Test passed successfully.")

if __name__ == "__main__":
    test_int8_rowwise()