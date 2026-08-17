import torch

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter


def test_oom_recovery_and_calib_scale_shrinkage(monkeypatch):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create converter
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        convrot=True,
        convrot_group_size=64,
        num_iter=10,
        device=device,
    )

    # We want to mock _convert_int8_tensorwise so that the first call raises OutOfMemoryError,
    # and the second call (after calib_scale shrinks to 0.5) succeeds.
    original_convert_tensorwise = converter._convert_int8_tensorwise
    call_count = 0

    def mock_convert_tensorwise(W_float32, calibration_data=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Raise synthetic OOM on first try
            raise torch.cuda.OutOfMemoryError("Synthetic CUDA out of memory.")
        else:
            # Succeed on subsequent tries
            return original_convert_tensorwise(W_float32, calibration_data=calibration_data)

    monkeypatch.setattr(converter, "_convert_int8_tensorwise", mock_convert_tensorwise)

    # Set up some dummy weight and calibration data
    W_orig = torch.randn(64, 128, device=device, dtype=torch.float32)
    X = torch.randn(32, 128, device=device, dtype=torch.float32)

    # We expect that the main convert() call succeeds on the second try
    assert converter.calib_scale == 1.0

    qdata, scale, dequant_w, extra_tensors = converter.convert(
        W_orig,
        key="test_layer.weight",
        depth=0,
        calibration_data=X,
    )

    # Assertions
    assert call_count == 2
    # Verify that during the retry loop, calib_scale was indeed shrunk to 0.5
    # (Though after convert() completes, the finally block resets calib_scale back to 1.0)
    assert converter.calib_scale == 1.0

    # Check output properties
    assert qdata.shape == W_orig.shape
    assert qdata.dtype == torch.int8
    assert scale.shape == (64, 1)


def test_finalize_int8_qdata_clamps_rounds_and_casts_in_place():
    converter = LearnedRoundingConverter(
        target_format="int8",
        scaling_mode="row",
        num_iter=1,
        device="cpu",
    )
    working = torch.tensor([-200.0, -1.6, -0.4, 2.6, 200.0], dtype=torch.float32)

    result = converter._finalize_int8_qdata(working)

    assert result.dtype == torch.int8
    assert result.tolist() == [-127, -2, 0, 3, 127]
