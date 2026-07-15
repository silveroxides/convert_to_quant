"""Exact seeded baselines for existing simple and learned quantization paths."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from convert_to_quant.converters.learned_rounding import LearnedRoundingConverter
from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from tests.characterization import safetensors_fingerprint, tensor_map_fingerprint

SIMPLE_CASES = {
    "fp8_tensor": {
        "args": {"int8": False, "scaling_mode": "tensor", "block_size": 8},
        "format": "float8_e4m3fn",
        "weight": "ceb1738289101c03a0159640e310835d9fc28491558118f4a3ae8eae28b382b3",
        "scale": ([], "c02b7678c649e84afdc0132a2aadecbdc3026aa1974e3c5aa57711cfd0d38d69"),
        "config": ([27], "dc6ecd1e9459beb7956b08ac094a9d0cd2e5fd54678085b6a8f43e2e96b6b28a"),
    },
    "fp8_row": {
        "args": {"int8": False, "scaling_mode": "row", "block_size": 8},
        "format": "float8_e4m3fn_rowwise",
        "weight": "8883ed367f660359db6d011596cbdbbd628765ce1f647c784e36d4a10ecc83b7",
        "scale": ([16], "8f39abfb6d4ad6ed3de22b317e25f2b4ea1deaca721850fd127c0b593c21be87"),
        "config": ([35], "0899f9495e1a1c262a2bf29bd3f125f8c870dae971f6b093805237c2117382e3"),
    },
    "fp8_block": {
        "args": {"int8": False, "scaling_mode": "block", "block_size": 8},
        "format": "float8_e4m3fn_blockwise",
        "weight": "eaa9c1ee09034c57cccce201d62138df5208df90abfdd9e7741e8032a2841041",
        "scale": ([2, 2], "ee4c1cd2cbc7d12d082069abc5ce446bf89b2716df4bd74fb1235c3236d16ed3"),
        "config": ([54], "740ea3b81c494ed1f8d3e65bf8a8b6be325b613c9a0a6d2e7958aa25b26b33d7"),
        "group_size": 8,
    },
    "int8_tensor": {
        "args": {"int8": True, "scaling_mode": "tensor", "block_size": 8},
        "format": "int8_tensorwise",
        "weight": "124c2fbf088c85faa99e92aa00d3e04c1a6ea24d5b4a42e79b7db954a773ea74",
        "scale": ([], "bcae4d63b169cb872e14129217a47716f7d0c5b68ac3b3a400af4a58674e427a"),
        "config": ([29], "7b072c49a7edfdc4a3929e8cea9c040cf7ffabb95c0fa05f12cb00ae30a42bf8"),
    },
    "int8_row": {
        "args": {"int8": True, "scaling_mode": "row", "block_size": 8},
        "format": "int8_tensorwise",
        "weight": "cdf13fbbc64574b848c7f13f8ff1ad9ca54aaac528f9364cafa6b813d541dbbe",
        "scale": ([16, 1], "c9ca1fe07c76c3c4510766373f6d4600607efc78963d211984a4351f57984813"),
        "config": ([46], "b727e26694008f0fa0d7c952cacbfe662a6e0ef7d104ac25e5c5d377a5bf72cd"),
    },
    "int8_block": {
        "args": {"int8": True, "scaling_mode": "block", "block_size": 8},
        "format": "int8_blockwise",
        "weight": "9b6ec59901f460b111564271bd2724e02f8b946085f0b2a1d67b438976330563",
        "scale": ([2, 2], "c7dff1a511002c801359a4ab0f6f95bfc54552176bea6b7f22ced58e6ad3f681"),
        "config": ([45], "efd4926adf5f7f0953c3e15ddf486e2bed1673984dff22e0af947b4a08ced83d"),
        "group_size": 8,
        "input_scale": "e00e5eb9444182f352323374ef4e08ebcb784725fdd4fd612d7730540b3e0c8c",
    },
    "int8_convrot": {
        "args": {
            "int8": True,
            "scaling_mode": "row",
            "block_size": 8,
            "convrot": True,
            "convrot_group_size": 16,
        },
        "format": "int8_tensorwise",
        "weight": "9ee30d6e3726a950b5550afc789dc9ad85209216ca3cc5e503cc6d7ab0c77ab3",
        "scale": ([16, 1], "b218629b3dade5fa2d3337ac2081e1c79a19cc02ce13857edff1b62cf2182477"),
        "config": ([88], "30f98dad6ac8ac40ce33ff2fb171eb3068bd04495abf422db068df367c92c2e9"),
        "convrot": True,
        "convrot_groupsize": 16,
    },
}


@pytest.mark.integration
@pytest.mark.parametrize("case_name", SIMPLE_CASES)
def test_simple_conversion_exact_seeded_snapshot(case_name):
    case = SIMPLE_CASES[case_name]
    input_path = Path(f"test_characterization_{case_name}_input.safetensors")
    output_path = Path(f"test_characterization_{case_name}_output.safetensors")
    try:
        generator = torch.Generator(device="cpu").manual_seed(1234)
        save_file({"layer.weight": torch.randn(16, 16, generator=generator)}, input_path)
        convert_to_fp8_scaled(
            str(input_path),
            str(output_path),
            comfy_quant=True,
            filter_flags={},
            calib_samples=4,
            seed=42,
            no_learned_rounding=True,
            save_quant_metadata=True,
            device="cpu",
            **case["args"],
        )
        actual = safetensors_fingerprint(output_path)
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    tensors = actual["tensors"]
    assert tensors["layer.weight"]["sha256"] == case["weight"]
    assert (tensors["layer.weight_scale"]["shape"], tensors["layer.weight_scale"]["sha256"]) == case["scale"]
    assert (tensors["layer.comfy_quant"]["shape"], tensors["layer.comfy_quant"]["sha256"]) == case["config"]
    if "input_scale" in case:
        assert tensors["layer.input_scale"]["sha256"] == case["input_scale"]

    layer_metadata = actual["metadata"]["_quantization_metadata"]["layers"]["layer"]
    expected_metadata = {
        key: case[key]
        for key in ("format", "group_size", "convrot", "convrot_groupsize")
        if key in case
    }
    assert layer_metadata == expected_metadata


LEARNED_HASHES = {
    "original": (
        "78519edd3c71ba49939759e85b62e3f43bf439e32aa11c51f929da22510a597c",
        "7fc5e85db4b089d294de2c062b65bd89ab3281550d972d669ecaf9487fd1bf6d",
    ),
    "adamw": (
        "2823119212e9b81b7e5a9f047cd400d47bffd369b63e65b31d6e0e28536fa2fb",
        "eae6e5c9ea2795abf852ccece9ff005a304826178289ff7bc31784e07e27b39a",
    ),
    "radam": (
        "56c891f8eaf67daa5cacea7209022bb3fac6db0fb7a18cbdf7c16a258fe2103f",
        "a1c206a6d933736db3597f223a5bad6c559552c95f0b83128e0ac945aa40c515",
    ),
    "prodigy": (
        "56c891f8eaf67daa5cacea7209022bb3fac6db0fb7a18cbdf7c16a258fe2103f",
        "a1c206a6d933736db3597f223a5bad6c559552c95f0b83128e0ac945aa40c515",
    ),
}


@pytest.mark.slow
@pytest.mark.parametrize("optimizer", ("original", "adamw", "radam", "prodigy"))
@pytest.mark.parametrize("schedule", ("adaptive", "exponential", "plateau"))
def test_learned_optimizer_and_scheduler_exact_seeded_snapshot(optimizer, schedule):
    torch.manual_seed(2468)
    weight = torch.randn(8, 8, dtype=torch.float32)
    torch.manual_seed(9876)
    converter = LearnedRoundingConverter(
        target_format="fp8",
        scaling_mode="tensor",
        optimizer=optimizer,
        lr_schedule=schedule,
        num_iter=2,
        lr=0.1,
        top_p=0.5,
        min_k=2,
        max_k=4,
        device="cpu",
        early_stop_loss=-1.0,
        early_stop_lr=-1.0,
        early_stop_stall=999,
    )

    qdata, scale, dequantized, extra = converter.convert(weight, key="layer.weight", has_bias=False)
    actual = tensor_map_fingerprint({"q": qdata, "scale": scale, "dequant": dequantized, **extra})
    expected_q, expected_dequant = LEARNED_HASHES[optimizer]

    assert actual["q"]["sha256"] == expected_q
    assert actual["dequant"]["sha256"] == expected_dequant
    assert actual["scale"] == {
        "dtype": "torch.float32",
        "shape": [],
        "sha256": "833c1475d56dc2fcba15cc35b17a20c7dbaa1e9e4682828d5b8751c9b58a23fd",
    }
