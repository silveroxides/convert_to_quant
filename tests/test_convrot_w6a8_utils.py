import sys
from pathlib import Path

import pytest
import torch

from convert_to_quant.utils.convrot import build_hadamard, dequantize_w6a8_rotated, dequantize_w6a8_to_int8_grid, pack_int6_w6a8, quantize_w6a8_grouped, resolve_w6a8_convrot_group_size, rotate_weight, unpack_int6_w6a8, validate_w6a8_shape


def test_pack_unpack_int6_roundtrip():
    torch.manual_seed(42)
    codes = torch.randint(0, 64, (8, 128), dtype=torch.int32)
    packed = pack_int6_w6a8(codes)
    assert packed.shape == (8, 128 * 3 // 4)
    assert packed.dtype == torch.int8
    unpacked = unpack_int6_w6a8(packed, 128)
    assert torch.equal(unpacked, codes)


def test_pack_int6_byte_plane_layout():
    codes = torch.randint(0, 64, (2, 32), dtype=torch.int32)
    packed = pack_int6_w6a8(codes)
    # First K/2 bytes: low nibbles (even col -> low nibble, odd col -> high nibble)
    expected_low = ((codes[:, 0::2] & 0xF) | ((codes[:, 1::2] & 0xF) << 4)).to(torch.int8)
    assert torch.equal(packed[:, :16], expected_low)
    # Next K/4 bytes: top 2 bits of col c at byte c//4, bits 2*(c%4)
    hi = packed[:, 16:].to(torch.int32) & 0xFF
    for c in range(32):
        assert torch.equal((hi[:, c // 4] >> (2 * (c % 4))) & 0x3, codes[:, c] >> 4)


def test_validate_w6a8_shape_rejections():
    validate_w6a8_shape(256, 16, 256)
    with pytest.raises(ValueError, match="divisible by 32"):
        validate_w6a8_shape(272, 16, 16)
    with pytest.raises(ValueError, match="multiple of 16"):
        validate_w6a8_shape(256, 8, 256)
    with pytest.raises(ValueError, match="power-of-4"):
        validate_w6a8_shape(256, 16, 128)


def test_resolve_w6a8_convrot_group_size():
    assert resolve_w6a8_convrot_group_size(1024, 256, dynamic=False) == 256
    assert resolve_w6a8_convrot_group_size(320, 256, dynamic=False) == 64
    assert resolve_w6a8_convrot_group_size(96, 256, dynamic=False) == 16
    assert resolve_w6a8_convrot_group_size(1024, 256, dynamic=True) == 1024


def test_quantize_w6a8_scale_search_improves_quality():
    torch.manual_seed(7)
    W = torch.randn(64, 256, dtype=torch.float32) * 0.02
    H = build_hadamard(256, device="cpu", dtype=torch.float32)
    W_rot = rotate_weight(W, H, 256)

    c_search, srel_s, sch_s = quantize_w6a8_grouped(W_rot, group_size=16, scale_search=True)
    c_plain, srel_p, sch_p = quantize_w6a8_grouped(W_rot, group_size=16, scale_search=False)

    assert c_search.min().item() >= 1 and c_search.max().item() <= 63
    dq_s = dequantize_w6a8_rotated(pack_int6_w6a8(c_search), srel_s, sch_s, group_size=16)
    dq_p = dequantize_w6a8_rotated(pack_int6_w6a8(c_plain), srel_p, sch_p, group_size=16)

    err_s = ((dq_s - W_rot).norm() / W_rot.norm()).item()
    err_p = ((dq_p - W_rot).norm() / W_rot.norm()).item()
    assert err_s < err_p


@pytest.mark.parametrize("group_size", [16, 32, 64])
@pytest.mark.parametrize("scale_dtype", [torch.float8_e4m3fn, torch.float32])
def test_parity_with_comfy_kitchen_w6a8(group_size, scale_dtype):
    kitchen_path = Path(__file__).resolve().parent.parent / "references" / "comfy-kitchen"
    if not (kitchen_path / "comfy_kitchen").exists():
        pytest.skip("references/comfy-kitchen not found")

    saved_modules = {k: v for k, v in sys.modules.items() if k == "comfy_kitchen" or k.startswith("comfy_kitchen.")}
    for k in saved_modules:
        del sys.modules[k]
    sys.path.insert(0, str(kitchen_path))
    try:
        try:
            from comfy_kitchen.backends.eager import w4a8_int8 as ck_eager
        except ImportError:
            pytest.skip("comfy_kitchen eager w4a8_int8 not importable")

        torch.manual_seed(100)
        W = torch.randn(32, 256, dtype=torch.float32) * 0.05

        ck_q, ck_srel, ck_sch, ck_corr, ck_cb = ck_eager.quantize_w4a8_int8_weight(
            W,
            group_size=group_size,
            convrot_groupsize=256,
            symmetric=True,
            scale_dtype=scale_dtype,
            codebook=False,
            bits=6,
            scale_search=True,
        )
        assert ck_corr is None and ck_cb is None
        ck_int8_grid = ck_eager._dequant_int4_grouped_to_int8(ck_q, ck_srel, None, group_size)
        ck_dq = ck_eager.dequantize_w4a8_int8_weight(
            ck_q, ck_srel, ck_sch, group_size=group_size, convrot_groupsize=256, output_dtype=torch.float32
        )
    finally:
        if str(kitchen_path) in sys.path:
            sys.path.remove(str(kitchen_path))
        for k in [k for k in sys.modules if k == "comfy_kitchen" or k.startswith("comfy_kitchen.")]:
            del sys.modules[k]
        sys.modules.update(saved_modules)

    from convert_to_quant.converters.w6a8_converter import W6A8Converter

    converter = W6A8Converter(
        convrot=True,
        convrot_group_size=256,
        group_size=group_size,
        dynamic_convrot=False,
        scale_dtype=scale_dtype,
        scale_search=True,
        device="cpu",
        no_learned_rounding=True,
    )
    our_q, our_srel, our_sch, our_dq, _ = converter.convert(W)

    assert torch.equal(our_q, ck_q)
    assert torch.equal(
        our_srel.view(torch.uint8) if scale_dtype == torch.float8_e4m3fn else our_srel,
        ck_srel.view(torch.uint8) if scale_dtype == torch.float8_e4m3fn else ck_srel,
    )
    assert torch.allclose(our_sch, ck_sch, atol=1e-6)
    assert torch.equal(dequantize_w6a8_to_int8_grid(our_q, our_srel, group_size=group_size), ck_int8_grid)
    assert torch.allclose(our_dq, ck_dq, atol=1e-5)
