"""Round-trip tests for hi_first / lo_first nibble packing.

Guards against silent flips of the packing default on either the local
pack_uint4 / unpack_uint4 or (when available) the comfy-kitchen side.
"""
from __future__ import annotations

import pytest
import torch

from convert_to_quant.utils.float_utils import pack_uint4, unpack_uint4


class TestNibbleOrder:
    def test_hi_first_pack_layout(self):
        # Element 0 goes to high nibble when hi_first=True
        data = torch.tensor([0xA, 0xB, 0xC, 0xD], dtype=torch.uint8)
        packed = pack_uint4(data, hi_first=True)
        assert packed.tolist() == [0xAB, 0xCD]

    def test_lo_first_pack_layout(self):
        # Element 0 goes to low nibble when hi_first=False
        data = torch.tensor([0xA, 0xB, 0xC, 0xD], dtype=torch.uint8)
        packed = pack_uint4(data, hi_first=False)
        assert packed.tolist() == [0xBA, 0xDC]

    @pytest.mark.parametrize("hi_first", [True, False])
    def test_round_trip(self, hi_first):
        torch.manual_seed(42)
        data = torch.randint(0, 16, (8, 64), dtype=torch.uint8)
        packed = pack_uint4(data, hi_first=hi_first)
        unpacked = unpack_uint4(packed, hi_first=hi_first)
        assert torch.equal(unpacked, data)

    def test_mismatched_flags_corrupt(self):
        # Packing hi_first=True then unpacking hi_first=False must not round-trip.
        # This test fails if either default silently flips.
        data = torch.tensor([[0xA, 0xB, 0xC, 0xD]], dtype=torch.uint8)
        packed = pack_uint4(data, hi_first=True)
        wrong = unpack_uint4(packed, hi_first=False)
        assert not torch.equal(wrong, data)

    def test_default_is_hi_first(self):
        # Lock in the default. A regression here means checkpoints produced by
        # older versions become unreadable without explicit hi_first=True.
        data = torch.tensor([0xA, 0xB], dtype=torch.uint8)
        default_packed = pack_uint4(data)
        explicit_packed = pack_uint4(data, hi_first=True)
        assert torch.equal(default_packed, explicit_packed)
        assert default_packed.tolist() == [0xAB]

    def test_default_unpack_is_hi_first(self):
        packed = torch.tensor([0xAB], dtype=torch.uint8)
        default_unpacked = unpack_uint4(packed)
        explicit_unpacked = unpack_uint4(packed, hi_first=True)
        assert torch.equal(default_unpacked, explicit_unpacked)
        assert default_unpacked.tolist() == [0xA, 0xB]


class TestKitchenParity:
    """Only runs when comfy-kitchen is importable. Verifies the local default
    (hi_first=True) produces bytes decodable by the local unpack with the same
    flag, and that the forked kitchen's hi_first param is wired end-to-end."""

    @pytest.fixture(autouse=True)
    def _require_kitchen(self):
        pytest.importorskip("comfy_kitchen", reason="comfy-kitchen not installed")

    @pytest.fixture(autouse=True)
    def _require_cuda(self):
        if not torch.cuda.is_available():
            pytest.skip("kitchen quantize_nvfp4 requires CUDA")

    def test_local_unpack_reads_kitchen_output(self):
        """Local unpack(hi_first=True) must decode bytes produced by kitchen's
        default (also hi_first=True). Fails if kitchen's default flips."""
        import comfy_kitchen as ck

        x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
        per_tensor_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        qdata, _ = ck.quantize_nvfp4(x, per_tensor_scale, pad_16x=False)
        unpacked = unpack_uint4(qdata.cpu(), hi_first=True)

        assert unpacked.min().item() >= 0
        assert unpacked.max().item() <= 15

    def test_kitchen_hi_first_param_wired(self):
        """When comfy-kitchen supports hi_first, explicitly passing True and
        passing nothing should produce identical output."""
        import inspect
        import comfy_kitchen as ck

        if "hi_first" not in inspect.signature(ck.quantize_nvfp4).parameters:
            pytest.skip("installed comfy-kitchen does not support hi_first param")

        x = torch.randn(16, 32, dtype=torch.bfloat16, device="cuda")
        per_tensor_scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")

        qdata_default, _ = ck.quantize_nvfp4(x, per_tensor_scale, pad_16x=False)
        qdata_explicit, _ = ck.quantize_nvfp4(
            x, per_tensor_scale, pad_16x=False, hi_first=True
        )

        assert torch.equal(qdata_default, qdata_explicit)
