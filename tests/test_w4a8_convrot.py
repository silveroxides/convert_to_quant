"""
Tests for W4A8-ConvRot (asym_w4a8_int8) quantization support.

Covers:
(a) round-trip fidelity — dequantize(quantize(W)) error bounded, rotation norm-preserving
(b) nibble packing order matches the serialization contract (even col = low nibble)
(c) serialization produces exactly the 5 tensors with contract dtypes/shapes
    and the exact comfy_quant JSON
(d) CLI flag parsing (--w4a8, custom-type/fallback choices, EXPERIMENTAL_ARGS)
(e) layer-config format mapping ("w4a8")
"""

import json
import os
import unittest

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from convert_to_quant.constants import VALID_QUANT_FORMATS
from convert_to_quant.converters.w4a8_convrot_converter import (
    W4A8_FIXED_LUT,
    W4A8ConvRotConverter,
    pack_w4a8_codes,
    unpack_w4a8_codes,
)
from convert_to_quant.formats.fp8_conversion import (
    convert_to_fp8_scaled,
    resolve_layer_config_format,
)
from convert_to_quant.utils.convrot import build_hadamard, rotate_weight
from convert_to_quant.utils.tensor_utils import tensor_to_dict


def _make_weight(n: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, k, generator=g)


class TestNibblePacking(unittest.TestCase):
    def test_even_column_is_low_nibble(self):
        codes = torch.tensor([[1, 2, 3, 15, 0, 9]], dtype=torch.uint8)
        packed = pack_w4a8_codes(codes)
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(packed.shape, (1, 3))
        u = packed[0].to(torch.uint8)
        # byte j holds code[2j] in the LOW nibble and code[2j+1] in the HIGH nibble
        self.assertEqual(int(u[0]), (1 & 0xF) | ((2 & 0xF) << 4))
        self.assertEqual(int(u[1]), (3 & 0xF) | ((15 & 0xF) << 4))
        self.assertEqual(int(u[2]), (0 & 0xF) | ((9 & 0xF) << 4))

    def test_round_trip_pack_unpack(self):
        codes = torch.randint(0, 16, (8, 512), dtype=torch.uint8)
        unpacked = unpack_w4a8_codes(pack_w4a8_codes(codes))
        self.assertTrue(torch.equal(codes, unpacked))

    def test_all_values_distinct_nibbles(self):
        codes = torch.arange(16, dtype=torch.uint8).view(1, 16)
        unpacked = unpack_w4a8_codes(pack_w4a8_codes(codes))
        self.assertTrue(torch.equal(codes, unpacked))


class TestRoundTripFidelity(unittest.TestCase):
    def test_rotation_preserves_norms(self):
        W = _make_weight(16, 512, seed=1)
        H = build_hadamard(256)
        W_rot = rotate_weight(W, H, 256)
        torch.testing.assert_close(W.norm(dim=-1), W_rot.norm(dim=-1), rtol=1e-4, atol=1e-4)
        # H is orthogonal: (W @ H^T) @ H == W
        back = rotate_weight(W_rot, H, 256)  # H^T == H for Hadamard, applying again = inverse
        torch.testing.assert_close(back, W, rtol=1e-4, atol=1e-5)

    def test_round_trip_error_bounded(self):
        for k in (256, 1024):
            with self.subTest(k=k):
                W = _make_weight(32, k, seed=2)
                conv = W4A8ConvRotConverter(convrot_group_size=256, device="cpu")
                packed, s_rel, s_channel, codebook, _, _ = conv.convert(W)
                W_hat = conv.dequantize(packed, s_rel, s_channel)
                rel_err = (W_hat - W).norm() / W.norm()
                # 4-bit codebook + INT8 grid: ~5-15% relative error for Gaussian weights
                self.assertLess(rel_err.item(), 0.25, f"round-trip error too high: {rel_err}")

    def test_decoded_weights_on_int8_grid(self):
        """The rotated dequantized weight / s_channel must be integers in [-127, 127]."""
        W = _make_weight(8, 256, seed=3)
        conv = W4A8ConvRotConverter(convrot_group_size=256, device="cpu")
        packed, s_rel, s_channel, _, _, _ = conv.convert(W)
        from convert_to_quant.converters.w4a8_convrot_converter import unpack_w4a8_codes

        codes = unpack_w4a8_codes(packed)
        lut = torch.tensor(W4A8_FIXED_LUT)
        levels = lut[codes.long()].view(8, 256 // 16, 16)
        s_rel_dec = s_rel.to(torch.float32)
        decoded = torch.round(torch.clamp(levels * s_rel_dec.unsqueeze(-1), -127.0, 127.0))
        self.assertTrue(torch.all(decoded.abs() <= 127))
        self.assertTrue(torch.all(decoded == torch.round(decoded)))

    def test_codebook_is_exact_contract_lut(self):
        conv = W4A8ConvRotConverter()
        self.assertEqual(conv.codebook.numel(), 16)
        self.assertEqual(conv.codebook.dtype, torch.float32)
        # Bit-exact: both float32 tensors built from the same contract decimals
        self.assertTrue(torch.equal(conv.codebook, torch.tensor(W4A8_FIXED_LUT, dtype=torch.float32)))
        # First 8 entries negative, last 8 positive (sorted LUT)
        self.assertTrue(torch.all(conv.codebook[:8] < 0))
        self.assertTrue(torch.all(conv.codebook[8:] > 0))


class TestConverterValidation(unittest.TestCase):
    def test_rejects_bad_group_size(self):
        with self.assertRaises(ValueError):
            W4A8ConvRotConverter(group_size=32)

    def test_rejects_k_not_divisible_by_convrot_group(self):
        conv = W4A8ConvRotConverter(device="cpu")
        with self.assertRaises(ValueError):
            conv.convert(torch.randn(4, 384))  # 384 % 256 != 0

    def test_rejects_k_not_divisible_by_16(self):
        conv = W4A8ConvRotConverter(convrot_group_size=64, device="cpu")
        with self.assertRaises(ValueError):
            conv.convert(torch.randn(4, 48))  # 48 % 16 == 0 but 48 % 64 != 0
        with self.assertRaises(ValueError):
            conv.convert(torch.randn(4, 64 + 8))  # 72 not divisible by 16


class TestEndToEndSerialization(unittest.TestCase):
    def _convert(self, tmpdir, layer_shapes, **extra_kwargs):
        input_path = os.path.join(tmpdir, "input.safetensors")
        output_path = os.path.join(tmpdir, "output.safetensors")
        tensors = {}
        for i, (n, k) in enumerate(layer_shapes):
            tensors[f"blocks.{i}.mlp.fc{i}.weight"] = _make_weight(n, k, seed=10 + i)
        save_file(tensors, input_path)

        kwargs = dict(
            comfy_quant=True,
            filter_flags={},
            calib_samples=2,
            seed=1,
            no_learned_rounding=True,
            save_quant_metadata=True,
            low_memory=True,
            device="cpu",
        )
        kwargs.update(extra_kwargs)
        convert_to_fp8_scaled(input_path, output_path, **kwargs)
        return input_path, output_path, tensors

    def test_primary_w4a8_serializes_contract_tensors(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            _, output_path, tensors = self._convert(
                tmpdir,
                [(8, 512), (16, 1024)],
                primary_format="w4a8",
                convrot=True,
                convrot_group_size=256,
            )
            converted = load_file(output_path)
            for key in tensors:
                base = key.removesuffix(".weight")
                n, k = tensors[key].shape
                # weight: INT8 [N, K/2]
                self.assertEqual(converted[f"{base}.weight"].dtype, torch.int8, key)
                self.assertEqual(tuple(converted[f"{base}.weight"].shape), (n, k // 2), key)
                # weight_s_rel: F8_E4M3 [N, K/16]
                self.assertEqual(converted[f"{base}.weight_s_rel"].dtype, torch.float8_e4m3fn, key)
                self.assertEqual(tuple(converted[f"{base}.weight_s_rel"].shape), (n, k // 16), key)
                # weight_s_channel: F32 [N]
                self.assertEqual(converted[f"{base}.weight_s_channel"].dtype, torch.float32, key)
                self.assertEqual(tuple(converted[f"{base}.weight_s_channel"].shape), (n,), key)
                # weight_codebook: F32 [16]
                self.assertEqual(converted[f"{base}.weight_codebook"].dtype, torch.float32, key)
                self.assertEqual(tuple(converted[f"{base}.weight_codebook"].shape), (16,), key)
                # comfy_quant: U8 JSON bytes with the exact contract content
                cq = converted[f"{base}.comfy_quant"]
                self.assertEqual(cq.dtype, torch.uint8, key)
                # Byte-exact contract JSON: exactly 89 bytes, no extra keys
                # (even if global --fpmm / orig_dtype were in effect)
                self.assertEqual(cq.numel(), 89, key)
                self.assertEqual(
                    bytes(cq.tolist()),
                    b'{"format": "asym_w4a8_int8", "group_size": 16, "convrot": true, "convrot_groupsize": 256}',
                    key,
                )
                config = tensor_to_dict(cq)
                self.assertEqual(
                    config,
                    {"format": "asym_w4a8_int8", "group_size": 16, "convrot": True, "convrot_groupsize": 256},
                )
                # Codebook tensor is the frozen LUT verbatim (bit-exact)
                self.assertTrue(
                    torch.equal(converted[f"{base}.weight_codebook"], torch.tensor(W4A8_FIXED_LUT, dtype=torch.float32))
                )
                # No weight_scale emitted for w4a8 layers (contract uses _s_rel/_s_channel)
                self.assertNotIn(f"{base}.weight_scale", converted)
                self.assertNotIn(f"{base}.input_scale", converted)

            # _quantization_metadata records the format + convrot flags
            with safe_open(output_path, framework="pt", device="cpu") as handle:
                meta = json.loads(handle.metadata()["_quantization_metadata"])
            for key in tensors:
                base = key.removesuffix(".weight")
                entry = meta["layers"][base]
                self.assertEqual(entry["format"], "asym_w4a8_int8")
                self.assertEqual(entry["group_size"], 16)
                self.assertTrue(entry["convrot"])
                self.assertEqual(entry["convrot_groupsize"], 256)

    def test_w4a8_output_matches_runtime_decode(self):
        """Full pipeline: dequantize the saved tensors and compare to the original weight."""
        import tempfile

        from convert_to_quant.converters.w4a8_convrot_converter import (
            unpack_w4a8_codes,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            _, output_path, tensors = self._convert(
                tmpdir, [(8, 512)], primary_format="w4a8", convrot_group_size=256
            )
            # (kwargs already pass device="cpu" via _convert)
            converted = load_file(output_path)
            key = next(iter(tensors))
            base = key.removesuffix(".weight")
            W = tensors[key]

            # Runtime decode math (contract §4): rotate back with H, decode on INT8 grid
            packed = converted[f"{base}.weight"]
            s_rel = converted[f"{base}.weight_s_rel"].to(torch.float32)
            s_channel = converted[f"{base}.weight_s_channel"]
            codebook = converted[f"{base}.weight_codebook"]

            codes = unpack_w4a8_codes(packed)
            n, k = W.shape
            levels = codebook[codes.long()].view(n, k // 16, 16)
            grid = torch.round(torch.clamp(levels * s_rel.unsqueeze(-1), -127.0, 127.0))
            w_rot = (grid * s_channel.view(n, 1, 1)).view(n, k)
            H = build_hadamard(256)
            W_hat = torch.matmul(w_rot.view(n, k // 256, 256), H).view(n, k)

            rel_err = (W_hat - W).norm() / W.norm()
            self.assertLess(rel_err.item(), 0.25)

    def test_w4a8_via_layer_config(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")
            save_file({"blocks.0.mlp.fc1.weight": _make_weight(8, 512, seed=5)}, input_path)

            convert_to_fp8_scaled(
                input_path,
                output_path,
                comfy_quant=True,
                filter_flags={},
                calib_samples=2,
                seed=1,
                no_learned_rounding=True,
                save_quant_metadata=True,
                low_memory=True,
                device="cpu",
                layer_config={"_default": {"format": "w4a8"}},
            )
            converted = load_file(output_path)
            base = "blocks.0.mlp.fc1"
            config = tensor_to_dict(converted[f"{base}.comfy_quant"])
            self.assertEqual(config["format"], "asym_w4a8_int8")
            self.assertEqual(
                tuple(converted[f"{base}.weight_s_rel"].shape), (8, 512 // 16)
            )


class TestW4A8SkipExclusions(unittest.TestCase):
    """Regression: W4A8 primary format must respect AVOID_KEY_NAMES base exclusions.

    Non-Linear tensors (embeddings, norms, heads...) must NEVER be W4A8-quantized:
    the format packs two 4-bit codes per byte ([N, K] -> [N, K/2]), which only
    ComfyUI's Linear-op path (asym_w4a8_int8) can decode. Packing e.g.
    model.embed_tokens.weight produces a size-mismatch error at load time.
    """

    def _build_model(self, tmpdir):
        input_path = os.path.join(tmpdir, "input.safetensors")
        output_path = os.path.join(tmpdir, "output.safetensors")
        g = torch.Generator().manual_seed(7)
        tensors = {
            # 2D but Non-Linear: AVOID_KEY_NAMES hits -> must stay raw
            "model.embed_tokens.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
            # 1D norm: never in the quantization loop, must pass through verbatim
            "model.norm.weight": torch.randn(512, generator=g, dtype=torch.bfloat16),
            "model.layers.0.input_layernorm.weight": torch.randn(512, generator=g, dtype=torch.bfloat16),
            # Linear weight: must be W4A8-quantized
            "model.layers.0.mlp.up_proj.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
        }
        save_file(tensors, input_path)
        return input_path, output_path, tensors

    def _convert(self, input_path, output_path, **extra):
        kwargs = dict(
            comfy_quant=True,
            filter_flags={},
            calib_samples=2,
            seed=1,
            no_learned_rounding=True,
            save_quant_metadata=True,
            low_memory=True,
            device="cpu",
            primary_format="w4a8",
            convrot_group_size=256,
        )
        kwargs.update(extra)
        convert_to_fp8_scaled(input_path, output_path, **kwargs)

    def test_avoid_key_names_left_unquantized(self):
        """embed_tokens/lm_head (AVOID_KEY_NAMES) must keep dtype+shape; Linear must be packed."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path, tensors = self._build_model(tmpdir)
            self._convert(input_path, output_path)
            out = load_file(output_path)

            # Non-Linear 2D weights: raw passthrough (same dtype, FULL [N, K] shape),
            # no comfy_quant / contract side-tensors
            for base in ("model.embed_tokens", "lm_head"):
                self.assertEqual(
                    out[f"{base}.weight"].dtype, torch.bfloat16, f"{base}.weight must stay bfloat16"
                )
                self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 512), f"{base}.weight shape must be preserved")
                self.assertNotIn(f"{base}.comfy_quant", out, f"{base} must not be quantized")
                self.assertNotIn(f"{base}.weight_s_rel", out)

            # 1D norms pass through verbatim
            for key in ("model.norm.weight", "model.layers.0.input_layernorm.weight"):
                self.assertEqual(out[key].dtype, torch.bfloat16, f"{key} must stay bfloat16")
                self.assertEqual(tuple(out[key].shape), (512,), f"{key} shape must be preserved")

            # Linear weight is W4A8-quantized: packed INT8 [N, K/2] + contract tensors
            base = "model.layers.0.mlp.up_proj"
            self.assertEqual(out[f"{base}.weight"].dtype, torch.int8)
            self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 256))
            config = tensor_to_dict(out[f"{base}.comfy_quant"])
            self.assertEqual(config["format"], "asym_w4a8_int8")
            self.assertTrue(
                torch.equal(
                    out[f"{base}.weight_codebook"],
                    torch.tensor(W4A8_FIXED_LUT, dtype=torch.float32),
                )
            )

    def test_model_filter_flags_still_applied(self):
        """MODEL_FILTERS exclusions (e.g. mistral) extend the base AVOID_KEY_NAMES skips."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path, _ = self._build_model(tmpdir)
            self._convert(input_path, output_path, filter_flags={"mistral": True})
            out = load_file(output_path)
            # mistral exclude == AVOID_KEY_NAMES: embed_tokens still raw
            self.assertEqual(out["model.embed_tokens.weight"].dtype, torch.bfloat16)
            self.assertEqual(tuple(out["model.embed_tokens.weight"].shape), (8, 512))
            # Linear layers unaffected by the filter: still quantized
            base = "model.layers.0.mlp.up_proj"
            self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 256))

    def test_shape_packed_fallback_never_packs_excluded_layers(self):
        """--fallback w4a8 must not pack excluded (non-Linear) layers either."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path, output_path, _ = self._build_model(tmpdir)
            self._convert(
                input_path,
                output_path,
                primary_format=None,  # default FP8 primary
                exclude_layers=r"embed_tokens",
                fallback="w4a8",
            )
            out = load_file(output_path)
            # Excluded layer keeps original dtype+shape despite w4a8 fallback
            self.assertEqual(out["model.embed_tokens.weight"].dtype, torch.bfloat16)
            self.assertEqual(tuple(out["model.embed_tokens.weight"].shape), (8, 512))
            self.assertNotIn("model.embed_tokens.comfy_quant", out)
            # Non-excluded Linear layer uses the primary (FP8) path: quantized (not packed)
            base = "model.layers.0.mlp.up_proj"
            self.assertNotEqual(tuple(out[f"{base}.weight"].shape), (8, 256))

    def test_k_not_divisible_by_convrot_group_kept_original(self):
        """SDXL skip-connection dims (K=640, 640 % 256 != 0) must be skipped
        and kept at original precision instead of raising ValueError, while
        divisible Linear weights are still W4A8-quantized. The bias of the
        skipped layer must pass through untouched (no weight change -> no
        bias correction).
        """
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")
            g = torch.Generator().manual_seed(9)
            bias = torch.randn(8, generator=g, dtype=torch.bfloat16)
            tensors = {
                # SDXL-style K=640 Linear: not divisible by convrot_group_size 256
                "input_blocks.4.1.proj_in.weight": torch.randn(8, 640, generator=g, dtype=torch.bfloat16),
                "input_blocks.4.1.proj_in.bias": bias.clone(),
                # Divisible Linear weight: must be W4A8-quantized
                "input_blocks.1.1.proj_in.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
            }
            save_file(tensors, input_path)
            self._convert(input_path, output_path)
            out = load_file(output_path)

            # K=640 layer: raw passthrough (same dtype, FULL [N, K] shape),
            # no comfy_quant / contract side-tensors
            base = "input_blocks.4.1.proj_in"
            self.assertEqual(out[f"{base}.weight"].dtype, torch.bfloat16)
            self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 640))
            self.assertNotIn(f"{base}.comfy_quant", out)
            self.assertNotIn(f"{base}.weight_s_rel", out)
            # Bias untouched (bitwise) since no weight change occurred
            self.assertTrue(torch.equal(out[f"{base}.bias"], bias))

            # Divisible layer still W4A8-quantized: packed INT8 [N, K/2]
            base = "input_blocks.1.1.proj_in"
            self.assertEqual(out[f"{base}.weight"].dtype, torch.int8)
            self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 256))

    def test_vlm_pos_embed_left_unquantized(self):
        """Regression: pos_embed packed to half-K broke ComfyUI load; must stay raw."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")
            g = torch.Generator().manual_seed(7)
            tensors = {
                # VLM embedding-like 2D weights not covered by AVOID_KEY_NAMES
                "model.visual.pos_embed.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.visual.pos_emb.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.visual.patch_embed.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.visual.patch_embedder.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                # Linear weight: must still be W4A8-quantized
                "model.visual.blocks.0.mlp.up_proj.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
            }
            save_file(tensors, input_path)
            # No filter_flags: generic hardening must apply on its own
            self._convert(input_path, output_path)
            out = load_file(output_path)

            # Embedding-like weights: raw passthrough (original dtype + FULL shape),
            # no comfy_quant / contract side-tensors
            for base in ("model.visual.pos_embed", "model.visual.pos_emb",
                         "model.visual.patch_embed", "model.visual.patch_embedder"):
                self.assertEqual(
                    out[f"{base}.weight"].dtype, torch.bfloat16, f"{base}.weight must stay bfloat16"
                )
                self.assertEqual(
                    tuple(out[f"{base}.weight"].shape), (8, 512), f"{base}.weight shape must be preserved"
                )
                self.assertNotIn(f"{base}.comfy_quant", out, f"{base} must not be quantized")
                self.assertNotIn(f"{base}.weight_s_rel", out)

            # Linear weight inside the visual tower is still W4A8-quantized:
            # packed INT8 [N, K/2] + contract tensors
            base = "model.visual.blocks.0.mlp.up_proj"
            self.assertEqual(out[f"{base}.weight"].dtype, torch.int8)
            self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 256))
            config = tensor_to_dict(out[f"{base}.comfy_quant"])
            self.assertEqual(config["format"], "asym_w4a8_int8")

    def test_sdxl_detection_critical_tensors_left_unquantized(self):
        """Regression: packed label_emb/attn2.to_k broke ComfyUI config detection (adm_in_channels 1408 vs 2816)."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, "input.safetensors")
            output_path = os.path.join(tmpdir, "output.safetensors")
            g = torch.Generator().manual_seed(11)
            # K=512 divisible by convrot_group_size=256: only the
            # SHAPE_PACKED_AVOID_KEY_NAMES exclusion can skip these tensors.
            tensors = {
                # SDXL UNet detection-critical linears (shapes feed unet_config)
                "model.diffusion_model.label_emb.0.0.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.diffusion_model.label_emb.0.2.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                # CLIP text encoder nn.Embedding weights (non-Linear)
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                # CLIP-L / CLIP-G text-encoder Linear weights: ComfyUI never wraps
                # text encoders with quant ops (sd1_clip.py falls back to
                # manual_cast -> plain F.linear), so they must never be packed.
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.out_proj.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                # VAE (defensive: no 2D linears in SDXL VAE, but must never pack)
                "first_stage_model.decoder.conv_in.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "vae.decoder.mid.attn_1.to_q.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                # Regular Linear weights: must still be W4A8-quantized
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
                "model.diffusion_model.time_embed.2.weight": torch.randn(8, 512, generator=g, dtype=torch.bfloat16),
            }
            save_file(tensors, input_path)
            # No filter_flags: generic hardening must apply on its own
            self._convert(input_path, output_path)
            out = load_file(output_path)

            # Detection-critical / Embedding tensors: raw passthrough
            # (original dtype + FULL shape), no comfy_quant / contract tensors
            raw_bases = (
                "model.diffusion_model.label_emb.0.0",
                "model.diffusion_model.label_emb.0.2",
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k",
                "conditioner.embedders.0.transformer.text_model.embeddings.token_embedding",
                "conditioner.embedders.0.transformer.text_model.embeddings.position_embedding",
                # Text-encoder Linears (CLIP-L + CLIP-G): runtime executes them
                # with plain nn.Linear -> packed weights would crash
                # (77x1280 @ 640x1280 style matmul shape error).
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.self_attn.out_proj",
                "conditioner.embedders.0.transformer.text_model.encoder.layers.0.mlp.fc1",
                "conditioner.embedders.1.model.transformer.resblocks.0.attn.out_proj",
                "conditioner.embedders.1.model.transformer.resblocks.0.mlp.c_fc",
                # VAE prefixes
                "first_stage_model.decoder.conv_in",
                "vae.decoder.mid.attn_1.to_q",
            )
            for base in raw_bases:
                self.assertEqual(
                    out[f"{base}.weight"].dtype, torch.bfloat16, f"{base}.weight must stay bfloat16"
                )
                self.assertEqual(
                    tuple(out[f"{base}.weight"].shape), (8, 512), f"{base}.weight shape must be preserved"
                )
                self.assertNotIn(f"{base}.comfy_quant", out, f"{base} must not be quantized")
                self.assertNotIn(f"{base}.weight_s_rel", out)

            # Regular Linears (incl. attn2.to_out.0 and time_embed.2) are still
            # W4A8-quantized: packed INT8 [N, K/2] + contract tensors
            for base in (
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_out.0",
                "model.diffusion_model.time_embed.2",
            ):
                self.assertEqual(out[f"{base}.weight"].dtype, torch.int8, f"{base}.weight must be packed int8")
                self.assertEqual(tuple(out[f"{base}.weight"].shape), (8, 256))
                config = tensor_to_dict(out[f"{base}.comfy_quant"])
                self.assertEqual(config["format"], "asym_w4a8_int8")


class TestLayerConfigFormatMapping(unittest.TestCase):
    def test_w4a8_in_valid_formats(self):
        self.assertIn("w4a8", VALID_QUANT_FORMATS)

    def test_w4a8_maps_to_w4a8(self):
        self.assertEqual(resolve_layer_config_format("w4a8"), "w4a8")

    def test_layer_config_accepts_w4a8(self):
        from convert_to_quant.config.layer_config import load_layer_config

        import tempfile

        config = {"mlp": {"format": "w4a8"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        loaded = load_layer_config(path)  # must not raise (format validation)
        self.assertIn("mlp", loaded)
        import os

        os.unlink(path)

    def test_layer_config_rejects_bad_format(self):
        from convert_to_quant.config.layer_config import load_layer_config

        import tempfile

        config = {"mlp": {"format": "asym_w4a8_int8_not_a_layer_config_format"}}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            path = f.name
        with self.assertRaises(ValueError):
            load_layer_config(path)
        import os

        os.unlink(path)


class TestCliParsing(unittest.TestCase):
    def _parse(self, argv):
        from convert_to_quant.cli.main import get_parser

        parser = get_parser()
        return parser.parse_args(argv)

    def test_w4a8_flag_parses(self):
        args = self._parse(["-i", "in.safetensors", "--w4a8"])
        self.assertTrue(args.w4a8)
        self.assertFalse(args.int8)
        self.assertFalse(args.nvfp4)
        self.assertFalse(args.mxfp8)

    def test_w4a8_in_experimental_args(self):
        from convert_to_quant.cli.argument_parser import EXPERIMENTAL_ARGS

        self.assertIn("w4a8", EXPERIMENTAL_ARGS)

    def test_w4a8_as_custom_type(self):
        args = self._parse(
            ["-i", "in.safetensors", "--custom-layers", "mlp", "--custom-type", "w4a8"]
        )
        self.assertEqual(args.custom_type, "w4a8")

    def test_w4a8_as_fallback(self):
        args = self._parse(["-i", "in.safetensors", "--fallback", "w4a8"])
        self.assertEqual(args.fallback, "w4a8")

    def test_w4a8_default_convrot_group(self):
        args = self._parse(["-i", "in.safetensors", "--w4a8"])
        self.assertEqual(args.convrot_group_size, 256)


if __name__ == "__main__":
    unittest.main()
