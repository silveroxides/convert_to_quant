"""
Filter flag regression tests.

Creates a small synthetic model containing 1D/2D/3D/4D tensors with a spread
of layer names, then runs each conversion path (FP8, NVFP4, MXFP8) through
convert_to_fp8_scaled / convert_to_nvfp4 / convert_to_mxfp8 with various
filter_flags combinations and checks that:

  - Excluded layers (via filter "exclude", "highprec", or "remove") are handled correctly
  - 1D/3D/4D tensors are never quantized regardless of layer name
  - AVOID_KEY_NAMES (norm, bias, embed_tokens…) are skipped in nvfp4/mxfp8 paths
  - TEXT_MODEL_ALIASES (qwen35 -> generic_text) propagate correctly
  - Quantized layers produce .weight_scale and .comfy_quant
  - Skipped layers do NOT produce .weight_scale / .comfy_quant / .input_scale
  - "remove" key deletes layers entirely from output (t5xxl decoder removal)

All tests run on CPU with simple=True to avoid GPU and keep runtimes short.
Real .safetensors files are written to CWD and cleaned up in tearDown.
"""

import os
import unittest
import torch
from safetensors.torch import save_file, load_file

from convert_to_quant.formats.fp8_conversion import convert_to_fp8_scaled
from convert_to_quant.formats.nvfp4_conversion import convert_to_nvfp4
from convert_to_quant.formats.mxfp8_conversion import convert_to_mxfp8
from convert_to_quant.cli.main import extract_filter_flags


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_quant_artifacts(tensors: dict, base: str) -> bool:
    """Return True if any quantization artifact exists for a given base name."""
    return any(
        k.startswith(base + ".") and k != base + ".weight" for k in tensors if k not in (base + ".weight", base + ".bias")
    )


def _is_quantized(tensors: dict, base: str) -> bool:
    """Return True if weight_scale (comfy path) or scale_weight (legacy) exists."""
    return f"{base}.weight_scale" in tensors or f"{base}.scale_weight" in tensors


def _has_comfy_quant(tensors: dict, base: str) -> bool:
    return f"{base}.comfy_quant" in tensors


def _has_input_scale(tensors: dict, base: str) -> bool:
    return f"{base}.input_scale" in tensors or f"{base}.scale_input" in tensors


def _build_model() -> dict:
    """
    Build a small synthetic model with:
      - Several 2D .weight tensors (quantizable)
      - 1D / 3D / 4D tensors (must never be quantized)
      - Layer names covering: normal layers, AVOID_KEY_NAMES hits,
        custom filter patterns (for anima, qwen35, t5xxl)
    """
    t = {}

    # ---- 2D weights that SHOULD be quantized (no filter active) ----
    t["transformer.blocks.2.attn.qkv.weight"] = torch.randn(64, 64)
    t["transformer.blocks.2.attn.proj.weight"] = torch.randn(64, 64)
    t["transformer.blocks.2.mlp.fc1.weight"] = torch.randn(128, 64)
    t["transformer.blocks.2.mlp.fc2.weight"] = torch.randn(64, 128)
    t["net.blocks.2.attn.weight"] = torch.randn(64, 64)

    # ---- 2D weights that match anima highprec patterns ----
    # "net.blocks.0."  "net.blocks.1.adaln_modulation"  "final_layer"
    # "llm_adapter"    "t_embedder"                     "x_embedder"
    t["net.blocks.0.attn.weight"] = torch.randn(64, 64)
    t["net.blocks.1.adaln_modulation.weight"] = torch.randn(64, 64)
    t["final_layer.linear.weight"] = torch.randn(64, 64)
    t["llm_adapter.proj.weight"] = torch.randn(64, 64)
    t["t_embedder.mlp.0.weight"] = torch.randn(64, 64)
    t["x_embedder.proj.weight"] = torch.randn(64, 64)

    # ---- 2D weights that match qwen35 exclude patterns ----
    # ".layers.0."  ".layers.63."  "lm_head"  "embed_tokens"
    # "in_proj_a"   "in_proj_b"    "merger"   "mtp.fc"
    # "visual.pos_embed"  "visual.patch_embed"  "visual.blocks.0."
    t["model.layers.0.attn.weight"] = torch.randn(64, 64)
    t["model.layers.63.attn.weight"] = torch.randn(64, 64)
    t["lm_head.weight"] = torch.randn(64, 64)
    t["embed_tokens.weight"] = torch.randn(64, 64)
    t["in_proj_a.weight"] = torch.randn(64, 64)
    t["in_proj_b.weight"] = torch.randn(64, 64)
    t["merger.dense.weight"] = torch.randn(64, 64)
    t["mtp.fc.weight"] = torch.randn(64, 64)
    t["visual.pos_embed.weight"] = torch.randn(64, 64)
    t["visual.patch_embed.proj.weight"] = torch.randn(64, 64)
    t["visual.blocks.0.attn.weight"] = torch.randn(64, 64)

    # ---- 2D weights matching t5xxl "remove" pattern (decoder) ----
    t["decoder.block.0.attn.weight"] = torch.randn(64, 64)
    t["lm_head.proj.weight"] = torch.randn(64, 64)  # also lm_head

    # ---- 1D tensors (bias / norm / embedding vectors) — never quantized ----
    t["transformer.blocks.2.attn.qkv.bias"] = torch.randn(64)
    t["transformer.norm.weight"] = torch.randn(64)  # hits AVOID_KEY_NAMES "norm"
    t["transformer.norm.bias"] = torch.randn(64)  # hits AVOID_KEY_NAMES "bias"

    # ---- 3D tensor ----
    t["conv1d_layer.weight"] = torch.randn(16, 8, 3)

    # ---- 4D tensor (conv2d) ----
    t["conv2d_layer.weight"] = torch.randn(16, 8, 3, 3)

    return t


# FP8 conversion kwargs shared across FP8 tests (CPU, simple, no learned rounding)
_FP8_KWARGS = dict(
    comfy_quant=True,
    calib_samples=4,
    seed=0,
    int8=False,
    no_learned_rounding=True,
    save_quant_metadata=False,
    low_memory=False,
    device="cpu",
    scaling_mode="tensor",
    optimizer="prodigy",
    num_iter=1,
    lr=1.0,
    lr_schedule="plateau",
    top_p=0.2,
    min_k=8,
    max_k=64,
    full_matrix=False,
    lr_gamma=0.99,
    lr_patience=1,
    lr_factor=0.95,
    lr_min=1e-8,
    lr_cooldown=0,
    lr_threshold=0.0,
    lr_adaptive_mode="simple-reset",
    lr_shape_influence=1.0,
    lr_threshold_mode="rel",
    early_stop_loss=5e-9,
    early_stop_lr=1.01e-8,
    early_stop_stall=2000,
    use_speed=False,
    extract_lora=False,
    lora_rank=4,
    lora_target=None,
    lora_depth=-1,
    lora_ar_threshold=0.0,
)

# NVFP4 kwargs
_NVFP4_KWARGS = dict(
    simple=True,
    calib_samples=4,
    seed=0,
    num_iter=1,
    optimizer="prodigy",
    lr=1.0,
    lr_schedule="plateau",
    top_p=0.2,
    min_k=8,
    max_k=64,
    full_matrix=False,
    lr_gamma=0.99,
    lr_patience=1,
    lr_factor=0.95,
    lr_min=1e-8,
    lr_cooldown=0,
    lr_threshold=0.0,
    lr_adaptive_mode="simple-reset",
    lr_shape_influence=1.0,
    lr_threshold_mode="rel",
    early_stop_loss=5e-9,
    early_stop_lr=1.01e-8,
    early_stop_stall=2000,
    low_memory=False,
    use_speed=False,
    extract_lora=False,
    lora_rank=4,
    lora_target=None,
    lora_depth=-1,
    lora_ar_threshold=0.0,
)

# MXFP8 kwargs
_MXFP8_KWARGS = dict(
    simple=True,
    calib_samples=4,
    seed=0,
    num_iter=1,
    optimizer="prodigy",
    lr=1.0,
    lr_schedule="plateau",
    top_p=0.2,
    min_k=8,
    max_k=64,
    full_matrix=False,
    lr_gamma=0.99,
    lr_patience=1,
    lr_factor=0.95,
    lr_min=1e-8,
    lr_cooldown=0,
    lr_threshold=0.0,
    lr_adaptive_mode="simple-reset",
    lr_shape_influence=1.0,
    lr_threshold_mode="rel",
    early_stop_loss=5e-9,
    early_stop_lr=1.01e-8,
    early_stop_stall=2000,
    low_memory=False,
    use_speed=False,
    extract_lora=False,
    lora_rank=4,
    lora_target=None,
    lora_depth=-1,
    lora_ar_threshold=0.0,
)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestFilterFlags(unittest.TestCase):
    def setUp(self):
        self.input_file = "_test_filter_input.safetensors"
        self.output_file = "_test_filter_output.safetensors"
        save_file(_build_model(), self.input_file)

    def tearDown(self):
        for f in [self.input_file, self.output_file]:
            if os.path.exists(f):
                os.remove(f)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_fp8(self, filter_flags, **extra):
        kw = {**_FP8_KWARGS, **extra}
        convert_to_fp8_scaled(self.input_file, self.output_file, filter_flags=filter_flags, **kw)
        return load_file(self.output_file)

    def _run_nvfp4(self, filter_flags, **extra):
        kw = {**_NVFP4_KWARGS, **extra}
        convert_to_nvfp4(self.input_file, self.output_file, filter_flags=filter_flags, **kw)
        return load_file(self.output_file)

    def _run_mxfp8(self, filter_flags, **extra):
        kw = {**_MXFP8_KWARGS, **extra}
        convert_to_mxfp8(self.input_file, self.output_file, filter_flags=filter_flags, **kw)
        return load_file(self.output_file)

    # ------------------------------------------------------------------
    # 1. Non-2D tensors are never quantized (all paths)
    # ------------------------------------------------------------------

    def test_fp8_non2d_never_quantized(self):
        """1D/3D/4D tensors must be copied verbatim with no quant artifacts."""
        out = self._run_fp8({})
        for base in ("conv1d_layer", "conv2d_layer"):
            self.assertIn(f"{base}.weight", out, f"{base}.weight missing from output")
            self.assertFalse(_is_quantized(out, base), f"{base} should not be quantized (non-2D)")
            self.assertFalse(_has_comfy_quant(out, base), f"{base} must not have comfy_quant (non-2D)")

    def test_nvfp4_non2d_never_quantized(self):
        out = self._run_nvfp4({})
        for base in ("conv1d_layer", "conv2d_layer"):
            self.assertIn(f"{base}.weight", out)
            self.assertFalse(_is_quantized(out, base))
            self.assertFalse(_has_comfy_quant(out, base))

    def test_mxfp8_non2d_never_quantized(self):
        out = self._run_mxfp8({})
        for base in ("conv1d_layer", "conv2d_layer"):
            self.assertIn(f"{base}.weight", out)
            self.assertFalse(_is_quantized(out, base))
            self.assertFalse(_has_comfy_quant(out, base))

    # ------------------------------------------------------------------
    # 2. 1D tensors (bias/norm) are passed through untouched
    # ------------------------------------------------------------------

    def test_fp8_1d_passthrough(self):
        """1D tensors must appear in output unchanged."""
        out = self._run_fp8({})
        self.assertIn("transformer.blocks.2.attn.qkv.bias", out)
        self.assertIn("transformer.norm.weight", out)
        self.assertIn("transformer.norm.bias", out)
        # No quant artifacts on 1D layers
        self.assertFalse(_is_quantized(out, "transformer.norm"))

    def test_nvfp4_1d_passthrough(self):
        out = self._run_nvfp4({})
        self.assertIn("transformer.blocks.2.attn.qkv.bias", out)

    # ------------------------------------------------------------------
    # 3. Normal 2D layers ARE quantized with no filters active
    # ------------------------------------------------------------------

    def test_fp8_normal_layers_quantized(self):
        out = self._run_fp8({})
        for base in ("transformer.blocks.2.attn.qkv", "transformer.blocks.2.mlp.fc1", "net.blocks.2.attn"):
            self.assertTrue(_is_quantized(out, base), f"{base} should be quantized")
            self.assertTrue(_has_comfy_quant(out, base), f"{base} should have comfy_quant")

    def test_nvfp4_normal_layers_quantized(self):
        out = self._run_nvfp4({})
        for base in ("transformer.blocks.2.attn.qkv", "transformer.blocks.2.mlp.fc1"):
            self.assertTrue(_is_quantized(out, base), f"{base} should be quantized (nvfp4)")
            self.assertTrue(_has_comfy_quant(out, base))

    def test_mxfp8_normal_layers_quantized(self):
        out = self._run_mxfp8({})
        for base in ("transformer.blocks.2.attn.qkv", "transformer.blocks.2.mlp.fc1"):
            self.assertTrue(_is_quantized(out, base), f"{base} should be quantized (mxfp8)")

    # ------------------------------------------------------------------
    # 4. --anima filter (highprec) — FP8 path
    # ------------------------------------------------------------------

    ANIMA_SKIPPED = [
        "net.blocks.0.attn",
        "net.blocks.1.adaln_modulation",
        "final_layer.linear",
        "llm_adapter.proj",
        "t_embedder.mlp.0",
        "x_embedder.proj",
    ]
    ANIMA_KEPT = ["transformer.blocks.2.attn.qkv", "net.blocks.2.attn"]

    def test_fp8_anima_flag_skips_highprec_layers(self):
        """--anima must prevent quantization of its highprec patterns."""
        out = self._run_fp8({"anima": True})
        for base in self.ANIMA_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"--anima: {base} should NOT be quantized")
            self.assertFalse(_has_comfy_quant(out, base), f"--anima: {base} must not have comfy_quant")
            # Original weight must still be in output
            self.assertIn(f"{base}.weight", out, f"--anima: {base}.weight must be preserved")

    def test_fp8_anima_flag_keeps_other_layers(self):
        """--anima must not affect layers outside its patterns."""
        out = self._run_fp8({"anima": True})
        for base in self.ANIMA_KEPT:
            self.assertTrue(_is_quantized(out, base), f"--anima: {base} should still be quantized")

    def test_nvfp4_anima_flag_skips_highprec_layers(self):
        out = self._run_nvfp4({"anima": True})
        for base in self.ANIMA_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"nvfp4 --anima: {base} should NOT be quantized")
            self.assertIn(f"{base}.weight", out)

    def test_mxfp8_anima_flag_skips_highprec_layers(self):
        out = self._run_mxfp8({"anima": True})
        for base in self.ANIMA_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"mxfp8 --anima: {base} should NOT be quantized")
            self.assertIn(f"{base}.weight", out)

    # ------------------------------------------------------------------
    # 5. --qwen35 filter (exclude) — all paths
    # ------------------------------------------------------------------

    QWEN35_SKIPPED = [
        "model.layers.0.attn",
        "model.layers.63.attn",
        "lm_head",
        "embed_tokens",
        "in_proj_a",
        "in_proj_b",
        "merger.dense",
        "mtp.fc",
        "visual.pos_embed",
        "visual.patch_embed.proj",
        "visual.blocks.0.attn",
    ]
    QWEN35_KEPT = ["transformer.blocks.2.attn.qkv", "net.blocks.2.attn"]

    def test_fp8_qwen35_flag_skips_excluded_layers(self):
        out = self._run_fp8({"qwen35": True, "generic_text": True})
        for base in self.QWEN35_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"--qwen35: {base} should NOT be quantized")
            self.assertFalse(_has_comfy_quant(out, base))
            self.assertIn(f"{base}.weight", out, f"--qwen35: {base}.weight must be preserved")

    def test_fp8_qwen35_flag_keeps_other_layers(self):
        out = self._run_fp8({"qwen35": True, "generic_text": True})
        for base in self.QWEN35_KEPT:
            self.assertTrue(_is_quantized(out, base), f"--qwen35: {base} should still be quantized")

    def test_nvfp4_qwen35_flag_skips_excluded_layers(self):
        out = self._run_nvfp4({"qwen35": True, "generic_text": True})
        for base in self.QWEN35_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"nvfp4 --qwen35: {base} should NOT be quantized")
            self.assertIn(f"{base}.weight", out)

    def test_mxfp8_qwen35_flag_skips_excluded_layers(self):
        out = self._run_mxfp8({"qwen35": True, "generic_text": True})
        for base in self.QWEN35_SKIPPED:
            self.assertFalse(_is_quantized(out, base), f"mxfp8 --qwen35: {base} should NOT be quantized")
            self.assertIn(f"{base}.weight", out)

    # ------------------------------------------------------------------
    # 6. TEXT_MODEL_ALIASES: qwen35 via extract_filter_flags injects generic_text
    # ------------------------------------------------------------------

    def test_extract_filter_flags_qwen35_injects_generic_text(self):
        """extract_filter_flags must set generic_text=True when qwen35 is set."""
        # Build a fake args namespace with all MODEL_FILTERS keys set to False
        # except qwen35
        from convert_to_quant.constants import MODEL_FILTERS
        import types

        ns = types.SimpleNamespace()
        for name in MODEL_FILTERS.keys():
            setattr(ns, name, False)
        setattr(ns, "qwen35", True)

        flags = extract_filter_flags(ns)

        self.assertTrue(flags.get("qwen35"), "qwen35 must be True in flags")
        self.assertTrue(flags.get("generic_text"), "generic_text must be auto-injected when qwen35 is set")

    def test_extract_filter_flags_no_alias_without_qwen35(self):
        """generic_text must NOT be injected when qwen35 is not set."""
        from convert_to_quant.constants import MODEL_FILTERS
        import types

        ns = types.SimpleNamespace()
        for name in MODEL_FILTERS.keys():
            setattr(ns, name, False)

        flags = extract_filter_flags(ns)

        self.assertNotIn("generic_text", flags, "generic_text must not appear when no alias is active")

    # ------------------------------------------------------------------
    # 7. --qwen35 input_scale behavior via generic_text injection (FP8)
    # ------------------------------------------------------------------

    def test_fp8_qwen35_generic_text_produces_input_scale(self):
        """
        With qwen35 + generic_text active, quantized layers must have
        input_scale set to the computed weight scale (not 1.0 scalar),
        same as mistral/t5xxl behavior.
        """
        out = self._run_fp8({"qwen35": True, "generic_text": True})
        # A layer that is NOT excluded by qwen35 should have input_scale
        base = "transformer.blocks.2.attn.qkv"
        self.assertIn(f"{base}.weight_scale", out, f"{base} should be quantized")
        self.assertIn(f"{base}.input_scale", out, f"generic_text must produce input_scale for {base}")

    def test_fp8_no_text_filter_no_input_scale_by_default(self):
        """Without any text filter, input_scale must not be added."""
        out = self._run_fp8({})
        base = "transformer.blocks.2.attn.qkv"
        self.assertNotIn(f"{base}.input_scale", out, "No text filter active: input_scale must not appear")

    # ------------------------------------------------------------------
    # 8. t5xxl "remove" key deletes layers from output (FP8 path)
    # ------------------------------------------------------------------

    def test_fp8_t5xxl_remove_deletes_decoder_tensors(self):
        """
        --t5xxl must completely remove tensors matching T5XXL_REMOVE_KEY_NAMES
        (decoder.*, lm_head.*) from the output.
        """
        out = self._run_fp8({"t5xxl": True})

        # These contain "decoder" or "lm_head" — must be absent entirely
        removed = ["decoder.block.0.attn.weight", "lm_head.proj.weight", "lm_head.weight"]
        for key in removed:
            self.assertNotIn(key, out, f"t5xxl remove: {key} must be deleted from output")

    def test_fp8_t5xxl_remove_keeps_non_decoder_tensors(self):
        """Non-decoder layers must still be processed normally with --t5xxl."""
        out = self._run_fp8({"t5xxl": True})
        base = "transformer.blocks.2.attn.qkv"
        self.assertTrue(_is_quantized(out, base), f"t5xxl: {base} should still be quantized")

    # ------------------------------------------------------------------
    # 9. AVOID_KEY_NAMES: norm/bias/embed_tokens skipped by nvfp4/mxfp8 paths
    #    (these substring patterns are pre-loaded from AVOID_KEY_NAMES list
    #     in nvfp4/mxfp8 even with empty filter_flags)
    # ------------------------------------------------------------------

    def test_nvfp4_avoid_key_names_skipped(self):
        """embed_tokens and lm_head are in AVOID_KEY_NAMES; nvfp4 skips them."""
        out = self._run_nvfp4({})
        # embed_tokens is in AVOID_KEY_NAMES
        self.assertFalse(_is_quantized(out, "embed_tokens"), "embed_tokens (AVOID_KEY_NAMES) must not be quantized in nvfp4")
        self.assertFalse(_is_quantized(out, "lm_head"), "lm_head (AVOID_KEY_NAMES) must not be quantized in nvfp4")

    def test_mxfp8_avoid_key_names_skipped(self):
        out = self._run_mxfp8({})
        self.assertFalse(_is_quantized(out, "embed_tokens"))
        self.assertFalse(_is_quantized(out, "lm_head"))

    # ------------------------------------------------------------------
    # 10. exclude and highprec are functionally identical (FP8 path)
    # ------------------------------------------------------------------

    def test_fp8_highprec_skips_anima_target(self):
        """A filter using "highprec" must skip quantization of its patterns."""
        # anima uses highprec
        out = self._run_fp8({"anima": True})
        self.assertFalse(_is_quantized(out, "net.blocks.0.attn"), "highprec (anima): net.blocks.0.attn must not be quantized")

    def test_fp8_exclude_skips_qwen35_target(self):
        """A filter using "exclude" must skip quantization of its patterns."""
        # qwen35 uses exclude — use a separate output file to avoid Windows
        # file-lock when calling _run_fp8 twice in the same test method
        out = self._run_fp8({"qwen35": True})
        self.assertFalse(
            _is_quantized(out, "model.layers.0.attn"), "exclude (qwen35): model.layers.0.attn must not be quantized"
        )

    # ------------------------------------------------------------------
    # 11. --exclude-layers regex (FP8 path)
    # ------------------------------------------------------------------

    def test_fp8_exclude_layers_regex(self):
        """--exclude-layers regex must skip matching layers."""
        out = self._run_fp8({}, exclude_layers=r"net\.blocks\.2")
        self.assertFalse(_is_quantized(out, "net.blocks.2.attn"), "exclude-layers regex: net.blocks.2.attn should be skipped")
        # Non-matching layer still quantized
        self.assertTrue(_is_quantized(out, "transformer.blocks.2.attn.qkv"))

    # ------------------------------------------------------------------
    # 12. Dimension shapes are preserved for skipped layers
    # ------------------------------------------------------------------

    def test_fp8_skipped_layer_weight_shape_preserved(self):
        """Skipped layers must be written with their original shape and dtype."""
        model = _build_model()
        out = self._run_fp8({"anima": True})

        for base in self.ANIMA_SKIPPED:
            key = f"{base}.weight"
            if key in model:
                orig = model[key]
                self.assertEqual(out[key].shape, orig.shape, f"Shape mismatch for skipped {key}")
                self.assertEqual(out[key].dtype, orig.dtype, f"Dtype mismatch for skipped {key}")

    def test_fp8_3d_4d_shape_preserved(self):
        """3D and 4D tensors must be copied with original shape and dtype."""
        model = _build_model()
        out = self._run_fp8({})
        for key in ("conv1d_layer.weight", "conv2d_layer.weight"):
            orig = model[key]
            self.assertEqual(out[key].shape, orig.shape)
            self.assertEqual(out[key].dtype, orig.dtype)


if __name__ == "__main__":
    unittest.main()
