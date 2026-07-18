"""Focused regression tests for the model filter registry."""

import unittest

from convert_to_quant.constants import MODEL_FILTERS, build_exclusion_patterns


class ModelFilterRegistryTests(unittest.TestCase):
    def _skip_patterns(self, name):
        skip, highprec, remove = build_exclusion_patterns({name: True})
        self.assertEqual(skip, highprec)
        self.assertEqual(remove, [])
        return skip

    def test_flux_family_filters_protect_expected_sensitive_layers(self):
        for name in ("flux1", "flux2"):
            patterns = self._skip_patterns(name)
            for expected in ("stream_modulation", "guidance_in", "time_in", "final_layer", "img_in", "txt_in"):
                self.assertIn(expected, patterns)

        klein_patterns = self._skip_patterns("flux_klein")
        self.assertNotIn("guidance_in", klein_patterns)
        self.assertIn("stream_modulation", klein_patterns)

    def test_ernie_image_filter_matches_author_recipe(self):
        patterns = self._skip_patterns("ernie_image")
        for expected in ("time_embedding", "adaLN_modulation", "final_linear", "final_norm", "x_embedder", "layers.0.self_attention", "layers.0.mlp.gate_proj", "layers.0.mlp.up_proj", "text_proj"):
            self.assertIn(expected, patterns)

    def test_qwen35_protects_supported_variant_boundaries_and_visual_stack(self):
        patterns = self._skip_patterns("qwen35")
        for expected in (".layers.0.", ".layers.23.", ".layers.31.", ".layers.63.", "visual."):
            self.assertIn(expected, patterns)

    def test_gemma4_protects_embeddings_kv_and_multimodal_projectors(self):
        patterns = self._skip_patterns("gemma4")
        for expected in ("embed_tokens", "self_attn.k_proj", "self_attn.v_proj", "per_layer_model_projection", "audio_projector", "multi_modal_projector"):
            self.assertIn(expected, patterns)

    def test_boogu_protects_norm_layers_from_public_recipe(self):
        patterns = self._skip_patterns("boogu")
        self.assertIn("norm1.linear", patterns)
        self.assertIn("norm_out", patterns)

    def test_ltx_aliases_share_the_current_ltxv2_recipe(self):
        expected = self._skip_patterns("ltxv2")
        self.assertEqual(self._skip_patterns("ltx2"), expected)
        self.assertEqual(self._skip_patterns("ltx2_3"), expected)

        for protected_layer in (
            "scale_shift_table",
            "text_embedding_projection",
            "audio_embeddings_connector",
            "video_embeddings_connector",
            "adaln_single",
            "audio_patchify_proj",
            "audio_proj_out",
            "transformer_blocks.0.",
            "transformer_blocks.1.",
            "transformer_blocks.46.",
            "transformer_blocks.47.",
            "to_gate_logits",
            "audio_vae",
            "vae.decoder",
            "vae.encoder",
            "vocoder",
        ):
            self.assertIn(protected_layer, expected)

    def test_generic_text_is_explicitly_input_scale_only(self):
        self.assertNotIn("exclude", MODEL_FILTERS["generic_text"])
        self.assertNotIn("highprec", MODEL_FILTERS["generic_text"])
        self.assertIn("input scales", MODEL_FILTERS["generic_text"]["help"])


if __name__ == "__main__":
    unittest.main()
