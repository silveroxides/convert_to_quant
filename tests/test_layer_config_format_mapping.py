"""
Unit tests for mapping a --layer-config entry's "format" string to the
internal layer_format used to select a converter.

Regression test for: "format": "nvfp4" (or "mxfp8") in --layer-config
silently becoming "fp8" because the fmt -> layer_format narrowing switch
had no branch for those formats. See BUG_layer_config_nvfp4.md.
"""

import unittest

from convert_to_quant.formats.fp8_conversion import resolve_layer_config_format


class TestResolveLayerConfigFormat(unittest.TestCase):

    def test_nvfp4_maps_to_nvfp4(self):
        self.assertEqual(resolve_layer_config_format("nvfp4"), "nvfp4")

    def test_mxfp8_maps_to_mxfp8(self):
        self.assertEqual(resolve_layer_config_format("mxfp8"), "mxfp8")

    def test_float8_variants_map_to_fp8(self):
        self.assertEqual(resolve_layer_config_format("float8_e4m3fn"), "fp8")

    def test_int8_variants_map_to_int8(self):
        self.assertEqual(resolve_layer_config_format("int8_symmetric"), "int8")

    def test_unknown_format_falls_back_to_fp8(self):
        self.assertEqual(resolve_layer_config_format("bogus"), "fp8")


if __name__ == "__main__":
    unittest.main()
