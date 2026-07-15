"""Central routing decisions for layer and converter quantization paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Pattern

from .config.layer_config import get_layer_settings
from .constants import MODEL_FILTERS, T5XXL_REMOVE_KEY_NAMES

RouteAction = Literal["quantize", "skip", "remove"]
RouteSource = Literal["primary", "custom", "fallback", "layer_config", "exclusion", "model_filter"]
QuantizationMode = Literal["simple", "learned"]


@dataclass(frozen=True)
class LayerRoute:
    """Complete internal decision for processing one weight layer."""

    action: RouteAction
    source: RouteSource
    target_format: Optional[str]
    mode: QuantizationMode
    optimizer: str
    layer_settings: Optional[Dict[str, Any]] = None
    exclusion_reason: str = ""

    @property
    def uses_layer_config(self) -> bool:
        return self.source == "layer_config"

    @property
    def uses_custom(self) -> bool:
        return self.source == "custom"

    @property
    def uses_fallback(self) -> bool:
        return self.source == "fallback"


@dataclass(frozen=True)
class ConversionRoute:
    """Resolved converter entry method and learned mode."""

    target_format: str
    scaling_mode: str
    mode: QuantizationMode
    optimizer: str
    method_name: str


def _config_target_format(format_name: str) -> str:
    """Preserve the existing layer-config to converter mapping."""
    if format_name.startswith("float8"):
        return "fp8"
    if format_name.startswith("int8"):
        return "int8"
    return "fp8"


def _mode(simple: bool) -> QuantizationMode:
    return "simple" if simple else "learned"


def resolve_layer_route(
    key: str,
    *,
    target_format: str,
    optimizer: str,
    primary_simple: bool,
    filter_flags: Dict[str, bool],
    layer_config: Optional[Dict[str, Any]] = None,
    layer_config_fullmatch: bool = False,
    custom_pattern: Optional[Pattern[str]] = None,
    custom_type: Optional[str] = None,
    custom_simple: bool = False,
    exclude_pattern: Optional[Pattern[str]] = None,
    fallback: Optional[str] = None,
    fallback_simple: bool = False,
) -> LayerRoute:
    """Resolve current routing precedence into one immutable decision."""
    if filter_flags.get("t5xxl") and any(name in key for name in T5XXL_REMOVE_KEY_NAMES):
        return LayerRoute("remove", "model_filter", None, _mode(primary_simple), optimizer)

    if layer_config:
        settings = get_layer_settings(key, layer_config, fullmatch=layer_config_fullmatch)
        if settings:
            if settings.get("skip", False):
                return LayerRoute("skip", "layer_config", None, _mode(primary_simple), optimizer, settings)
            simple = primary_simple or bool(settings.get("simple", False))
            return LayerRoute(
                "quantize",
                "layer_config",
                _config_target_format(settings["format"]),
                _mode(simple),
                optimizer,
                settings,
            )

    if custom_pattern and custom_pattern.search(key):
        return LayerRoute("quantize", "custom", custom_type, _mode(custom_simple), optimizer)

    exclusion_reason = ""
    exclusion_source: RouteSource = "exclusion"
    if exclude_pattern and exclude_pattern.search(key):
        exclusion_reason = "regex exclusion (--exclude-layers)"

    for filter_name, is_active in filter_flags.items():
        if not is_active:
            continue
        config = MODEL_FILTERS[filter_name]
        skip_patterns = config.get("exclude", []) + config.get("highprec", [])
        if skip_patterns and any(name in key for name in skip_patterns):
            exclusion_reason = f"{filter_name} skip"
            exclusion_source = "model_filter"
            break

    if exclusion_reason:
        if fallback:
            return LayerRoute(
                "quantize",
                "fallback",
                fallback,
                _mode(fallback_simple),
                optimizer,
                exclusion_reason=exclusion_reason,
            )
        return LayerRoute(
            "skip",
            exclusion_source,
            None,
            _mode(primary_simple),
            optimizer,
            exclusion_reason=exclusion_reason,
        )

    return LayerRoute("quantize", "primary", target_format, _mode(primary_simple), optimizer)


def resolve_conversion_route(
    target_format: str,
    scaling_mode: str,
    *,
    no_learned_rounding: bool,
    optimizer: str,
) -> ConversionRoute:
    """Resolve format, scaling, simple mode, and learned optimizer once."""
    if target_format == "int8":
        method_name = "_convert_int8_tensorwise" if scaling_mode in ("tensor", "row") else "_convert_int8"
    elif scaling_mode == "row":
        method_name = "_convert_fp8_rowwise"
    elif scaling_mode in ("block", "block2d"):
        method_name = "_convert_fp8_block2d"
    else:
        method_name = "_convert_fp8"

    return ConversionRoute(
        target_format=target_format,
        scaling_mode=scaling_mode,
        mode=_mode(no_learned_rounding),
        optimizer=optimizer,
        method_name=method_name,
    )
