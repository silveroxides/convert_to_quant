"""Output dtype policy for quantized and unquantized model weights."""

import re
from typing import Dict, Iterable, Optional, Pattern, Union

import torch

from ..constants import get_model_filter_config, resolve_model_filter_patterns

OUTPUT_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}

_CASTABLE_WEIGHT_DTYPES = {
    torch.bfloat16,
    torch.float16,
    torch.float32,
    torch.float64,
}


def resolve_output_dtype(value: Union[str, torch.dtype]) -> torch.dtype:
    if isinstance(value, torch.dtype):
        if value not in OUTPUT_DTYPES.values():
            raise ValueError("output_dtype must be torch.bfloat16 or torch.float16")
        return value

    try:
        return OUTPUT_DTYPES[value]
    except (KeyError, TypeError) as exc:
        raise ValueError("output_dtype must be 'bfloat16' or 'float16'") from exc


def compile_preserve_layers(pattern: Optional[str]) -> Optional[Pattern[str]]:
    if not pattern:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid preserve-layers regex '{pattern}': {exc}") from exc


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _mapped_preservation_dtype(
    key: str,
    source_dtype: torch.dtype,
    filter_flags: Optional[Dict[str, bool]],
    resolved_filter_patterns: Dict[str, tuple],
) -> Optional[torch.dtype]:
    if not filter_flags:
        return None

    matched_dtypes = []
    for filter_name, enabled in filter_flags.items():
        if not enabled:
            continue
        if filter_name not in resolved_filter_patterns:
            continue
        config = get_model_filter_config(filter_name)

        exclusion_patterns = resolved_filter_patterns[filter_name]
        if not any(pattern in key for pattern in exclusion_patterns):
            continue

        for dtype_name, patterns in config.get("preserve_dtype", {}).items():
            mapped_dtype = OUTPUT_DTYPES.get(dtype_name)
            if dtype_name == "float32":
                mapped_dtype = torch.float32
            if mapped_dtype is None:
                raise ValueError(f"Unknown preserve_dtype '{dtype_name}' in model filter '{filter_name}'")
            for pattern in patterns:
                if re.search(pattern, key):
                    matched_dtypes.append(mapped_dtype)
                    break

    if not matched_dtypes:
        return None

    requested_dtype = max(matched_dtypes, key=_dtype_size)
    if _dtype_size(source_dtype) < _dtype_size(requested_dtype):
        return source_dtype
    return requested_dtype


def cast_unquantized_weights(
    tensors: Dict[str, torch.Tensor],
    quantized_weight_keys: Iterable[str],
    output_dtype: Union[str, torch.dtype],
    preserve_pattern: Optional[Pattern[str]],
    filter_flags: Optional[Dict[str, bool]],
) -> int:
    target_dtype = resolve_output_dtype(output_dtype)
    quantized_keys = set(quantized_weight_keys)
    cast_count = 0
    resolved_filter_patterns = resolve_model_filter_patterns(filter_flags or {}, tensors.keys())

    for key, tensor in tensors.items():
        if not key.endswith(".weight") or key in quantized_keys or tensor.ndim != 2:
            continue
        if tensor.dtype not in _CASTABLE_WEIGHT_DTYPES:
            continue

        if preserve_pattern and preserve_pattern.search(key):
            requested_dtype = tensor.dtype
        else:
            requested_dtype = _mapped_preservation_dtype(
                key, tensor.dtype, filter_flags, resolved_filter_patterns
            ) or target_dtype

        if tensor.dtype != requested_dtype:
            tensors[key] = tensor.to(dtype=requested_dtype)
            cast_count += 1

    return cast_count
