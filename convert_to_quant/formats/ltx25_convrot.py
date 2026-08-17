"""Streaming LTX-2.5 22B HQ INT8 ConvRot checkpoint conversion."""

from __future__ import annotations

import collections
import json
import math
import os
import re
import struct
import time
from typing import Dict, Optional, Tuple

import torch
from safetensors import safe_open

from ..utils.convrot import build_hadamard, rotate_weight


GROUP_SIZES = (16, 64, 256)
RECIPES = ("full", "compact", "video")
VARIANTS = ("auto", "dev", "distilled")
EXPECTED_ELIGIBLE_LAYERS = 1440
EXPECTED_BLOCKS = set(range(48))

_TRANSFORMER_BLOCK_RE = re.compile(
    r"^model[.]diffusion_model[.]transformer_blocks[.](\d+)[.]"
)
_EXCLUDED_SEGMENT_RE = re.compile(
    r"scale_shift|rope|rotary|rel_pos|pos_?embed|embedder|"
    r"gate_logits|router|routing|logit|temperature|"
    r"(?:^|_)time|temb|t_emb|guidance|register|refiner_blocks|adapter|"
    r"(?:^|_)(?:final|head|proj_out|out_layer)(?:_|$)"
)


def _best_supported_group_size(width: int) -> Optional[int]:
    return next((size for size in reversed(GROUP_SIZES) if width % size == 0), None)


def _block_index(name: str) -> Optional[int]:
    match = _TRANSFORMER_BLOCK_RE.match(name)
    return int(match.group(1)) if match else None


def is_ltx25_quantizable(name: str, shape: Tuple[int, ...], min_gemm: int = 256) -> bool:
    """Return whether a source weight belongs to the validated LTX-2.5 recipe."""
    if len(shape) != 2:
        return False
    rows, columns = shape
    if rows < 8 or _best_supported_group_size(columns) is None:
        return False
    segments = name.split(".")
    if not any(segment.isdigit() for segment in segments[:-1]):
        return False
    if any(_EXCLUDED_SEGMENT_RE.search(segment) for segment in segments):
        return False
    return not min_gemm or min(shape) >= min_gemm


def preserve_ltx25_bf16(name: str, variant: str, recipe: str = "video") -> bool:
    """Return whether a validated sensitivity-selected projection stays BF16."""
    if recipe == "full":
        return False

    block = _block_index(name)
    if block is None:
        return False

    compact = (
        (".audio_attn1." in name and name.endswith(".to_out.0"))
        or (
            ".attn1." in name
            and name.endswith(".to_out.0")
            and block in {*range(12, 20), *range(24, 28)}
        )
    )
    if compact or recipe == "compact":
        return compact

    if ".video_to_audio_attn." in name and name.endswith(".to_v") and block in range(4, 8):
        return True
    return (
        variant == "dev"
        and ".video_to_audio_attn." in name
        and name.endswith(".to_q")
        and block in range(20, 24)
    )


def _detect_variant(source_path: str, metadata: Dict[str, str], requested: str) -> str:
    if requested not in VARIANTS:
        raise ValueError(f"Unknown LTX-2.5 variant '{requested}'. Choose from: {', '.join(VARIANTS)}")
    if requested != "auto":
        return requested

    searchable = " ".join(
        [os.path.basename(source_path), *[str(value) for value in metadata.values() if len(str(value)) < 1000]]
    ).lower()
    if "distill" in searchable:
        return "distilled"
    if re.search(r"(?:^|[-_. ])dev(?:[-_. ]|$)", searchable):
        return "dev"
    raise ValueError(
        "Could not determine whether the LTX-2.5 checkpoint is dev or distilled. "
        "Include 'dev' or 'distilled' in the filename, or pass --ltx25-variant explicitly."
    )


def _config_tensor(group_size: int) -> torch.Tensor:
    config = {
        "format": "int8_tensorwise",
        "convrot": True,
        "convrot_groupsize": group_size,
    }
    encoded = json.dumps(config).encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("LTX-2.5 comfy_quant configuration exceeded its fixed storage size")
    return torch.tensor(list(encoded.ljust(72, b" ")), dtype=torch.uint8)


def _search_scales(rotated: torch.Tensor):
    absmax = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    best_alpha = torch.ones_like(absmax)
    best_mse = torch.full_like(absmax, float("inf"))

    def search(deltas: torch.Tensor, centered: bool) -> None:
        nonlocal best_alpha, best_mse
        center = best_alpha if centered else torch.zeros_like(best_alpha)
        for delta in deltas:
            alpha = (center + delta).clamp(0.5, 1.0)
            scale = (absmax * alpha / 127.0).clamp_min(1e-30)
            quantized = (rotated / scale).round().clamp(-127, 127)
            mse = (quantized * scale - rotated).square().mean(dim=1, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_alpha = torch.where(better, alpha, best_alpha)

    search(torch.linspace(0.60, 1.00, 41, device=rotated.device), centered=False)
    search(torch.linspace(-0.012, 0.012, 25, device=rotated.device), centered=True)
    search(torch.linspace(-0.0010, 0.0010, 21, device=rotated.device), centered=True)
    scale = (absmax * best_alpha / 127.0).clamp_min(1e-30)
    quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    return quantized, scale, best_mse


@torch.no_grad()
def _quantize_best(weight: torch.Tensor, device: torch.device):
    weight = weight.to(device=device, dtype=torch.float32)
    weight_energy = weight.square().sum().clamp_min(1e-30)
    candidates = []
    for group_size in GROUP_SIZES:
        if weight.shape[1] % group_size:
            continue
        hadamard = build_hadamard(group_size, device=device, dtype=torch.float32)
        rotated = rotate_weight(weight, hadamard, group_size)
        quantized, scale, row_mse = _search_scales(rotated)
        squared_error = (row_mse.sum() * rotated.shape[1]).item()
        relative_error = math.sqrt(squared_error / weight_energy.item()) * 100.0
        candidates.append((relative_error, group_size, quantized.cpu(), scale.cpu()))
        del rotated, quantized, scale, row_mse
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if not candidates:
        raise ValueError(f"No supported ConvRot group size divides input width {weight.shape[1]}")
    candidates.sort(key=lambda item: item[0])
    return candidates[0]


@torch.no_grad()
def _quantize_absmax(weight: torch.Tensor, device: torch.device, group_size: int = 256):
    weight = weight.to(device=device, dtype=torch.float32)
    hadamard = build_hadamard(group_size, device=device, dtype=torch.float32)
    rotated = rotate_weight(weight, hadamard, group_size)
    scale = (rotated.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-30)
    quantized = (rotated / scale).round().clamp(-127, 127).to(torch.int8)
    squared_error = (quantized.float() * scale - rotated).square().sum()
    relative_error = (squared_error.sqrt() / weight.norm()).item() * 100.0
    return relative_error, group_size, quantized.cpu(), scale.cpu()


def _is_main_video_ff(name: str) -> bool:
    return _block_index(name) is not None and ".ff." in name and ".audio_ff." not in name


def _tensor_nbytes(dtype: str, shape) -> int:
    bits = {
        "BOOL": 8,
        "U8": 8,
        "I8": 8,
        "F8_E4M3": 8,
        "F8_E5M2": 8,
        "I16": 16,
        "U16": 16,
        "F16": 16,
        "BF16": 16,
        "I32": 32,
        "U32": 32,
        "F32": 32,
        "I64": 64,
        "U64": 64,
        "F64": 64,
    }.get(dtype)
    if bits is None:
        raise ValueError(f"Unsupported safetensors dtype in LTX-2.5 source: {dtype}")
    return math.prod(shape) * bits // 8


def _write_tensor(handle, tensor: torch.Tensor) -> None:
    handle.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def _validate_architecture(metadata, eligible, preserved, variant, recipe) -> None:
    version = str(metadata.get("model_version", ""))
    if not version.startswith("2.5"):
        raise ValueError(f"Expected an LTX-2.5 checkpoint, but model_version is '{version or 'missing'}'")
    if len(eligible) != EXPECTED_ELIGIBLE_LAYERS:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_LAYERS} eligible LTX-2.5 22B linears, found {len(eligible)}. "
            "The checkpoint layout does not match the validated recipe."
        )
    blocks = {_block_index(name) for name in eligible}
    blocks.discard(None)
    if blocks != EXPECTED_BLOCKS:
        raise ValueError("Expected LTX-2.5 transformer blocks 0-47; the checkpoint layout is incompatible")
    expected_preserved = {"full": 0, "compact": 60, "video": 68 if variant == "dev" else 64}[recipe]
    if len(preserved) != expected_preserved:
        raise ValueError(
            f"Expected {expected_preserved} BF16 layers for the {variant} {recipe} recipe, found {len(preserved)}"
        )


def inspect_ltx25_checkpoint(
    source_path: str,
    variant: str = "auto",
    recipe: str = "video",
    min_gemm: int = 256,
    strict_architecture: bool = True,
):
    """Inspect a source header and return the exact quantized/BF16 layer plan."""
    if recipe not in RECIPES:
        raise ValueError(f"Unknown LTX-2.5 recipe '{recipe}'. Choose from: {', '.join(RECIPES)}")
    source_path = os.path.abspath(source_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)

    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        keys = list(source.keys())
        if any(key.endswith((".comfy_quant", ".weight_scale", ".scale_weight")) for key in keys):
            raise ValueError("LTX-2.5 HQ conversion requires the original BF16 checkpoint, not a quantized checkpoint")
        resolved_variant = _detect_variant(source_path, metadata, variant)
        eligible = {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            shape = tuple(source.get_slice(key).get_shape())
            name = key[: -len(".weight")]
            if is_ltx25_quantizable(name, shape, min_gemm):
                eligible[name] = shape
        preserved = {
            name for name in eligible if preserve_ltx25_bf16(name, resolved_variant, recipe)
        }
        if strict_architecture:
            _validate_architecture(metadata, eligible, preserved, resolved_variant, recipe)

    return {
        "source": source_path,
        "variant": resolved_variant,
        "recipe": recipe,
        "eligible": eligible,
        "preserved": preserved,
        "quantized": {name: shape for name, shape in eligible.items() if name not in preserved},
        "metadata": metadata,
    }


def _output_plan(source, quantized_names):
    plan = {key: ("source", key) for key in source.keys()}
    config_length = len(_config_tensor(256))
    for name, shape in quantized_names.items():
        del plan[f"{name}.weight"]
        plan[f"{name}.comfy_quant"] = ("quant_config", name, "U8", [config_length])
        plan[f"{name}.weight"] = ("quant_weight", name, "I8", list(shape))
        plan[f"{name}.weight_scale"] = ("quant_scale", name, "F32", [shape[0], 1])
    return plan


def _metadata_for_plan(original, plan):
    metadata = dict(original)
    policies = []
    if plan["recipe"] != "full":
        policies.extend(["audio_self_out", "video_self_out_blocks_12-19_24-27"])
    if plan["recipe"] == "video":
        policies.append("video_to_audio_v_blocks_4-7")
        if plan["variant"] == "dev":
            policies.append("video_to_audio_q_blocks_20-23")
    metadata.update(
        {
            "convrot_hq_method": (
                "per-row multistage MSE clipping and per-layer group-size search; "
                "absmax gs256 for main video FF"
            ),
            "convrot_hq_modified": "Converted from BF16 to a mixed BF16/INT8 ConvRot derivative",
            "convrot_hq_policy": f"ltx25_{plan['recipe']}_{plan['variant']}",
            "convrot_hq_variant": plan["variant"],
            "convrot_hq_recipe": plan["recipe"],
            "convrot_hq_bf16_policy": "+".join(policies) if policies else "none",
            "convrot_hq_eligible_layers": str(len(plan["eligible"])),
            "convrot_hq_int8_layers": str(len(plan["quantized"])),
            "convrot_hq_restored_bf16_layers": str(len(plan["preserved"])),
        }
    )
    return metadata


def _make_header(source, output_plan, metadata):
    header = {"__metadata__": metadata}
    offset = 0
    for key in sorted(output_plan):
        item = output_plan[key]
        if item[0] == "source":
            tensor = source.get_slice(item[1])
            dtype = tensor.get_dtype()
            shape = tensor.get_shape()
        else:
            dtype = item[2]
            shape = item[3]
        size = _tensor_nbytes(dtype, shape)
        header[key] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    return header


def default_ltx25_output_path(source_path: str) -> str:
    base, _ = os.path.splitext(source_path)
    base = re.sub(r"(?:[-_]transformer)?[-_]bf16$", "-transformer", base, flags=re.IGNORECASE)
    return f"{base}-comfy-int8-convrot-hq-22gb-video.safetensors"


def convert_ltx25_convrot_hq(
    source_path: str,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    variant: str = "auto",
    recipe: str = "video",
    min_gemm: int = 256,
    device: Optional[str] = None,
    strict_architecture: bool = True,
):
    """Create the validated mixed BF16/INT8 LTX-2.5 checkpoint in one pass."""
    source_path = os.path.abspath(source_path)
    output_path = os.path.abspath(output_path or default_ltx25_output_path(source_path))
    report_path = os.path.abspath(report_path or f"{os.path.splitext(output_path)[0]}_report.json")
    if os.path.abspath(source_path) == output_path:
        raise ValueError("Output file cannot be the same as the LTX-2.5 BF16 source")
    plan = inspect_ltx25_checkpoint(
        source_path,
        variant=variant,
        recipe=recipe,
        min_gemm=min_gemm,
        strict_architecture=strict_architecture,
    )
    quant_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if quant_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for LTX-2.5 conversion, but CUDA is not available")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    partial_path = f"{output_path}.partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    started = time.time()
    results = []
    group_counts = collections.Counter()
    with safe_open(source_path, framework="pt", device="cpu") as source:
        output_plan = _output_plan(source, plan["quantized"])
        metadata = _metadata_for_plan(plan["metadata"], plan)
        header = _make_header(source, output_plan, metadata)
        header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        header_length = (len(header_bytes) + 7) & ~7

        print(
            "LTX-2.5 22B HQ INT8 ConvRot\n"
            f"SOURCE {source_path}\nOUTPUT {output_path}\n"
            f"VARIANT {plan['variant']} | RECIPE {recipe} | DEVICE {quant_device}\n"
            f"LAYERS {len(plan['quantized'])} INT8 ConvRot + {len(plan['preserved'])} sensitivity-selected BF16",
            flush=True,
        )
        cache_name = None
        cache = None
        try:
            with open(partial_path, "wb") as handle:
                handle.write(struct.pack("<Q", header_length))
                handle.write(header_bytes)
                handle.write(b" " * (header_length - len(header_bytes)))
                for key in sorted(output_plan):
                    item = output_plan[key]
                    if item[0] == "source":
                        _write_tensor(handle, source.get_tensor(item[1]))
                        continue

                    name = item[1]
                    if cache_name != name:
                        weight = source.get_tensor(f"{name}.weight")
                        layer_started = time.time()
                        if _is_main_video_ff(name):
                            relative_error, group_size, quantized, scale = _quantize_absmax(
                                weight, quant_device
                            )
                        else:
                            relative_error, group_size, quantized, scale = _quantize_best(
                                weight, quant_device
                            )
                        cache_name = name
                        cache = {
                            "quant_config": _config_tensor(group_size),
                            "quant_weight": quantized,
                            "quant_scale": scale,
                        }
                        group_counts[group_size] += 1
                        results.append(
                            {
                                "layer": name,
                                "shape": list(weight.shape),
                                "group_size": group_size,
                                "relative_weight_error_pct": relative_error,
                            }
                        )
                        print(
                            f"  {len(results):4}/{len(plan['quantized'])} gs{group_size:<3} "
                            f"err={relative_error:.4f}% {time.time() - layer_started:.2f}s {name}",
                            flush=True,
                        )
                    _write_tensor(handle, cache[item[0]])
                    if item[0] == "quant_scale":
                        cache_name = None
                        cache = None
            os.replace(partial_path, output_path)
        except BaseException:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise

    errors = [item["relative_weight_error_pct"] for item in results]
    report = {
        "source": source_path,
        "output": output_path,
        "variant": plan["variant"],
        "recipe": recipe,
        "method": metadata["convrot_hq_method"],
        "eligible_layers": len(plan["eligible"]),
        "int8_layers": len(results),
        "bf16_layers": len(plan["preserved"]),
        "bf16_layer_names": sorted(plan["preserved"]),
        "group_sizes": dict(sorted(group_counts.items())),
        "mean_relative_weight_error_pct": sum(errors) / len(errors),
        "max_relative_weight_error_pct": max(errors),
        "output_bytes": os.path.getsize(output_path),
        "conversion_seconds": time.time() - started,
        "layers": results,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"DONE {output_path} ({os.path.getsize(output_path) / 1e9:.3f} GB)\nREPORT {report_path}",
        flush=True,
    )
    return report
