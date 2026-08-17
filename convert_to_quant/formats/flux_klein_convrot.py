"""Measured mixed-precision ConvRot compiler for FLUX.2 Klein 9B.

The compiler writes native ComfyUI ``int8_tensorwise`` W8A8 and packed
``asym_w4a8_int8`` W4A8 layouts.  The legacy ``convrot_w4a4`` layout remains
available as an explicit research option.  Routing is deliberately separate
from quantization: a versioned JSON policy decides whether each eligible
projection stays BF16 or becomes W8/W4 ConvRot.  This keeps calibration
results reproducible and the GUI presets auditable.
"""

from __future__ import annotations

import collections
import json
import math
import os
import re
import struct
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
from safetensors import safe_open

from ..utils.convrot import build_hadamard, rotate_weight


SCHEMA_VERSION = 1
GROUP_SIZE = 256
DEFAULT_INT4_LAYOUT = "w4a8"
DEFAULT_INT4_GROUP_SIZE = 16
VALID_INT4_LAYOUTS = frozenset({"w4a4", "w4a8"})
VALID_INT4_GROUP_SIZES = frozenset({4, 8, 16, 32, 64})
VALID_MODES = frozenset({"bf16", "int8", "int4"})
EXPECTED_DOUBLE_BLOCKS = frozenset(range(8))
EXPECTED_SINGLE_BLOCKS = frozenset(range(24))
EXPECTED_ELIGIBLE_LAYERS = 112

_DOUBLE_LAYER_RE = re.compile(
    r"^double_blocks[.](\d+)[.]"
    r"(?:img|txt)_(?:attn[.](?:qkv|proj)|mlp[.](?:0|2))$"
)
_SINGLE_LAYER_RE = re.compile(r"^single_blocks[.](\d+)[.]linear[12]$")


@dataclass(frozen=True)
class FluxKleinPolicy:
    """Resolved exact layer routing policy."""

    name: str
    default_mode: str
    layers: Dict[str, str]
    metadata: Dict[str, object]
    source_path: Optional[str] = None

    def mode_for(self, layer: str) -> str:
        return self.layers.get(layer, self.default_mode)


def _layer_name(weight_key: str) -> str:
    return weight_key[: -len(".weight")]


def is_flux_klein_projection(name: str, shape: Tuple[int, ...]) -> bool:
    """Return whether ``name`` is one of Klein 9B's transformer projections."""
    if len(shape) != 2 or shape[1] % GROUP_SIZE:
        return False
    return bool(_DOUBLE_LAYER_RE.fullmatch(name) or _SINGLE_LAYER_RE.fullmatch(name))


def _validate_mode(mode: object, context: str) -> str:
    if not isinstance(mode, str) or mode not in VALID_MODES:
        choices = ", ".join(sorted(VALID_MODES))
        raise ValueError(f"Invalid FLUX Klein mode for {context}: {mode!r}. Choose from: {choices}")
    return mode


def load_flux_klein_policy(path: Optional[str], default_mode: str) -> FluxKleinPolicy:
    """Load a policy or create an all-quantized research baseline."""
    default_mode = _validate_mode(default_mode, "default")
    if not path:
        return FluxKleinPolicy(
            name=f"all_{default_mode}_research_baseline",
            default_mode=default_mode,
            layers={},
            metadata={"purpose": "research_baseline"},
        )

    absolute = os.path.abspath(os.path.expanduser(path))
    with open(absolute, encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("FLUX Klein policy must be a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported FLUX Klein policy schema {raw.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("FLUX Klein policy requires a non-empty 'name'")
    policy_default = _validate_mode(raw.get("default_mode", default_mode), "default_mode")
    raw_layers = raw.get("layers", {})
    if not isinstance(raw_layers, dict):
        raise ValueError("FLUX Klein policy 'layers' must be an object")
    layers = {
        str(layer): _validate_mode(mode, f"layer {layer!r}")
        for layer, mode in raw_layers.items()
    }
    metadata = raw.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("FLUX Klein policy 'metadata' must be an object")
    return FluxKleinPolicy(name.strip(), policy_default, layers, metadata, absolute)


def _validate_architecture(eligible: Dict[str, Tuple[int, ...]], strict: bool) -> None:
    if not strict:
        return
    if len(eligible) != EXPECTED_ELIGIBLE_LAYERS:
        raise ValueError(
            f"Expected {EXPECTED_ELIGIBLE_LAYERS} FLUX.2 Klein 9B projections, "
            f"found {len(eligible)}"
        )
    double_blocks = {
        int(match.group(1))
        for name in eligible
        if (match := _DOUBLE_LAYER_RE.fullmatch(name))
    }
    single_blocks = {
        int(match.group(1))
        for name in eligible
        if (match := _SINGLE_LAYER_RE.fullmatch(name))
    }
    if double_blocks != EXPECTED_DOUBLE_BLOCKS or single_blocks != EXPECTED_SINGLE_BLOCKS:
        raise ValueError(
            "Expected FLUX.2 Klein 9B double blocks 0-7 and single blocks 0-23; "
            "the checkpoint layout is incompatible"
        )


def inspect_flux_klein_checkpoint(
    source_path: str,
    policy_path: Optional[str] = None,
    default_mode: str = "int8",
    strict_architecture: bool = True,
):
    """Inspect the source header and resolve its exact precision plan."""
    source_path = os.path.abspath(os.path.expanduser(source_path))
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)
    policy = load_flux_klein_policy(policy_path, default_mode)

    with safe_open(source_path, framework="pt", device="cpu") as source:
        metadata = dict(source.metadata() or {})
        keys = list(source.keys())
        if any(key.endswith((".comfy_quant", ".weight_scale", ".scale_weight")) for key in keys):
            raise ValueError(
                "FLUX Klein ConvRot compilation requires the original BF16 checkpoint, "
                "not a quantized checkpoint"
            )
        eligible = {}
        for key in keys:
            if not key.endswith(".weight"):
                continue
            shape = tuple(source.get_slice(key).get_shape())
            name = _layer_name(key)
            if is_flux_klein_projection(name, shape):
                eligible[name] = shape

    _validate_architecture(eligible, strict_architecture)
    unknown = sorted(set(policy.layers) - set(eligible))
    if unknown:
        preview = ", ".join(unknown[:5])
        suffix = " ..." if len(unknown) > 5 else ""
        raise ValueError(f"Policy references {len(unknown)} unknown layers: {preview}{suffix}")

    routes = {name: policy.mode_for(name) for name in eligible}
    counts = collections.Counter(routes.values())
    return {
        "source": source_path,
        "metadata": metadata,
        "eligible": eligible,
        "routes": routes,
        "counts": dict(counts),
        "policy": policy,
    }


def _pack_signed_int4(values: torch.Tensor) -> torch.Tensor:
    """Pack signed nibbles row-major, matching comfy-kitchen storage."""
    if values.shape[-1] % 2:
        raise ValueError("INT4 packing requires an even input width")
    lo = values[..., 0::2].to(torch.int32) & 0x0F
    hi = values[..., 1::2].to(torch.int32) & 0x0F
    return (lo | (hi << 4)).to(torch.int8)


def _alpha_schedule(qmax: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if qmax == 127:
        coarse = torch.linspace(0.88, 1.0, 13)
        fine = torch.linspace(-0.010, 0.010, 9)
    else:
        coarse = torch.linspace(0.55, 1.0, 19)
        fine = torch.linspace(-0.025, 0.025, 11)
    return coarse, fine


@torch.no_grad()
def _mse_clipped_symmetric_quantize(
    rotated: torch.Tensor,
    qmax: int,
    row_chunk: int = 256,
):
    """Per-row two-stage clipping search with bounded working memory."""
    coarse, fine = _alpha_schedule(qmax)
    q_chunks = []
    scale_chunks = []
    squared_error = 0.0
    for start in range(0, rotated.shape[0], row_chunk):
        values = rotated[start : start + row_chunk]
        absmax = values.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
        best_alpha = torch.ones_like(absmax)
        best_mse = torch.full_like(absmax, float("inf"))

        for alpha_value in coarse.to(device=values.device, dtype=values.dtype):
            alpha = torch.full_like(absmax, alpha_value)
            scale = absmax * alpha / qmax
            restored = (values / scale).round().clamp(-qmax, qmax) * scale
            mse = (restored - values).square().mean(dim=1, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_alpha = torch.where(better, alpha, best_alpha)

        for delta in fine.to(device=values.device, dtype=values.dtype):
            alpha = (best_alpha + delta).clamp(0.5, 1.0)
            scale = absmax * alpha / qmax
            restored = (values / scale).round().clamp(-qmax, qmax) * scale
            mse = (restored - values).square().mean(dim=1, keepdim=True)
            better = mse < best_mse
            best_mse = torch.where(better, mse, best_mse)
            best_alpha = torch.where(better, alpha, best_alpha)

        scale = (absmax * best_alpha / qmax).clamp_min(1e-30)
        quantized = (values / scale).round().clamp(-qmax, qmax).to(torch.int8)
        q_chunks.append(quantized.cpu())
        scale_chunks.append(scale.float().cpu())
        squared_error += float(best_mse.sum().item() * values.shape[1])
    return torch.cat(q_chunks), torch.cat(scale_chunks), squared_error


@torch.no_grad()
def _quantize_projection(
    weight: torch.Tensor,
    mode: str,
    device: torch.device,
    scale_search: str,
    int4_layout: str,
    int4_group_size: int,
):
    if mode == "int4" and int4_layout == "w4a8":
        try:
            from comfy_kitchen.tensor import QuantizedTensor
        except ImportError as exc:
            raise RuntimeError(
                "The grouped W4A8 ConvRot HQ route requires comfy-kitchen >= 0.2.31"
            ) from exc

        work = weight.to(device=device).contiguous()
        original_energy = float(work.float().square().sum().item())
        quantized = QuantizedTensor.from_float(
            work,
            "AsymW4A8Int8Layout",
            group_size=int4_group_size,
            convrot_groupsize=GROUP_SIZE,
            symmetric=True,
            scale_dtype=torch.float8_e4m3fn,
            codebook=True,
        )
        restored = quantized.dequantize()
        squared_error = float((restored.float() - work.float()).square().sum().item())
        state = {
            key.removeprefix("weight"): value.detach().cpu().contiguous()
            for key, value in quantized.state_dict("weight").items()
        }
        del restored, quantized, work
        relative_error = math.sqrt(squared_error / max(original_energy, 1e-30)) * 100.0
        return {
            "quant_weight": state[""],
            "quant_s_rel": state["_s_rel"],
            "quant_s_channel": state["_s_channel"],
            "quant_codebook": state["_codebook"],
        }, relative_error

    original_energy = float(weight.float().square().sum().item())
    work = weight.to(device=device, dtype=torch.float32)
    hadamard = build_hadamard(GROUP_SIZE, device=device, dtype=torch.float32)
    rotated = rotate_weight(work, hadamard, GROUP_SIZE)
    del work
    qmax = 127 if mode == "int8" else 7

    if scale_search == "mse":
        quantized, scale, squared_error = _mse_clipped_symmetric_quantize(rotated, qmax)
    elif scale_search == "absmax":
        scale_gpu = (rotated.abs().amax(dim=1, keepdim=True) / qmax).clamp_min(1e-30)
        q_gpu = (rotated / scale_gpu).round().clamp(-qmax, qmax).to(torch.int8)
        squared_error = float(
            (q_gpu.float() * scale_gpu - rotated).square().sum().item()
        )
        quantized = q_gpu.cpu()
        scale = scale_gpu.float().cpu()
    else:
        raise ValueError("scale_search must be 'absmax' or 'mse'")
    del rotated

    if mode == "int4":
        quantized = _pack_signed_int4(quantized)
        scale = scale.reshape(-1)
    relative_error = math.sqrt(squared_error / max(original_energy, 1e-30)) * 100.0
    return {
        "quant_weight": quantized.contiguous(),
        "quant_scale": scale.contiguous(),
    }, relative_error


def _quant_config(
    mode: str,
    w4_activation: str,
    int4_layout: str,
    int4_group_size: int,
) -> torch.Tensor:
    if mode == "int8":
        config = {
            "format": "int8_tensorwise",
            "convrot": True,
            "convrot_groupsize": GROUP_SIZE,
        }
    elif mode == "int4":
        if int4_layout == "w4a8":
            config = {
                "format": "asym_w4a8_int8",
                "group_size": int4_group_size,
                "convrot_groupsize": GROUP_SIZE,
            }
        else:
            config = {
                "format": "convrot_w4a4",
                "convrot_groupsize": GROUP_SIZE,
            }
            if w4_activation != "int4":
                config["linear_dtype"] = w4_activation
    else:
        raise ValueError(f"No quant config for mode {mode!r}")
    return torch.tensor(list(json.dumps(config, separators=(",", ":")).encode("utf-8")), dtype=torch.uint8)


def _tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
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
        raise ValueError(f"Unsupported safetensors dtype: {dtype}")
    return math.prod(shape) * bits // 8


def _write_tensor(handle, tensor: torch.Tensor) -> None:
    handle.write(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def _output_plan(
    source,
    routes: Dict[str, str],
    w4_activation: str,
    int4_layout: str,
    int4_group_size: int,
):
    plan = {key: ("source", key) for key in source.keys()}
    for name, mode in routes.items():
        if mode == "bf16":
            continue
        shape = tuple(source.get_slice(f"{name}.weight").get_shape())
        qshape = shape if mode == "int8" else (shape[0], shape[1] // 2)
        config_len = len(
            _quant_config(mode, w4_activation, int4_layout, int4_group_size)
        )
        del plan[f"{name}.weight"]
        plan[f"{name}.comfy_quant"] = ("quant_config", name, "U8", [config_len])
        plan[f"{name}.weight"] = ("quant_weight", name, "I8", list(qshape))
        if mode == "int8" or int4_layout == "w4a4":
            scale_shape = (shape[0], 1) if mode == "int8" else (shape[0],)
            plan[f"{name}.weight_scale"] = (
                "quant_scale", name, "F32", list(scale_shape)
            )
        else:
            plan[f"{name}.weight_s_rel"] = (
                "quant_s_rel", name, "F8_E4M3", [shape[0], shape[1] // int4_group_size]
            )
            plan[f"{name}.weight_s_channel"] = (
                "quant_s_channel", name, "F32", [shape[0]]
            )
            plan[f"{name}.weight_codebook"] = (
                "quant_codebook", name, "F32", [16]
            )
    return plan


def _metadata_for_plan(
    original,
    inspection,
    scale_search,
    w4_activation,
    int4_layout,
    int4_group_size,
):
    policy = inspection["policy"]
    counts = collections.Counter(inspection["routes"].values())
    metadata = dict(original)
    metadata.update(
        {
            "flux_klein_convrot_schema": str(SCHEMA_VERSION),
            "flux_klein_convrot_policy": policy.name,
            "flux_klein_convrot_method": (
                f"regular Hadamard gs{GROUP_SIZE}; "
                + (
                    f"grouped Lloyd-Max W4A8 gs{int4_group_size} with FP8 relative scales"
                    if int4_layout == "w4a8"
                    else f"{scale_search} per-row symmetric W4A4 scaling"
                )
            ),
            "flux_klein_convrot_bf16_layers": str(counts["bf16"]),
            "flux_klein_convrot_int8_layers": str(counts["int8"]),
            "flux_klein_convrot_int4_layers": str(counts["int4"]),
            "flux_klein_convrot_w4_activation": w4_activation,
            "flux_klein_convrot_int4_layout": int4_layout,
            "flux_klein_convrot_int4_group_size": str(int4_group_size),
            "flux_klein_convrot_modified": (
                "Compiled from the original BF16 FLUX.2 Klein checkpoint with an auditable mixed-precision policy"
            ),
        }
    )
    if policy.source_path:
        metadata["flux_klein_convrot_policy_file"] = os.path.basename(policy.source_path)
    for key, value in policy.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            metadata[f"flux_klein_policy_{key}"] = str(value)
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


def default_flux_klein_output_path(source_path: str, default_mode: str) -> str:
    base, _ = os.path.splitext(source_path)
    suffix = "int8" if default_mode == "int8" else "int4"
    return f"{base}-comfy-{suffix}-convrot-hq.safetensors"


def convert_flux_klein_convrot_hq(
    source_path: str,
    output_path: Optional[str] = None,
    report_path: Optional[str] = None,
    policy_path: Optional[str] = None,
    default_mode: str = "int8",
    scale_search: str = "mse",
    w4_activation: str = "int8",
    int4_layout: str = DEFAULT_INT4_LAYOUT,
    int4_group_size: int = DEFAULT_INT4_GROUP_SIZE,
    device: Optional[str] = None,
    strict_architecture: bool = True,
):
    """Compile one policy-selected FLUX.2 Klein ConvRot checkpoint."""
    if w4_activation not in {"int4", "int8"}:
        raise ValueError("w4_activation must be 'int4' or 'int8'")
    if int4_layout not in VALID_INT4_LAYOUTS:
        raise ValueError(
            f"int4_layout must be one of {sorted(VALID_INT4_LAYOUTS)}, got {int4_layout!r}"
        )
    if int4_group_size not in VALID_INT4_GROUP_SIZES:
        raise ValueError(
            f"int4_group_size must be one of {sorted(VALID_INT4_GROUP_SIZES)}, "
            f"got {int4_group_size!r}"
        )
    source_path = os.path.abspath(os.path.expanduser(source_path))
    output_path = os.path.abspath(
        os.path.expanduser(output_path or default_flux_klein_output_path(source_path, default_mode))
    )
    report_path = os.path.abspath(
        os.path.expanduser(report_path or f"{os.path.splitext(output_path)[0]}_report.json")
    )
    if source_path == output_path:
        raise ValueError("Output file cannot be the same as the FLUX Klein BF16 source")

    inspection = inspect_flux_klein_checkpoint(
        source_path,
        policy_path=policy_path,
        default_mode=default_mode,
        strict_architecture=strict_architecture,
    )
    quant_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if quant_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for FLUX Klein conversion, but CUDA is unavailable")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    partial_path = f"{output_path}.partial"
    if os.path.exists(partial_path):
        os.remove(partial_path)

    started = time.time()
    results = []
    with safe_open(source_path, framework="pt", device="cpu") as source:
        output_plan = _output_plan(
            source,
            inspection["routes"],
            w4_activation,
            int4_layout,
            int4_group_size,
        )
        metadata = _metadata_for_plan(
            inspection["metadata"], inspection, scale_search, w4_activation,
            int4_layout, int4_group_size,
        )
        header = _make_header(source, output_plan, metadata)
        header_bytes = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        header_length = (len(header_bytes) + 7) & ~7

        counts = collections.Counter(inspection["routes"].values())
        print(
            "FLUX.2 Klein 9B mixed ConvRot compiler\n"
            f"SOURCE {source_path}\nOUTPUT {output_path}\n"
            f"POLICY {inspection['policy'].name} | DEVICE {quant_device} | SCALE {scale_search}\n"
            f"INT4 {int4_layout} group={int4_group_size} | "
            f"LAYERS {counts['bf16']} BF16 + {counts['int8']} W8 + {counts['int4']} W4",
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
                        mode = inspection["routes"][name]
                        weight = source.get_tensor(f"{name}.weight")
                        layer_started = time.time()
                        quant_tensors, relative_error = _quantize_projection(
                            weight,
                            mode,
                            quant_device,
                            scale_search,
                            int4_layout,
                            int4_group_size,
                        )
                        cache_name = name
                        cache = {
                            "quant_config": _quant_config(
                                mode, w4_activation, int4_layout, int4_group_size
                            ),
                            **quant_tensors,
                        }
                        results.append(
                            {
                                "layer": name,
                                "mode": mode,
                                "shape": list(weight.shape),
                                "convrot_group_size": GROUP_SIZE,
                                "relative_weight_error_pct": relative_error,
                            }
                        )
                        print(
                            f"  {len(results):3}/{counts['int8'] + counts['int4']} "
                            f"{mode:<4} err={relative_error:7.4f}% "
                            f"{time.time() - layer_started:6.2f}s {name}",
                            flush=True,
                        )
                        del weight
                        if quant_device.type == "cuda":
                            torch.cuda.empty_cache()
                    _write_tensor(handle, cache[item[0]])
                    if item[0] in {"quant_scale", "quant_s_rel"}:
                        cache_name = None
                        cache = None
            os.replace(partial_path, output_path)
        except BaseException:
            if os.path.exists(partial_path):
                os.remove(partial_path)
            raise

    errors_by_mode = {}
    for mode in ("int8", "int4"):
        errors = [item["relative_weight_error_pct"] for item in results if item["mode"] == mode]
        if errors:
            errors_by_mode[mode] = {
                "mean_relative_weight_error_pct": sum(errors) / len(errors),
                "max_relative_weight_error_pct": max(errors),
            }
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": source_path,
        "output": output_path,
        "policy": inspection["policy"].name,
        "policy_file": inspection["policy"].source_path,
        "scale_search": scale_search,
        "w4_activation": w4_activation,
        "int4_layout": int4_layout,
        "int4_group_size": int4_group_size,
        "counts": dict(collections.Counter(inspection["routes"].values())),
        "bf16_layer_names": sorted(
            name for name, mode in inspection["routes"].items() if mode == "bf16"
        ),
        "int8_layer_names": sorted(
            name for name, mode in inspection["routes"].items() if mode == "int8"
        ),
        "int4_layer_names": sorted(
            name for name, mode in inspection["routes"].items() if mode == "int4"
        ),
        "weight_error": errors_by_mode,
        "output_bytes": os.path.getsize(output_path),
        "conversion_seconds": time.time() - started,
        "layers": results,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(
        f"DONE {output_path} ({os.path.getsize(output_path) / 1e9:.3f} GB)\n"
        f"REPORT {report_path}",
        flush=True,
    )
    return report


__all__ = [
    "FluxKleinPolicy",
    "convert_flux_klein_convrot_hq",
    "default_flux_klein_output_path",
    "inspect_flux_klein_checkpoint",
    "is_flux_klein_projection",
    "load_flux_klein_policy",
]
