"""Contract tests keeping the CLI and programmatic API in sync."""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from convert_to_quant import quantize
from convert_to_quant.cli.main import extract_filter_flags, get_parser, run_conversion
from convert_to_quant.constants import MODEL_FILTERS


@pytest.mark.unit
def test_every_args_access_has_a_parser_definition():
    main_path = Path(__file__).parents[1] / "convert_to_quant" / "cli" / "main.py"
    accesses = set(re.findall(r"args\.([A-Za-z0-9_]+)", main_path.read_text(encoding="utf-8")))
    parser_dests = {action.dest for action in get_parser()._actions}

    assert accesses <= parser_dests, f"args used but not defined by the parser: {sorted(accesses - parser_dests)}"


@pytest.mark.unit
def test_parser_option_strings_and_destinations_are_unique():
    parser = get_parser()
    option_owners: dict[str, str] = {}
    duplicate_options: dict[str, set[str]] = {}

    for action in parser._actions:
        for option in action.option_strings:
            previous = option_owners.setdefault(option, action.dest)
            if previous != action.dest:
                duplicate_options.setdefault(option, {previous}).add(action.dest)

    assert not duplicate_options
    assert len([action.dest for action in parser._actions]) == len({action.dest for action in parser._actions})


@pytest.mark.unit
def test_programmatic_api_uses_parser_defaults(monkeypatch):
    captured = {}

    def capture(namespace):
        captured.update(vars(namespace))

    cli_main = importlib.import_module("convert_to_quant.cli.main")
    monkeypatch.setattr(cli_main, "run_conversion", capture)
    quantize("input.safetensors", output="output.safetensors", int8=True, simple=True)

    expected = {
        action.dest: action.default
        for action in get_parser()._actions
        if action.dest != "help"
    }
    expected.update(input="input.safetensors", output="output.safetensors", int8=True, simple=True)
    assert captured == expected


@pytest.mark.unit
def test_programmatic_api_rejects_unknown_arguments():
    with pytest.raises(ValueError, match="Unknown parameter"):
        quantize("input.safetensors", definitely_not_an_argument=True)


@pytest.mark.unit
def test_every_model_filter_is_exposed_and_extractable():
    args = get_parser().parse_args(["--input", "input.safetensors"])
    for name in MODEL_FILTERS:
        assert hasattr(args, name)
        setattr(args, name, True)

    extracted = extract_filter_flags(args)
    assert set(MODEL_FILTERS) <= set(extracted)


@pytest.mark.unit
def test_run_conversion_forwards_the_complete_unified_contract(monkeypatch):
    input_path = Path("test_cli_forwarding_input.safetensors")
    output_path = Path("test_cli_forwarding_output.safetensors")
    captured = {}

    def capture(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    cli_main = importlib.import_module("convert_to_quant.cli.main")
    monkeypatch.setattr(cli_main, "convert_to_fp8_scaled", capture)
    input_path.write_bytes(b"existence check only")
    try:
        namespace = get_parser().parse_args(
            [
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--int8",
                "--scaling-mode",
                "tensor",
                "--simple",
                "--manual-seed",
                "123",
            ]
        )
        run_conversion(namespace)
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    assert captured["args"] == (str(input_path), str(output_path), False)
    assert set(captured["kwargs"]) == {
        "block_size",
        "calib_cpu",
        "calib_samples",
        "convrot",
        "convrot_group_size",
        "custom_block_size",
        "custom_convrot",
        "custom_convrot_group_size",
        "custom_full_precision_mm",
        "custom_heur",
        "custom_layers",
        "custom_scaling_mode",
        "custom_simple",
        "custom_type",
        "device",
        "dynamic_convrot",
        "early_stop_loss",
        "early_stop_lr",
        "early_stop_stall",
        "exclude_layers",
        "extract_lora",
        "fallback",
        "fallback_block_size",
        "fallback_simple",
        "filter_flags",
        "full_matrix",
        "full_precision_matrix_mult",
        "include_input_scale",
        "int8",
        "layer_config",
        "layer_config_fullmatch",
        "lora_ar_threshold",
        "lora_depth",
        "lora_output",
        "lora_rank",
        "lora_target",
        "low_memory",
        "lr",
        "lr_adaptive_mode",
        "lr_cooldown",
        "lr_factor",
        "lr_gamma",
        "lr_min",
        "lr_patience",
        "lr_schedule",
        "lr_shape_influence",
        "lr_threshold",
        "lr_threshold_mode",
        "max_k",
        "min_k",
        "no_learned_rounding",
        "num_iter",
        "optimizer",
        "primary_format",
        "save_quant_metadata",
        "scale_optimization",
        "scaling_mode",
        "seed",
        "skip_inefficient_layers",
        "top_p",
        "use_speed",
    }
    assert captured["kwargs"]["seed"] == 123
    assert captured["kwargs"]["int8"] is True
    assert captured["kwargs"]["no_learned_rounding"] is True
    assert captured["kwargs"]["scaling_mode"] == "tensor"
