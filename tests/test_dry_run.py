from __future__ import annotations

import subprocess
import sys

import torch
from safetensors.torch import save_file


def test_dry_run_analyzes_without_writing_and_template_honors_output(tmp_path):
    input_path = tmp_path / "input.safetensors"
    output_path = tmp_path / "must_not_exist.safetensors"
    template_path = tmp_path / "custom" / "routing.json"
    save_file(
        {
            "transformer.block.weight": torch.ones((8, 8), dtype=torch.float32),
            "transformer.block.bias": torch.ones((8,), dtype=torch.float32),
        },
        str(input_path),
    )

    analyze = subprocess.run(
        [
            sys.executable,
            "-m",
            "convert_to_quant.cli.main",
            "-i",
            str(input_path),
            "-o",
            str(output_path),
            "--int8",
            "--dry-run",
            "analyze",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert analyze.returncode == 0, analyze.stderr
    assert "Dry-run analysis (no conversion will be performed)" in analyze.stdout
    assert "transformer.block.weight [8, 8] -> primary:int8" in analyze.stdout
    assert "passthrough tensors: 1" in analyze.stdout
    assert "No output file was written." in analyze.stdout
    assert "RuntimeWarning" not in analyze.stderr
    assert not output_path.exists()

    template = subprocess.run(
        [
            sys.executable,
            "-m",
            "convert_to_quant.cli.main",
            "-i",
            str(input_path),
            "-o",
            str(template_path),
            "--dry-run",
            "create-template",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert template.returncode == 0, template.stderr
    assert template_path.is_file()
    assert not input_path.with_name("input_layer_config_template.json").exists()
