"""Tests for the frozen GRACE-FO force-level validation."""

import json
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

from validation.density.gracefo import read_density_zip

ROOT = Path(__file__).resolve().parents[1]


def test_gracefo_parser_units_flags_and_stride(tmp_path):
    archive = tmp_path / "density.zip"
    rows = ["# header"]
    for index in range(3):
        rows.append(
            f"2024-04-01 00:00:{index * 10:02d}.000 GPS 500000.000 10.000 "
            f"20.000 12.000 30.000 {1 + index}.0E-12 2.0E-12 0.0 0.0"
        )
    with ZipFile(archive, "w") as output:
        output.writestr("density.txt", "\n".join(rows))
    observations = list(read_density_zip(archive, stride=2))
    assert len(observations) == 2
    assert observations[0].altitude_m == 500000.0
    assert observations[1].density_kg_m3 == pytest.approx(3e-12)


def test_density_protocol_is_frozen_and_storms_are_not_training():
    manifest = yaml.safe_load((ROOT / "validation/density/manifest.yaml").read_text())
    protocol = yaml.safe_load((ROOT / "validation/density/protocol.yaml").read_text())
    assert len({entry["sha256"] for entry in manifest["files"]}) == 3
    assert protocol["splits"]["train"] == [[date(2024, 4, 1), date(2024, 4, 4)]]
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert len(manifest["space_weather"]["sha256"]) == 64
    assert "d80449d2f08ea036b54809106f00e1936ff97768" in manifest["space_weather"]["url"]


def test_committed_force_level_density_gate_passes():
    result = json.loads((ROOT / "validation/density/results/results.json").read_text())
    assert result["passed"] is True
    test = result["metrics"]["test"]
    assert test["static_improvement_fraction"] >= 0.02
    assert test["online_improvement_fraction"] >= 0.10
    assert 0.2 <= test["online_mean_nis"] <= 5.0
