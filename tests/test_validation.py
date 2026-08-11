"""Integrity checks for the committed independent validation evidence."""

import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_validation_matrix_covers_regimes_models_and_horizons():
    config = yaml.safe_load((ROOT / "validation/scenarios.yaml").read_text())
    assert len(config["scenarios"]) == 5
    assert config["profiles"] == [
        "two_body", "gravity", "third_body", "srp_sunlight", "srp"
    ]
    assert config["horizons_s"] == [86400.0, 259200.0, 604800.0]


def test_committed_cross_validation_passes_acceptance_limit():
    result = json.loads((ROOT / "validation/results/results.json").read_text())
    expected = 5 * 5 * 3
    assert len(result["results"]) == expected
    assert result["passed"] is True
    limits = result["acceptance"]["max_position_error_m_by_profile"]
    assert all(row["max_position_error_m"] <= limits[row["profile"]]
               for row in result["results"])
