"""Tests for the leakage-safe real-data validation pipeline."""

from pathlib import Path

import numpy as np
import yaml

from validation.real_data.benchmark import fit_effective_area
from validation.real_data.sentinel_eof import read_eof


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_immutable_disjoint_splits():
    manifest = yaml.safe_load((ROOT / "validation/real_data/manifest.yaml").read_text())
    filenames = [entry["filename"] for entry in manifest["files"]]
    assert len(filenames) == len(set(filenames))
    assert {entry["split"] for entry in manifest["files"]} == {
        "train", "validation", "test"
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert all(entry["split"] == "test" for entry in manifest["files"]
               if entry["regime"] == "storm")


def test_eof_parser_rejects_wrong_frame(tmp_path):
    eof = tmp_path / "bad.EOF"
    eof.write_text("""<Earth_Explorer_File><Earth_Explorer_Header><Variable_Header>
      <Ref_Frame>GCRF</Ref_Frame><Time_Reference>UTC</Time_Reference>
      </Variable_Header></Earth_Explorer_Header><Data_Block/></Earth_Explorer_File>""")
    try:
        read_eof(eof)
    except ValueError as error:
        assert "unsupported EOF frame/time" in str(error)
    else:
        raise AssertionError("wrong-frame EOF was accepted")


def test_robust_area_fit_resists_one_maneuver_outlier():
    base = np.zeros((3, 6))
    response = np.zeros((3, 6)); response[:, 0] = [0.0, 10.0, 20.0]
    training = []
    for area in (8.0, 10.0, 9.0, 200.0):
        truth = base.copy()
        truth[:, 0] = response[:, 0] * area / 20.0
        training.append((truth, base.copy(), response.copy()))
    fitted, estimates = fit_effective_area(training, 20.0, [0.1, 40.0])
    assert estimates == [8.0, 10.0, 9.0, 200.0]
    assert fitted == 9.5


def test_committed_real_data_gate_passes():
    import json
    result = json.loads((ROOT / "validation/real_data/results/results.json").read_text())
    assert result["passed"] is True
    assert result["validation_rms_improvement_fraction"] >= 0.05
    assert result["storm_test_rms_change_fraction"] <= 0.10
