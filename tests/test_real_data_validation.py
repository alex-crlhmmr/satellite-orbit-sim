"""Tests for the leakage-safe real-data validation pipeline."""

from pathlib import Path

import numpy as np
import yaml

from validation.real_data.benchmark import fit_effective_area
from validation.real_data.sentinel_eof import read_eof
from validation.real_data.swarm_sp3 import read_sp3_zip


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_has_immutable_disjoint_splits():
    manifest = yaml.safe_load((ROOT / "validation/real_data/manifest.yaml").read_text())
    filenames = [entry["filename"] for entry in manifest["files"]]
    assert len(filenames) == len(set(filenames))
    assert {entry["split"] for entry in manifest["files"]} == {
        "train", "validation", "test"
    }
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"])
    assert all(entry["split"] != "train" for entry in manifest["files"]
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


def test_swarm_sp3_parser_converts_units_and_gps_time(tmp_path):
    from zipfile import ZipFile
    archive = tmp_path / "orbit.ZIP"
    header = """<Earth_Explorer_Header><Fixed_Header><Validity_Period>
      <Validity_Start>UTC=2024-04-01T23:59:42</Validity_Start>
      </Validity_Period></Fixed_Header></Earth_Explorer_Header>"""
    sp3 = """#dV2024  4  2  0  0  0.00000000       2 TEST
%c M  cc GPS ccc cccc cccc cccc cccc ccccc ccccc ccccc ccccc
*  2024 04 02 00 00 00.00000000
PL47     1.0000000     2.0000000     3.0000000 999999.999999
VL47    10.0000000    20.0000000    30.0000000 999999.999999
*  2024 04 02 00 00 10.00000000
PL47     1.1000000     2.1000000     3.1000000 999999.999999
VL47    11.0000000    21.0000000    31.0000000 999999.999999
EOF
"""
    with ZipFile(archive, "w") as output:
        output.writestr("orbit.HDR", header)
        output.writestr("orbit.sp3", sp3)
    arc = read_sp3_zip(archive)
    assert arc.epochs[0].isoformat() == "2024-04-01T23:59:42+00:00"
    np.testing.assert_allclose(arc.states_itrf[0], [1000, 2000, 3000, 1, 2, 3])


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
