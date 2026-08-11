"""Configuration contract tests: documented keys must not be silent no-ops."""

from copy import deepcopy

import pytest

from core.high_fidelity import HighFidelityPropagator
from core.propagator import Propagator
from main import (
    build_argument_parser,
    build_constellation_telemetry,
    build_propagator,
    build_spacecraft,
    load_config,
    spacecraft_configs,
)


def test_default_selects_high_fidelity_backend():
    propagator = build_propagator(load_config(), 2460390.0)
    assert isinstance(propagator, HighFidelityPropagator)
    assert propagator.gravity_degree == 20
    assert propagator.enable_drag is True


def test_legacy_backend_uses_legacy_schema():
    config = deepcopy(load_config())
    config["propagator"]["backend"] = "legacy"
    config["propagator"]["legacy"]["enable_drag"] = False
    propagator = build_propagator(config, 2460390.0)
    assert isinstance(propagator, Propagator)
    assert propagator.enable_drag is False


def test_minimal_config_override_inherits_defaults(tmp_path):
    override = tmp_path / "legacy.yaml"
    override.write_text("propagator:\n  backend: legacy\n", encoding="utf-8")

    config = load_config(override)

    assert config["propagator"]["backend"] == "legacy"
    assert config["propagator"]["dt"] == 10.0
    assert config["satellite"]["mass_kg"] == 100.0


def test_config_root_must_be_mapping(tmp_path):
    override = tmp_path / "invalid.yaml"
    override.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="root must be a mapping"):
        load_config(override)


def test_backend_cli_override_is_constrained():
    parser = build_argument_parser()
    assert parser.parse_args(["--backend", "legacy"]).backend == "legacy"
    assert parser.parse_args(["--target", "deputy-1"]).target == "deputy-1"
    with pytest.raises(SystemExit):
        parser.parse_args(["--backend", "unknown"])


def test_high_fidelity_rejects_unsupported_geometry():
    config = deepcopy(load_config())
    config["propagator"]["high_fidelity"]["drag_geometry"] = {
        "box_dimensions_m": [1.0, 1.0, 1.0]
    }
    with pytest.raises(ValueError, match="does not support: drag_geometry"):
        build_propagator(config, 2460390.0)


def test_unknown_backend_is_rejected():
    config = deepcopy(load_config())
    config["propagator"]["backend"] = "magic"
    with pytest.raises(ValueError, match="unknown propagator backend"):
        build_propagator(config, 2460390.0)


def test_unknown_backend_specific_key_is_rejected():
    config = deepcopy(load_config())
    config["propagator"]["high_fidelity"]["gravty_degree"] = 20
    with pytest.raises(ValueError, match="unknown high_fidelity keys: gravty_degree"):
        build_propagator(config, 2460390.0)


def test_legacy_single_satellite_configuration_is_preserved():
    config = load_config()
    expanded = spacecraft_configs(config)
    assert len(expanded) == 1
    assert expanded[0]["id"] == "sat-0"
    assert expanded[0]["orbit"] == config["orbit"]


def test_constellation_overrides_are_isolated():
    config = deepcopy(load_config())
    config["propagator"]["backend"] = "legacy"
    config["propagator"]["legacy"]["enable_drag"] = False
    config["satellites"] = [
        {"id": "chief"},
        {"id": "deputy", "orbit": {"true_anomaly_deg": 12.0},
         "satellite": {"mass_kg": 12.0}},
    ]
    expanded = spacecraft_configs(config)
    assert expanded[0]["orbit"]["true_anomaly_deg"] == 0.0
    assert expanded[1]["orbit"]["true_anomaly_deg"] == 12.0
    assert expanded[0]["satellite"]["mass_kg"] == 100.0
    assert expanded[1]["satellite"]["mass_kg"] == 12.0

    runtime = build_spacecraft(config, 2460390.0, trail_length=8)
    assert [item.identifier for item in runtime] == ["chief", "deputy"]
    assert runtime[0].propagator is not runtime[1].propagator
    telemetry = build_constellation_telemetry(runtime, 0.0, 2460390.0, 1)
    assert telemetry["schema_version"] == 2
    assert telemetry["satellite_count"] == 2
    assert telemetry["target_id"] == "deputy"
    assert [item["id"] for item in telemetry["satellites"]] == ["chief", "deputy"]
    assert telemetry["position_eci_m"] == telemetry["satellites"][1]["position_eci_m"]
    assert all(item["active"] for item in telemetry["satellites"])


@pytest.mark.parametrize("satellites, message", [
    ([], "non-empty"),
    ([{"id": "same"}, {"id": "same"}], "duplicate"),
    (["invalid"], "mapping"),
])
def test_invalid_constellation_configuration_is_rejected(satellites, message):
    config = load_config()
    config["satellites"] = satellites
    with pytest.raises(ValueError, match=message):
        spacecraft_configs(config)
