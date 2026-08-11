"""Configuration contract tests: documented keys must not be silent no-ops."""

from copy import deepcopy

import pytest

from core.high_fidelity import HighFidelityPropagator
from core.propagator import Propagator
from main import build_argument_parser, build_propagator, load_config


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
