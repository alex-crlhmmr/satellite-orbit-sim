"""Configuration contract tests: documented keys must not be silent no-ops."""

from copy import deepcopy

import pytest

from core.high_fidelity import HighFidelityPropagator
from core.propagator import Propagator
from main import build_propagator, load_config


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
