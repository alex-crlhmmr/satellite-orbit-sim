"""Regression tests for the experimental RL environment configuration."""

from env.orbital_env import OrbitalEnv
from main import load_config


def test_default_yaml_constructs_env_and_uses_calendar_epoch():
    env = OrbitalEnv(load_config())
    observation, _ = env.reset(seed=7)

    assert observation.shape == (10,)
    # 2024-03-20 noon UTC, rather than the old silent J2000 fallback.
    assert abs(env.epoch_jd - 2460390.0) < 1e-9

