"""
Gymnasium-compatible orbital mechanics environment for reinforcement learning.

Provides an RK4-propagated two-body (with optional perturbations) simulation
where an agent applies thrust in the RTN frame to accomplish orbital maneuvers.

Observation space (10,):
    [0:6]  - Quasi-nonsingular ROE (da, dl, dex, dey, dix, diy)
    [6]    - Normalized altitude (altitude_km / 1000)
    [7]    - Normalized orbital period (period_s / 10000)
    [8]    - Eclipse flag (0.0 = sunlit, 1.0 = in shadow)
    [9]    - Time fraction (elapsed_time / max_time)

Action space (3,):
    Continuous [-1, 1] thrust commands in RTN frame, scaled by max_thrust/mass.
"""

import math
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from core.constants import (
    AU,
    DEG2RAD,
    JD_J2000,
    M2KM,
    MU_EARTH,
    R_EARTH,
    SECONDS_PER_DAY,
)
from core.elements import (
    cartesian_to_keplerian,
    cartesian_to_roe,
    keplerian_to_cartesian,
)
from core.frames import rtn_to_eci
from core.propagator import Propagator
from env.rewards import RewardFunction

# ---------------------------------------------------------------------------
# Sun position and shadow utilities (self-contained for this module)
# ---------------------------------------------------------------------------

def sun_position_eci(jd: float) -> torch.Tensor:
    """
    Approximate Sun position in ECI using low-precision solar coordinates.

    Based on the Astronomical Almanac, accurate to ~1 degree.

    Args:
        jd: Julian Date.

    Returns:
        (3,) float64 tensor of Sun position in ECI [m].
    """
    # Centuries since J2000.0
    T = (jd - JD_J2000) / 36525.0

    # Mean longitude and mean anomaly of the Sun [degrees]
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T ** 2) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T ** 2) % 360.0

    M_rad = math.radians(M)

    # Equation of center [degrees]
    C = (1.914602 - 0.004817 * T - 0.000014 * T ** 2) * math.sin(M_rad) \
        + (0.019993 - 0.000101 * T) * math.sin(2.0 * M_rad) \
        + 0.000289 * math.sin(3.0 * M_rad)

    # Sun true longitude and true anomaly [degrees]
    sun_lon = math.radians((L0 + C) % 360.0)

    # Obliquity of the ecliptic [degrees]
    epsilon = math.radians(23.439291 - 0.0130042 * T)

    # Distance to Sun [AU] -> [m]
    e_sun = 0.016708634 - 0.000042037 * T - 0.0000001267 * T ** 2
    nu_sun = M_rad + math.radians(C)
    r_sun = AU * (1.0 - e_sun ** 2) / (1.0 + e_sun * math.cos(nu_sun))

    # ECI coordinates
    x = r_sun * math.cos(sun_lon)
    y = r_sun * math.sin(sun_lon) * math.cos(epsilon)
    z = r_sun * math.sin(sun_lon) * math.sin(epsilon)

    return torch.tensor([x, y, z], dtype=torch.float64)


def cylindrical_shadow(r_sat: torch.Tensor, r_sun: torch.Tensor) -> float:
    """
    Cylindrical Earth shadow model.

    Args:
        r_sat: (3,) satellite position in ECI [m].
        r_sun: (3,) Sun position in ECI [m].

    Returns:
        1.0 if in shadow (eclipse), 0.0 if sunlit.
    """
    r_sat_64 = r_sat.to(torch.float64)
    r_sun_64 = r_sun.to(torch.float64)

    # Unit vector from Earth to Sun
    sun_dir = r_sun_64 / torch.norm(r_sun_64)

    # Project satellite position onto Sun direction
    proj = torch.dot(r_sat_64, sun_dir)

    # If satellite is on the Sun-side, it is not in shadow
    if proj.item() > 0.0:
        return 0.0

    # Perpendicular distance from the Earth-Sun line
    perp = r_sat_64 - proj * sun_dir
    perp_dist = torch.norm(perp).item()

    if perp_dist < R_EARTH:
        return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------

# This MUST mirror the schema in config/default.yaml so the same YAML
# file drives both the free-flight simulator (main.py) and this RL env.
DEFAULT_CONFIG = {
    # Orbital elements (ISS-like orbit)
    "orbit": {
        "semi_major_axis_m": 6778137.0,   # ~400 km altitude
        "eccentricity": 0.0001,
        "inclination_deg": 51.6,
        "raan_deg": 0.0,
        "arg_periapsis_deg": 0.0,
        "true_anomaly_deg": 0.0,
    },
    # Satellite properties
    "satellite": {
        "mass_kg": 100.0,
        "drag_coefficient": 2.2,
        "reflectivity_coefficient": 1.5,
        "area_to_mass_ratio": 0.01,       # m^2/kg
        "max_thrust_n": 0.1,              # max thrust per axis [N]
    },
    # Propagator settings (force model + integration step)
    "propagator": {
        "backend": "legacy",
        "dt": 10.0,                       # RK4 step [s]
        "legacy": {
            "enable_j2": True,
            "max_j_degree": 6,            # J2 through J6
            "enable_drag": True,
            "enable_srp": True,
            "enable_third_body": True,
            "epoch_jd": JD_J2000,
        },
    },
    # Environment / episode + reward settings
    "environment": {
        "env_dt": 60.0,                   # time per env.step() [s]
        "max_steps": 1440,
        "reward_type": "station_keeping",
        "reward_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "fuel_penalty": 0.1,
        "reward_scale": 1.0,
        "deorbit_target_altitude_km": 120.0,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base, returning a new dict."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Old env-only key names -> their canonical replacement. Passing the old
# names used to be silently ignored (the YAML config was a no-op against
# the env's private defaults); now it raises so the mismatch is loud.
_LEGACY_KEYS = {
    ("orbit", "a"): "orbit.semi_major_axis_m",
    ("orbit", "e"): "orbit.eccentricity",
    ("orbit", "i"): "orbit.inclination_deg",
    ("orbit", "raan"): "orbit.raan_deg",
    ("orbit", "argp"): "orbit.arg_periapsis_deg",
    ("orbit", "nu"): "orbit.true_anomaly_deg",
    ("satellite", "cd"): "satellite.drag_coefficient",
    ("satellite", "cr"): "satellite.reflectivity_coefficient",
    ("satellite", "area_mass"): "satellite.area_to_mass_ratio",
    ("satellite", "mass"): "satellite.mass_kg",
    ("satellite", "max_thrust"): "satellite.max_thrust_n",
    ("propagator", "j2"): "propagator.enable_j2",
    ("propagator", "drag"): "propagator.enable_drag",
    ("propagator", "max_degree"): "propagator.max_j_degree",
}


def _reject_legacy_config(config: dict) -> None:
    """Raise a clear error if a caller passes the pre-unification key names."""
    problems = []
    for (section, key), canonical in _LEGACY_KEYS.items():
        sec = config.get(section)
        if isinstance(sec, dict) and key in sec:
            problems.append(f"  {section}.{key}  ->  {canonical}")
    if "sim" in config:
        problems.append("  the 'sim' block  ->  'propagator.dt' + 'environment.*'")
    if "reward" in config:
        problems.append("  the 'reward' block  ->  'environment.reward_*' keys")
    if problems:
        raise ValueError(
            "OrbitalEnv received legacy config keys. It now shares the "
            "schema in config/default.yaml. Rename:\n"
            + "\n".join(sorted(set(problems)))
        )


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OrbitalEnv(gym.Env):
    """
    Gymnasium environment for orbital mechanics with continuous thrust control.

    The agent controls a satellite via thrust commands in the RTN
    (Radial, Along-track, Cross-track) reference frame. The dynamics are
    propagated with the shared core.Propagator, so the env uses exactly
    the same force model as the free-flight simulator (zonal harmonics
    J2-J6, atmospheric drag, solar radiation pressure, and Sun/Moon
    third-body perturbations), each individually configurable.

    See module docstring for observation and action space details.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict = None):
        super().__init__()

        # Merge user config with defaults (reject pre-unification keys
        # loudly instead of silently ignoring them)
        if config is None:
            config = {}
        _reject_legacy_config(config)
        supplied_prop_root = config.get("propagator", {})
        supplied_prop = supplied_prop_root.get("legacy", supplied_prop_root)
        explicit_epoch_jd = "epoch_jd" in supplied_prop
        self.config = _deep_merge(DEFAULT_CONFIG, config)

        # Unpack configuration
        orb = self.config["orbit"]
        sat = self.config["satellite"]
        prop_root = self.config["propagator"]
        prop = prop_root.get("legacy", prop_root)
        env_cfg = self.config["environment"]

        # Orbital elements (convert degrees to radians for angular elements)
        self.a0 = orb["semi_major_axis_m"]
        self.e0 = orb["eccentricity"]
        self.i0 = orb["inclination_deg"] * DEG2RAD
        self.raan0 = orb["raan_deg"] * DEG2RAD
        self.argp0 = orb["arg_periapsis_deg"] * DEG2RAD
        self.nu0 = orb["true_anomaly_deg"] * DEG2RAD

        # Satellite properties
        self.cd = sat["drag_coefficient"]
        self.cr = sat["reflectivity_coefficient"]
        self.area_mass = sat["area_to_mass_ratio"]
        self.mass = sat["mass_kg"]
        self.max_thrust = sat["max_thrust_n"]

        # Simulation parameters
        self.dt = prop_root["dt"]
        self.env_dt = env_cfg["env_dt"]
        self.max_steps = env_cfg["max_steps"]
        if "epoch" in config and not explicit_epoch_jd:
            from core.frames import datetime_to_jd
            ep = self.config["epoch"]
            self.epoch_jd = datetime_to_jd(datetime(
                ep["year"], ep["month"], ep["day"], ep.get("hour", 0),
                ep.get("minute", 0), ep.get("second", 0),
            ))
        else:
            self.epoch_jd = prop.get("epoch_jd", JD_J2000)

        # Build the SAME propagator the free-flight simulator uses, so a
        # trained policy sees identical physics (J2-J6 + drag + SRP +
        # Sun/Moon third-body) rather than the old env-only J2+drag model.
        prop_config = {
            "mu": MU_EARTH,
            "dt": self.dt,
            "enable_j2": prop["enable_j2"],
            "max_j_degree": prop["max_j_degree"],
            "enable_drag": prop["enable_drag"],
            "cd": self.cd,
            "area_mass": self.area_mass,
            "enable_srp": prop["enable_srp"],
            "cr": self.cr,
            "enable_third_body": prop["enable_third_body"],
            "epoch_jd": self.epoch_jd,
            "atmosphere": self.config.get("atmosphere", {}),
            "device": "cpu",
            "dtype": torch.float64,
        }
        # Two instances: one carries thrust, the reference stays ballistic.
        self._prop = Propagator(prop_config)
        self._prop_ref = Propagator(prop_config)

        # Reward function (map the unified environment.* keys onto the
        # RewardFunction config schema)
        self.reward_fn = RewardFunction({
            "type": env_cfg["reward_type"],
            "weights": env_cfg.get("reward_weights", [1.0] * 6),
            "fuel_penalty": env_cfg.get("fuel_penalty", 0.1),
            "scale": env_cfg.get("reward_scale", 1.0),
            "target_altitude": env_cfg.get(
                "deorbit_target_altitude_km", 150.0) * 1000.0,
        })

        # Compute reference orbit initial state (Cartesian ECI)
        self._init_reference_state()

        # Gymnasium spaces
        # Observation: ROE(6) + alt_norm + period_norm + eclipse + time_frac
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(10,),
            dtype=np.float32,
        )

        # Action: thrust in RTN, normalized to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )

        # State variables (set in reset)
        self.r = None           # Current position ECI [m] (float64 tensor)
        self.v = None           # Current velocity ECI [m/s] (float64 tensor)
        self.r_ref = None       # Reference orbit position ECI [m]
        self.v_ref = None       # Reference orbit velocity ECI [m/s]
        self.elapsed_time = 0.0
        self.step_count = 0
        self.prev_a = None      # Previous semi-major axis (for orbit raising)
        self.thrust_accel = torch.zeros(3, dtype=torch.float64)

    def _init_reference_state(self):
        """Compute initial Cartesian state from Keplerian elements."""
        a = torch.tensor(self.a0, dtype=torch.float64)
        e = torch.tensor(self.e0, dtype=torch.float64)
        i = torch.tensor(self.i0, dtype=torch.float64)
        raan = torch.tensor(self.raan0, dtype=torch.float64)
        argp = torch.tensor(self.argp0, dtype=torch.float64)
        nu = torch.tensor(self.nu0, dtype=torch.float64)

        r0, v0 = keplerian_to_cartesian(a, e, i, raan, argp, nu)
        self.r0 = r0.to(torch.float64)
        self.v0 = v0.to(torch.float64)

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial orbital state.

        Args:
            seed: Random seed for reproducibility.
            options: Optional dict (unused).

        Returns:
            observation: (10,) float32 numpy array.
            info: dict with auxiliary information.
        """
        super().reset(seed=seed)

        # Initialize satellite state to the initial orbit
        self.r = self.r0.clone()
        self.v = self.v0.clone()

        # Reference orbit also starts at the same state
        self.r_ref = self.r0.clone()
        self.v_ref = self.v0.clone()

        # Reset counters
        self.elapsed_time = 0.0
        self.step_count = 0
        self.thrust_accel = torch.zeros(3, dtype=torch.float64)

        # Track semi-major axis for orbit-raising reward
        oe = cartesian_to_keplerian(self.r, self.v)
        self.prev_a = oe["a"].item()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """
        Advance the simulation by one environment step.

        Args:
            action: (3,) numpy array or tensor in [-1, 1], thrust in RTN frame
                    normalized by max_thrust/mass.

        Returns:
            observation: (10,) float32 numpy array.
            reward: float.
            terminated: bool (altitude < 150 km).
            truncated: bool (max_steps reached).
            info: dict with auxiliary information.
        """
        # Convert action to float64 tensor
        if isinstance(action, np.ndarray):
            action_t = torch.from_numpy(action).to(torch.float64)
        else:
            action_t = torch.as_tensor(action, dtype=torch.float64)

        # Scale action: [-1, 1] -> actual acceleration [m/s^2]
        accel_magnitude = self.max_thrust / self.mass
        thrust_rtn = action_t * accel_magnitude  # (3,) in RTN

        # Convert RTN thrust to ECI
        dcm_rtn_to_eci = rtn_to_eci(self.r, self.v)  # (3, 3)
        thrust_eci = dcm_rtn_to_eci @ thrust_rtn      # (3,)
        self.thrust_accel = thrust_eci

        # Propagate env_dt using the shared Propagator. Absolute time
        # (t0=elapsed_time) is threaded through so the Sun/Moon advance
        # instead of staying frozen at epoch.
        sat_state = torch.cat([self.r, self.v])
        self._prop.set_thrust(thrust_eci)
        sat_state, _ = self._prop.propagate(
            sat_state, self.env_dt, t0=self.elapsed_time)
        self.r, self.v = sat_state[:3], sat_state[3:6]

        # Reference orbit: same force model, no thrust.
        ref_state = torch.cat([self.r_ref, self.v_ref])
        ref_state, _ = self._prop_ref.propagate(
            ref_state, self.env_dt, t0=self.elapsed_time)
        self.r_ref, self.v_ref = ref_state[:3], ref_state[3:6]

        self.elapsed_time += self.env_dt
        self.step_count += 1

        # Current orbital elements
        oe = cartesian_to_keplerian(self.r, self.v)
        current_a = oe["a"].item()

        # Altitude [m]
        altitude_m = torch.norm(self.r).item() - R_EARTH

        # Compute ROE
        roe = cartesian_to_roe(self.r_ref, self.v_ref, self.r, self.v)

        # Build reward observation dict (tensors for reward computation)
        delta_a = torch.tensor(current_a - self.prev_a, dtype=torch.float64)
        reward_obs = {
            "roe": roe,
            "delta_a": delta_a,
            "altitude": torch.tensor(altitude_m, dtype=torch.float64),
        }

        reward_tensor = self.reward_fn(reward_obs, action_t)
        reward = reward_tensor.item()

        self.prev_a = current_a

        # Termination: altitude below 150 km
        terminated = altitude_m < 150_000.0

        # Truncation: max steps reached
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """
        Compute the observation vector from the current state.

        Returns:
            (10,) float32 numpy array:
                [0:6]  ROE
                [6]    altitude_km / 1000 (normalized)
                [7]    period_s / 10000 (normalized)
                [8]    eclipse flag (0 or 1)
                [9]    time fraction (elapsed / max_time)
        """
        # ROE relative to reference orbit
        roe = cartesian_to_roe(self.r_ref, self.v_ref, self.r, self.v)

        # Altitude
        altitude_m = torch.norm(self.r).item() - R_EARTH
        altitude_km = altitude_m * M2KM
        alt_norm = altitude_km / 1000.0

        # Orbital period
        oe = cartesian_to_keplerian(self.r, self.v)
        a = oe["a"].item()
        period_s = 2.0 * math.pi * math.sqrt(abs(a) ** 3 / MU_EARTH)
        period_norm = period_s / 10000.0

        # Eclipse flag
        current_jd = self.epoch_jd + self.elapsed_time / SECONDS_PER_DAY
        r_sun = sun_position_eci(current_jd)
        eclipse = cylindrical_shadow(self.r, r_sun)

        # Time fraction
        max_time = self.max_steps * self.env_dt
        time_frac = self.elapsed_time / max_time if max_time > 0 else 0.0

        # Build observation (float64 physics, cast to float32 for gym)
        obs = np.array([
            roe[0].item(),
            roe[1].item(),
            roe[2].item(),
            roe[3].item(),
            roe[4].item(),
            roe[5].item(),
            alt_norm,
            period_norm,
            eclipse,
            time_frac,
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> dict:
        """
        Return auxiliary information about the current state.

        Returns:
            dict with keys:
                altitude_km: float
                period_s: float
                roe: (6,) float64 tensor
                keplerian: dict with a, e, i, raan, argp, nu (floats, angles in rad)
                elapsed_time: float [s]
        """
        # Altitude
        altitude_m = torch.norm(self.r).item() - R_EARTH
        altitude_km = altitude_m * M2KM

        # Keplerian elements
        oe = cartesian_to_keplerian(self.r, self.v)

        a = oe["a"].item()
        period_s = 2.0 * math.pi * math.sqrt(abs(a) ** 3 / MU_EARTH)

        # ROE
        roe = cartesian_to_roe(self.r_ref, self.v_ref, self.r, self.v)

        return {
            "altitude_km": altitude_km,
            "period_s": period_s,
            "roe": roe.detach().cpu().numpy().tolist(),
            "keplerian": {
                "a": oe["a"].item(),
                "e": oe["e"].item(),
                "i": oe["i"].item(),
                "raan": oe["raan"].item(),
                "argp": oe["argp"].item(),
                "nu": oe["nu"].item(),
            },
            "elapsed_time": self.elapsed_time,
        }
