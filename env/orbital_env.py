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
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

from core.constants import (
    MU_EARTH,
    R_EARTH,
    DEG2RAD,
    M2KM,
    JD_J2000,
    SECONDS_PER_DAY,
    AU,
    R_SUN,
)
from core.elements import (
    cartesian_to_keplerian,
    cartesian_to_roe,
    keplerian_to_cartesian,
)
from core.frames import eci_to_rtn, rtn_to_eci
from core.gravity import zonal_acceleration
from core.atmosphere import drag_acceleration
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

DEFAULT_CONFIG = {
    # Orbital elements (ISS-like orbit)
    "orbit": {
        "a": 6778137.0,       # Semi-major axis [m] (~400 km altitude)
        "e": 0.0001,          # Eccentricity
        "i": 51.6,            # Inclination [deg]
        "raan": 0.0,          # RAAN [deg]
        "argp": 0.0,          # Argument of periapsis [deg]
        "nu": 0.0,            # True anomaly [deg]
    },
    # Satellite properties
    "satellite": {
        "cd": 2.2,            # Drag coefficient
        "cr": 1.5,            # Reflectivity coefficient
        "area_mass": 0.01,    # Area-to-mass ratio [m^2/kg]
        "mass": 100.0,        # Mass [kg]
        "max_thrust": 1.0,    # Maximum thrust [N]
    },
    # Simulation parameters
    "sim": {
        "dt": 10.0,           # RK4 integration step [s]
        "env_dt": 60.0,       # Environment step duration [s]
        "max_steps": 1000,    # Maximum episode steps
    },
    # Reward configuration
    "reward": {
        "type": "station_keeping",
        "weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "fuel_penalty": 0.1,
        "scale": 1.0,
        "target_altitude": 150000.0,
    },
    # Propagator configuration
    "propagator": {
        "j2": True,
        "j3": False,
        "j4": False,
        "j5": False,
        "j6": False,
        "drag": True,
        "max_degree": 2,
        "epoch_jd": JD_J2000,
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


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class OrbitalEnv(gym.Env):
    """
    Gymnasium environment for orbital mechanics with continuous thrust control.

    The agent controls a satellite via thrust commands in the RTN
    (Radial, Along-track, Cross-track) reference frame. The dynamics
    are propagated using an RK4 integrator with configurable perturbations
    (zonal harmonics, atmospheric drag).

    See module docstring for observation and action space details.
    """

    metadata = {"render_modes": []}

    def __init__(self, config: dict = None):
        super().__init__()

        # Merge user config with defaults
        if config is None:
            self.config = DEFAULT_CONFIG.copy()
            self.config = _deep_merge(DEFAULT_CONFIG, {})
        else:
            self.config = _deep_merge(DEFAULT_CONFIG, config)

        # Unpack configuration
        orb = self.config["orbit"]
        sat = self.config["satellite"]
        sim = self.config["sim"]
        prop = self.config["propagator"]

        # Orbital elements (convert degrees to radians for angular elements)
        self.a0 = orb["a"]
        self.e0 = orb["e"]
        self.i0 = orb["i"] * DEG2RAD
        self.raan0 = orb["raan"] * DEG2RAD
        self.argp0 = orb["argp"] * DEG2RAD
        self.nu0 = orb["nu"] * DEG2RAD

        # Satellite properties
        self.cd = sat["cd"]
        self.cr = sat["cr"]
        self.area_mass = sat["area_mass"]
        self.mass = sat["mass"]
        self.max_thrust = sat["max_thrust"]

        # Simulation parameters
        self.dt = sim["dt"]
        self.env_dt = sim["env_dt"]
        self.max_steps = sim["max_steps"]

        # Propagator settings
        self.use_j2 = prop.get("j2", True)
        self.use_drag = prop.get("drag", True)
        self.max_degree = prop.get("max_degree", 2)
        self.epoch_jd = prop.get("epoch_jd", JD_J2000)

        # Reward function
        self.reward_fn = RewardFunction(self.config["reward"])

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

        # Propagate env_dt seconds using RK4 with substeps
        n_substeps = max(1, int(self.env_dt / self.dt))
        substep_dt = self.env_dt / n_substeps

        for _ in range(n_substeps):
            self.r, self.v = self._rk4_step(self.r, self.v, substep_dt, thrust_eci)

        # Also propagate the reference orbit (no thrust, same perturbations)
        for _ in range(n_substeps):
            self.r_ref, self.v_ref = self._rk4_step(
                self.r_ref, self.v_ref, substep_dt,
                torch.zeros(3, dtype=torch.float64),
            )

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

    def _rk4_step(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
        dt: float,
        thrust_eci: torch.Tensor,
    ) -> tuple:
        """
        Single RK4 integration step for orbital dynamics.

        Args:
            r: (3,) position ECI [m].
            v: (3,) velocity ECI [m/s].
            dt: Time step [s].
            thrust_eci: (3,) thrust acceleration in ECI [m/s^2].

        Returns:
            (r_new, v_new): Updated position and velocity.
        """
        def deriv(r_k, v_k):
            a_total = self._compute_acceleration(r_k, v_k) + thrust_eci
            return v_k, a_total

        # k1
        dr1, dv1 = deriv(r, v)

        # k2
        r2 = r + 0.5 * dt * dr1
        v2 = v + 0.5 * dt * dv1
        dr2, dv2 = deriv(r2, v2)

        # k3
        r3 = r + 0.5 * dt * dr2
        v3 = v + 0.5 * dt * dv2
        dr3, dv3 = deriv(r3, v3)

        # k4
        r4 = r + dt * dr3
        v4 = v + dt * dv3
        dr4, dv4 = deriv(r4, v4)

        r_new = r + (dt / 6.0) * (dr1 + 2.0 * dr2 + 2.0 * dr3 + dr4)
        v_new = v + (dt / 6.0) * (dv1 + 2.0 * dv2 + 2.0 * dv3 + dv4)

        return r_new, v_new

    def _compute_acceleration(
        self,
        r: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute total gravitational + perturbation acceleration.

        Args:
            r: (3,) position ECI [m].
            v: (3,) velocity ECI [m/s].

        Returns:
            (3,) total acceleration in ECI [m/s^2].
        """
        r_mag = torch.norm(r)

        # Two-body acceleration
        a_twobody = -MU_EARTH / r_mag ** 3 * r

        a_total = a_twobody

        # Zonal harmonics
        if self.use_j2 and self.max_degree >= 2:
            a_total = a_total + zonal_acceleration(r, max_degree=self.max_degree)

        # Atmospheric drag
        if self.use_drag:
            a_total = a_total + drag_acceleration(
                r, v, cd=self.cd, area_mass=self.area_mass,
            )

        return a_total

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
