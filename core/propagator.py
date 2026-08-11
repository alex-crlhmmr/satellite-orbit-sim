"""
RK4 fixed-step integrator with configurable force model assembly.

Assembles accelerations from: two-body gravity, zonal harmonics (J2-J6),
atmospheric drag, solar radiation pressure, Sun/Moon third-body
perturbations, and optional external thrust.

The hot path runs on NumPy float64 for speed (an inner-loop torch
RK4 spent most of its time in framework overhead for a 6-element
state). The public API still accepts and returns torch tensors so
callers (env, main) need no changes.

State vectors are (6,) with [x, y, z, vx, vy, vz] in ECI [m, m/s].
"""

import math
import numpy as np
import torch

from .constants import (
    MU_EARTH, R_EARTH, OMEGA_EARTH, SECONDS_PER_DAY,
    DEFAULT_CD, DEFAULT_CR, DEFAULT_AREA_MASS, JD_J2000,
)
from .gravity import zonal_acceleration
from .third_body import sun_moon_acceleration


class Propagator:
    """
    RK4 orbit propagator with configurable perturbation forces.

    Force models are toggled via the config dictionary. Internals are
    numpy float64; the public step()/propagate() accept torch tensors
    and return torch tensors of the same dtype and device as input.
    """

    def __init__(self, config: dict):
        self.mu = float(config.get("mu", MU_EARTH))
        self.dt = float(config["dt"])
        self.enable_j2 = bool(config.get("enable_j2", True))
        self.max_j_degree = int(config.get("max_j_degree", 6))
        self.enable_drag = bool(config.get("enable_drag", True))
        self.cd = float(config.get("cd", DEFAULT_CD))
        self.area_mass = float(config.get("area_mass", DEFAULT_AREA_MASS))
        self.mass = float(config.get("mass", 1.0))
        self._drag_geometry = None
        geometry = config.get("drag_geometry")
        if geometry is not None:
            from .aerodynamics import BoxWingGeometry
            self._drag_geometry = BoxWingGeometry.from_config(geometry)
        self.enable_srp = bool(config.get("enable_srp", True))
        self.cr = float(config.get("cr", DEFAULT_CR))
        self.enable_third_body = bool(config.get("enable_third_body", True))
        self.epoch_jd = float(config.get("epoch_jd", JD_J2000))
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", torch.float64)

        # External thrust (numpy (3,) float64), or None
        self._thrust_np = None

        self._drag_module = None
        self._srp_module = None
        self._atmosphere = None

        if self.enable_drag:
            try:
                from . import atmosphere as _atm
                self._drag_module = _atm
                self._atmosphere = _atm.make_atmosphere(
                    config.get("atmosphere", {})
                )
            except ImportError:
                self.enable_drag = False

        if self.enable_srp:
            try:
                from . import srp as _srp
                self._srp_module = _srp
            except ImportError:
                self.enable_srp = False

    # ------------------------------------------------------------------
    # Thrust API
    # ------------------------------------------------------------------

    def set_thrust(self, thrust_eci):
        """
        Set external thrust acceleration in ECI frame.

        Args:
            thrust_eci: (3,) torch.Tensor or numpy array, m/s².
        """
        if isinstance(thrust_eci, torch.Tensor):
            self._thrust_np = thrust_eci.detach().cpu().numpy().astype(
                np.float64, copy=False
            )
        else:
            self._thrust_np = np.asarray(thrust_eci, dtype=np.float64)

    # ------------------------------------------------------------------
    # Internal numpy hot path
    # ------------------------------------------------------------------

    def _acceleration_np(self, t: float, state: np.ndarray) -> np.ndarray:
        """Total acceleration for a (6,) numpy state at time t [s since epoch]."""
        r = state[:3]
        v = state[3:]

        # Two-body
        r_mag2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2]
        r_mag = math.sqrt(r_mag2)
        k = -self.mu / (r_mag2 * r_mag)
        a_total = np.array([k * r[0], k * r[1], k * r[2]], dtype=np.float64)

        if self.enable_j2:
            a_total += zonal_acceleration(
                r, mu=self.mu, re=R_EARTH, max_degree=self.max_j_degree,
            )

        if self.enable_drag and self._drag_module is not None:
            jd = self.epoch_jd + t / SECONDS_PER_DAY
            area_mass = self.area_mass
            if self._drag_geometry is not None:
                omega_cross_r = np.array(
                    [-OMEGA_EARTH * r[1], OMEGA_EARTH * r[0], 0.0]
                )
                area_mass = self._drag_geometry.area_mass_ratio_lvlh(
                    r, v, v - omega_cross_r, self.mass
                )
            a_total += self._drag_module.drag_acceleration(
                r, v, self.cd, area_mass,
                atmosphere=self._atmosphere, jd=jd,
            )

        if self.enable_srp and self._srp_module is not None:
            jd = self.epoch_jd + t / SECONDS_PER_DAY
            r_sun = self._srp_module._sun_position_eci_np(jd)
            a_total += self._srp_module.srp_acceleration(
                r, r_sun, self.cr, self.area_mass,
            )

        if self.enable_third_body:
            jd = self.epoch_jd + t / SECONDS_PER_DAY
            a_total += sun_moon_acceleration(r, jd)

        if self._thrust_np is not None:
            a_total += self._thrust_np

        return a_total

    def _state_derivative_np(self, t: float, state: np.ndarray) -> np.ndarray:
        a = self._acceleration_np(t, state)
        dstate = np.empty(6, dtype=np.float64)
        dstate[0] = state[3]
        dstate[1] = state[4]
        dstate[2] = state[5]
        dstate[3] = a[0]
        dstate[4] = a[1]
        dstate[5] = a[2]
        return dstate

    def _rk4_step_np(self, t: float, state: np.ndarray, dt: float) -> np.ndarray:
        k1 = self._state_derivative_np(t, state)
        k2 = self._state_derivative_np(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = self._state_derivative_np(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = self._state_derivative_np(t + dt, state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # ------------------------------------------------------------------
    # Public torch-facing API
    # ------------------------------------------------------------------

    def acceleration(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """Compute total acceleration. Accepts torch (6,) state, returns torch (3,)."""
        state_np = state.detach().cpu().numpy().astype(np.float64, copy=False)
        a_np = self._acceleration_np(t, state_np)
        return torch.from_numpy(a_np).to(device=state.device, dtype=self.dtype)

    def step(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """Single RK4 step. Accepts torch (6,) state, returns torch (6,)."""
        state_np = state.detach().cpu().numpy().astype(np.float64, copy=True)
        out_np = self._rk4_step_np(t, state_np, self.dt)
        return torch.from_numpy(out_np).to(device=state.device, dtype=self.dtype)

    def propagate(
        self,
        state: torch.Tensor,
        duration: float,
        t0: float = 0.0,
    ) -> tuple:
        """
        Propagate the state for a given duration.

        Time-dependent forces (SRP, Sun/Moon) evaluate ephemerides at
        epoch_jd + (t0 + elapsed)/86400, so chunked callers MUST pass
        the running sim time as ``t0``.

        Returns:
            (final_state, trajectory) where final_state is a torch
            tensor of the same shape as input and trajectory is a list
            of torch states at each integration sample (incl. initial).
        """
        state_np = state.detach().cpu().numpy().astype(np.float64, copy=True)
        device = state.device
        dtype = self.dtype

        trajectory = [torch.from_numpy(state_np.copy()).to(device=device, dtype=dtype)]

        t = float(t0)
        n_steps = int(duration / self.dt)
        remainder = duration - n_steps * self.dt

        for _ in range(n_steps):
            state_np = self._rk4_step_np(t, state_np, self.dt)
            t += self.dt
            trajectory.append(
                torch.from_numpy(state_np.copy()).to(device=device, dtype=dtype)
            )

        if remainder > 1e-12:
            state_np = self._rk4_step_np(t, state_np, remainder)
            trajectory.append(
                torch.from_numpy(state_np.copy()).to(device=device, dtype=dtype)
            )

        final = torch.from_numpy(state_np).to(device=device, dtype=dtype)
        return final, trajectory
