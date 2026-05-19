"""
RK4 fixed-step integrator with configurable force model assembly.

Assembles accelerations from: two-body gravity, zonal harmonics (J2-J6),
atmospheric drag, solar radiation pressure, Sun/Moon third-body
perturbations, and optional external thrust.

All computations use float64 tensors. State vectors are (6,) or (B, 6)
with [x, y, z, vx, vy, vz] in ECI frame.
"""

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

    The propagator integrates the equations of motion using a fixed-step
    fourth-order Runge-Kutta scheme. Force models are toggled via the
    config dictionary.
    """

    def __init__(self, config: dict):
        """
        Initialize propagator with force model configuration.

        Args:
            config: dict with keys:
                mu            - gravitational parameter [m^3/s^2] (default MU_EARTH)
                dt            - integration time step [s] (required)
                enable_j2     - enable zonal harmonics (default True)
                max_j_degree  - max zonal harmonic degree, 2-6 (default 6)
                enable_drag   - enable atmospheric drag (default True)
                cd            - drag coefficient (default 2.2)
                area_mass     - area-to-mass ratio [m^2/kg] (default 0.01)
                enable_srp    - enable solar radiation pressure (default True)
                cr            - reflectivity coefficient (default 1.5)
                enable_third_body - enable Sun/Moon perturbations (default True)
                epoch_jd      - Julian date of epoch (default JD_J2000)
                device        - torch device (default 'cpu')
                dtype         - torch dtype (default torch.float64)
        """
        self.mu = config.get("mu", MU_EARTH)
        self.dt = config["dt"]
        self.enable_j2 = config.get("enable_j2", True)
        self.max_j_degree = config.get("max_j_degree", 6)
        self.enable_drag = config.get("enable_drag", True)
        self.cd = config.get("cd", DEFAULT_CD)
        self.area_mass = config.get("area_mass", DEFAULT_AREA_MASS)
        self.enable_srp = config.get("enable_srp", True)
        self.cr = config.get("cr", DEFAULT_CR)
        self.enable_third_body = config.get("enable_third_body", True)
        self.epoch_jd = config.get("epoch_jd", JD_J2000)
        self.device = config.get("device", "cpu")
        self.dtype = config.get("dtype", torch.float64)

        # External thrust acceleration (optional)
        self._thrust = None

        # Lazy-import drag and SRP modules to avoid hard failures
        # if those modules are not yet available
        self._drag_module = None
        self._srp_module = None

        if self.enable_drag:
            try:
                from . import atmosphere as _atm
                self._drag_module = _atm
            except ImportError:
                self.enable_drag = False

        if self.enable_srp:
            try:
                from . import srp as _srp
                self._srp_module = _srp
            except ImportError:
                self.enable_srp = False

    def set_thrust(self, thrust_eci: torch.Tensor):
        """
        Set external thrust acceleration in ECI frame.

        Args:
            thrust_eci: (3,) or (B, 3) thrust acceleration [m/s^2]
        """
        self._thrust = thrust_eci.to(device=self.device, dtype=self.dtype)

    def acceleration(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute total acceleration from all enabled force models.

        Args:
            t: time since epoch [s]
            state: (6,) or (B, 6) state vector [x, y, z, vx, vy, vz] in ECI [m, m/s]

        Returns:
            accel: (3,) or (B, 3) total acceleration [m/s^2]
        """
        batched = state.dim() == 2
        r = state[..., :3]  # (3,) or (B, 3)
        v = state[..., 3:]  # (3,) or (B, 3)

        r_mag = torch.norm(r, dim=-1, keepdim=True)  # (1,) or (B, 1)

        # --- Two-body (Keplerian) gravity ---
        a_total = -self.mu * r / r_mag**3

        # --- Zonal harmonics (J2-J6) ---
        if self.enable_j2:
            a_total = a_total + zonal_acceleration(
                r, mu=self.mu, re=R_EARTH, max_degree=self.max_j_degree,
            )

        # --- Atmospheric drag ---
        if self.enable_drag and self._drag_module is not None:
            a_drag = self._drag_module.drag_acceleration(
                r, v, self.cd, self.area_mass,
            )
            a_total = a_total + a_drag

        # --- Solar radiation pressure ---
        if self.enable_srp and self._srp_module is not None:
            jd = self.epoch_jd + t / SECONDS_PER_DAY
            r_sun = self._srp_module.sun_position_eci(jd)
            a_srp = self._srp_module.srp_acceleration(
                r, r_sun, self.cr, self.area_mass,
            )
            a_total = a_total + a_srp

        # --- Third-body (Sun + Moon) ---
        if self.enable_third_body:
            jd = self.epoch_jd + t / SECONDS_PER_DAY
            a_total = a_total + sun_moon_acceleration(r, jd)

        # --- External thrust ---
        if self._thrust is not None:
            a_total = a_total + self._thrust

        return a_total

    def _state_derivative(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the time derivative of the state vector.

        Args:
            t: time since epoch [s]
            state: (6,) or (B, 6) state [x, y, z, vx, vy, vz]

        Returns:
            dstate: same shape as state [vx, vy, vz, ax, ay, az]
        """
        v = state[..., 3:]
        a = self.acceleration(t, state)
        return torch.cat([v, a], dim=-1)

    def step(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Perform a single RK4 integration step.

        Args:
            t: current time since epoch [s]
            state: (6,) or (B, 6) state vector [m, m/s]

        Returns:
            new_state: (6,) or (B, 6) state at t + dt
        """
        dt = self.dt

        k1 = self._state_derivative(t, state)
        k2 = self._state_derivative(t + 0.5 * dt, state + 0.5 * dt * k1)
        k3 = self._state_derivative(t + 0.5 * dt, state + 0.5 * dt * k2)
        k4 = self._state_derivative(t + dt, state + dt * k3)

        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def propagate(
        self,
        state: torch.Tensor,
        duration: float,
        t0: float = 0.0,
    ) -> tuple:
        """
        Propagate the state for a given duration.

        Args:
            state: (6,) or (B, 6) initial state vector [m, m/s]
            duration: propagation duration [s]
            t0: absolute time since epoch at the start of this call [s].
                Time-dependent forces (SRP, Sun/Moon third-body) evaluate
                their ephemerides at epoch_jd + (t0 + elapsed)/86400, so
                callers that propagate in chunks MUST pass the running
                simulation time here — otherwise the Sun and Moon stay
                frozen at epoch for the entire run.

        Returns:
            (final_state, trajectory): final_state has same shape as input,
                trajectory is a list of state tensors at each time step
                (including the initial state).
        """
        state = state.to(device=self.device, dtype=self.dtype)
        trajectory = [state.clone()]

        t = t0
        n_steps = int(duration / self.dt)
        remainder = duration - n_steps * self.dt

        for i in range(n_steps):
            state = self.step(t, state)
            t += self.dt
            trajectory.append(state.clone())

        # Handle any fractional remainder step
        if remainder > 1e-12:
            original_dt = self.dt
            self.dt = remainder
            state = self.step(t, state)
            self.dt = original_dt
            trajectory.append(state.clone())

        return state, trajectory
