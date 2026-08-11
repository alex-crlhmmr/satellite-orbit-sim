"""Validated high-fidelity orbit propagation backed by Brahe.

The legacy :class:`core.propagator.Propagator` remains useful for fast,
transparent experiments.  This adapter is the production/research profile:
IERS Earth orientation, EGM2008 spherical harmonics, measured space weather,
DE440s ephemerides, conical eclipses, tides, relativity, adaptive integration,
and state-transition/covariance propagation.
"""

from __future__ import annotations

import numpy as np
import torch

from .uncertainty import white_acceleration_process_noise


class HighFidelityPropagator:
    """Torch-compatible adapter around ``brahe.NumericalOrbitPropagator``."""

    _data_initialized = False

    def __init__(self, config: dict):
        try:
            import brahe as bh
        except ImportError as exc:  # pragma: no cover - actionable deployment error
            raise RuntimeError(
                "The high_fidelity backend requires `pip install brahe>=1.7`."
            ) from exc

        self.bh = bh
        self.dt = float(config.get("dt", 10.0))
        self.epoch_jd = float(config["epoch_jd"])
        self.dtype = config.get("dtype", torch.float64)
        self.mass = float(config.get("mass", 100.0))
        area_mass = float(config.get("area_mass", 0.01))
        self.drag_area = float(config.get("drag_area_m2", area_mass * self.mass))
        self.srp_area = float(config.get("srp_area_m2", area_mass * self.mass))
        self.cd = float(config.get("cd", 2.2))
        self.cr = float(config.get("cr", 1.5))
        self.gravity_degree = int(config.get("gravity_degree", 20))
        self.gravity_order = int(config.get("gravity_order", self.gravity_degree))
        self.abs_tol = float(config.get("abs_tol", 1e-6))
        self.rel_tol = float(config.get("rel_tol", 1e-11))
        self.max_step = float(config.get("max_step", 60.0))
        self.enable_drag = bool(config.get("enable_drag", True))
        self.enable_srp = bool(config.get("enable_srp", True))
        self.enable_third_body = bool(config.get("enable_third_body", True))
        self.enable_relativity = bool(config.get("enable_relativity", True))
        self.enable_tides = bool(config.get("enable_tides", True))
        self.initial_covariance = np.asarray(
            config.get("initial_covariance", np.eye(6)), dtype=np.float64
        )
        self.last_covariance = self.initial_covariance.copy()
        self.process_noise_acceleration_psd_rtn = np.asarray(
            config.get("process_noise_acceleration_psd_rtn", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        self._thrust_np = None

        if not HighFidelityPropagator._data_initialized:
            bh.initialize_eop()
            if self.enable_drag:
                bh.initialize_sw()
            HighFidelityPropagator._data_initialized = True

        self._force_config = self._make_force_config()
        self._propagation_config = (
            bh.NumericalPropagationConfig
            .with_method(bh.IntegrationMethod.RKF78)
            .with_abs_tol(self.abs_tol)
            .with_rel_tol(self.rel_tol)
            .with_max_step(self.max_step)
            .with_stm()
        )

    def _make_force_config(self):
        bh = self.bh
        fixed = bh.ParameterSource.value
        drag = None
        if self.enable_drag:
            drag = bh.DragConfiguration(
                bh.AtmosphericModel.NRLMSISE00, fixed(self.drag_area), fixed(self.cd)
            )
        srp = None
        if self.enable_srp:
            srp = bh.SolarRadiationPressureConfiguration(
                fixed(self.srp_area), fixed(self.cr), bh.EclipseModel.CONICAL
            )
        third_body = None
        if self.enable_third_body:
            third_body = [
                bh.ThirdBodyConfiguration(bh.ThirdBody.SUN, bh.EphemerisSource.DE440s),
                bh.ThirdBodyConfiguration(bh.ThirdBody.MOON, bh.EphemerisSource.DE440s),
            ]
        tides = None
        if self.enable_tides:
            tides = bh.TidesConfiguration(
                bh.PermanentTideConfig.AUTO,
                bh.SolidTideConfig(frequency_dependent=True, pole_tide=True),
                None,
                bh.EphemerisSource.DE440s,
            )
        return bh.ForceModelConfig(
            gravity=bh.GravityConfiguration.spherical_harmonic(
                self.gravity_degree, self.gravity_order
            ),
            drag=drag,
            srp=srp,
            third_body=third_body,
            relativity=self.enable_relativity,
            mass=fixed(self.mass),
            tides=tides,
            frame_transform=bh.FrameTransformationModel.FULL_EARTH_ROTATION,
        )

    def set_thrust(self, thrust_eci):
        """Record thrust and reject nonzero commands for this non-RL backend."""
        value = (thrust_eci.detach().cpu().numpy() if isinstance(thrust_eci, torch.Tensor)
                 else np.asarray(thrust_eci))
        self._thrust_np = np.asarray(value, dtype=np.float64)
        if np.any(self._thrust_np != 0.0):
            raise NotImplementedError(
                "External thrust is not yet connected to the high_fidelity backend; "
                "use the legacy backend for the experimental RL environment."
            )

    def _build(self, state: np.ndarray, t0: float):
        epoch = self.bh.Epoch.from_jd(
            self.epoch_jd + t0 / 86400.0, self.bh.TimeSystem.UTC
        )
        return (
            self.bh.NumericalOrbitPropagator.builder(
                epoch, state, self._force_config
            )
            .propagation_config(self._propagation_config)
            .initial_covariance(self.last_covariance)
            .build()
        ), epoch

    def propagate(self, state: torch.Tensor, duration: float, t0: float = 0.0):
        state_np = state.detach().cpu().numpy().astype(np.float64, copy=True)
        prop, epoch = self._build(state_np, float(t0))
        final_epoch = epoch + float(duration)
        prop.propagate_to(final_epoch)
        out = np.asarray(prop.current_state(), dtype=np.float64)
        self.last_covariance = np.asarray(prop.covariance(final_epoch), dtype=np.float64)
        self.last_covariance += white_acceleration_process_noise(
            out, float(duration), self.process_noise_acceleration_psd_rtn
        )
        self.last_covariance = 0.5 * (self.last_covariance + self.last_covariance.T)
        final = torch.from_numpy(out).to(device=state.device, dtype=self.dtype)
        return final, [state.detach().clone(), final.detach().clone()]

    def step(self, t: float, state: torch.Tensor) -> torch.Tensor:
        return self.propagate(state, self.dt, t0=t)[0]
