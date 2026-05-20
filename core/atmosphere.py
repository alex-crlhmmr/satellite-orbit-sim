"""
Exponential atmosphere drag model with 28 altitude bands (US Standard 1976).

Internal hot path is pure NumPy float64. ``atmospheric_density`` accepts
torch tensors for backward compatibility with the test suite.
"""

import numpy as np
import torch

from .constants import ATMOSPHERE_BANDS, OMEGA_EARTH, R_EARTH, MU_EARTH


_BAND_BASE_ALT_M = np.array(
    [_alt_km * 1000.0 for _alt_km, _, _ in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_BAND_BASE_DENSITY = np.array(
    [_rho for _, _rho, _ in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_BAND_SCALE_HEIGHT_M = np.array(
    [_H_km * 1000.0 for _, _, _H_km in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_NUM_BANDS = len(ATMOSPHERE_BANDS)
_MAX_ALT_M = 1000.0 * 1000.0


def _density_scalar(altitude_m: float) -> float:
    """Scalar fast path — used in the Propagator hot loop."""
    h = max(0.0, min(_MAX_ALT_M, altitude_m))
    # searchsorted returns the insertion index; we want the largest base_alt <= h,
    # so subtract one. Clamp into [0, _NUM_BANDS - 1].
    idx = int(np.searchsorted(_BAND_BASE_ALT_M, h, side="right")) - 1
    if idx < 0:
        idx = 0
    elif idx >= _NUM_BANDS:
        idx = _NUM_BANDS - 1
    h_base = _BAND_BASE_ALT_M[idx]
    rho_base = _BAND_BASE_DENSITY[idx]
    H = _BAND_SCALE_HEIGHT_M[idx]
    return rho_base * np.exp(-(h - h_base) / H)


def _density_array(altitude_m: np.ndarray) -> np.ndarray:
    """Vectorised numpy version for batched altitude arrays."""
    h = np.clip(altitude_m, 0.0, _MAX_ALT_M)
    idx = np.searchsorted(_BAND_BASE_ALT_M, h, side="right") - 1
    idx = np.clip(idx, 0, _NUM_BANDS - 1)
    h_base = _BAND_BASE_ALT_M[idx]
    rho_base = _BAND_BASE_DENSITY[idx]
    H = _BAND_SCALE_HEIGHT_M[idx]
    return rho_base * np.exp(-(h - h_base) / H)


def atmospheric_density(altitude_m):
    """
    Compute atmospheric density using the 28-band US Standard 1976 model.

    Accepts either a torch.Tensor or numpy.ndarray (or scalar). Returns
    the same type as the input. The Propagator hot path goes through
    the internal _density_scalar/_density_array helpers directly.
    """
    if isinstance(altitude_m, torch.Tensor):
        alt_np = altitude_m.detach().cpu().numpy().astype(np.float64, copy=False)
        if alt_np.ndim == 0:
            rho = np.float64(_density_scalar(float(alt_np)))
        else:
            rho = _density_array(alt_np)
        return torch.from_numpy(np.asarray(rho)).to(
            device=altitude_m.device, dtype=torch.float64
        ).reshape(altitude_m.shape)
    arr = np.asarray(altitude_m, dtype=np.float64)
    if arr.ndim == 0:
        return _density_scalar(float(arr))
    return _density_array(arr)


def drag_acceleration(
    r: np.ndarray,
    v: np.ndarray,
    cd: float,
    area_mass: float,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    omega: float = OMEGA_EARTH,
) -> np.ndarray:
    """
    Atmospheric drag acceleration in ECI [m/s²].

    Supports both single (3,) and batched (B, 3) numpy arrays.
    """
    if r.ndim == 1:
        # Scalar fast path
        r_mag = np.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
        rho = _density_scalar(r_mag - re)
        # v_rel = v - omega x r, omega = [0, 0, omega]
        vrel0 = v[0] + omega * r[1]
        vrel1 = v[1] - omega * r[0]
        vrel2 = v[2]
        vrel_mag = np.sqrt(vrel0 * vrel0 + vrel1 * vrel1 + vrel2 * vrel2)
        k = -0.5 * rho * cd * area_mass * vrel_mag
        return np.array([k * vrel0, k * vrel1, k * vrel2], dtype=np.float64)

    # Batched
    r_mag = np.linalg.norm(r, axis=-1)
    rho = _density_array(r_mag - re)
    omega_cross_r = np.zeros_like(r)
    omega_cross_r[..., 0] = -omega * r[..., 1]
    omega_cross_r[..., 1] = omega * r[..., 0]
    v_rel = v - omega_cross_r
    v_rel_mag = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    return (-0.5 * cd * area_mass) * rho[..., None] * v_rel_mag * v_rel
