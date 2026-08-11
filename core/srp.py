"""
Solar radiation pressure with cylindrical Earth shadow model.

Internal hot path is pure NumPy float64. ``sun_position_eci`` returns a
torch tensor for backward compatibility with the test suite; the
Propagator uses ``_sun_position_eci_np`` directly.
"""

import math

import numpy as np
import torch

from .constants import (
    AU,
    DAYS_PER_CENTURY,
    JD_J2000,
    P_SUN,
    R_EARTH,
)


def _sun_position_eci_np(jd: float) -> np.ndarray:
    """Low-precision Meeus Sun ephemeris, returned as numpy (3,) float64."""
    T = (jd - JD_J2000) / DAYS_PER_CENTURY

    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    M_rad = math.radians(M % 360.0)

    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M_rad)
        + (0.019993 - 0.000101 * T) * math.sin(2.0 * M_rad)
        + 0.000289 * math.sin(3.0 * M_rad)
    )

    sun_lon = L0 + C
    sun_lon_rad = math.radians(sun_lon % 360.0)

    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T
    nu = M_rad + math.radians(C)
    R_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * math.cos(nu))

    eps = 23.439291 - 0.0130042 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T
    eps_rad = math.radians(eps)

    R_m = R_au * AU

    x = R_m * math.cos(sun_lon_rad)
    y = R_m * math.sin(sun_lon_rad) * math.cos(eps_rad)
    z = R_m * math.sin(sun_lon_rad) * math.sin(eps_rad)

    return np.array([x, y, z], dtype=np.float64)


def sun_position_eci(jd: float) -> torch.Tensor:
    """Torch-returning wrapper (used by tests + renderer)."""
    return torch.from_numpy(_sun_position_eci_np(jd))


def _cylindrical_shadow_np(r_sat: np.ndarray, r_sun: np.ndarray) -> float:
    """Scalar shadow factor for the (3,) hot path. 1.0 = sunlit, 0.0 = umbra."""
    r_sun_mag = math.sqrt(r_sun[0] ** 2 + r_sun[1] ** 2 + r_sun[2] ** 2)
    ex = r_sun[0] / r_sun_mag
    ey = r_sun[1] / r_sun_mag
    ez = r_sun[2] / r_sun_mag
    sat_proj = r_sat[0] * ex + r_sat[1] * ey + r_sat[2] * ez
    perp_x = r_sat[0] - sat_proj * ex
    perp_y = r_sat[1] - sat_proj * ey
    perp_z = r_sat[2] - sat_proj * ez
    perp_dist = math.sqrt(perp_x * perp_x + perp_y * perp_y + perp_z * perp_z)
    if sat_proj < 0.0 and perp_dist < R_EARTH:
        return 0.0
    return 1.0


def cylindrical_shadow(r_sat, r_sun):
    """Public torch-or-numpy wrapper. Kept for API compatibility."""
    if isinstance(r_sat, torch.Tensor):
        r_np = r_sat.detach().cpu().numpy().astype(np.float64, copy=False)
        s_np = (
            r_sun.detach().cpu().numpy().astype(np.float64, copy=False)
            if isinstance(r_sun, torch.Tensor)
            else np.asarray(r_sun, dtype=np.float64)
        )
        if r_np.ndim == 1:
            result = _cylindrical_shadow_np(r_np, s_np)
            return torch.tensor(result, dtype=torch.float64, device=r_sat.device)
        # Batched torch path falls through to numpy batched
        shadow = np.array(
            [_cylindrical_shadow_np(r_np[k], s_np if s_np.ndim == 1 else s_np[k])
             for k in range(r_np.shape[0])],
            dtype=np.float64,
        )
        return torch.from_numpy(shadow).to(device=r_sat.device)
    r_np = np.asarray(r_sat, dtype=np.float64)
    s_np = np.asarray(r_sun, dtype=np.float64)
    if r_np.ndim == 1:
        return _cylindrical_shadow_np(r_np, s_np)
    return np.array(
        [_cylindrical_shadow_np(r_np[k], s_np if s_np.ndim == 1 else s_np[k])
         for k in range(r_np.shape[0])],
        dtype=np.float64,
    )


def srp_acceleration(
    r: np.ndarray,
    r_sun: np.ndarray,
    cr: float,
    area_mass: float,
) -> np.ndarray:
    """Solar radiation pressure acceleration in ECI [m/s²] (numpy hot path)."""
    if r.ndim == 1:
        shadow = _cylindrical_shadow_np(r, r_sun)
        if shadow == 0.0:
            return np.zeros(3, dtype=np.float64)
        dx = r_sun[0] - r[0]
        dy = r_sun[1] - r[1]
        dz = r_sun[2] - r[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        ex = dx / dist
        ey = dy / dist
        ez = dz / dist
        r_sun_mag = math.sqrt(
            r_sun[0] * r_sun[0] + r_sun[1] * r_sun[1] + r_sun[2] * r_sun[2]
        )
        flux_scale = (AU / r_sun_mag) ** 2
        k = -P_SUN * flux_scale * cr * area_mass
        return np.array([k * ex, k * ey, k * ez], dtype=np.float64)

    # Batched
    r_sat_to_sun = r_sun - r
    dist = np.linalg.norm(r_sat_to_sun, axis=-1, keepdims=True)
    e_sat_sun = r_sat_to_sun / dist
    r_sun_mag = np.linalg.norm(r_sun, axis=-1, keepdims=True)
    flux_scale = (AU / r_sun_mag) ** 2
    shadow = np.array(
        [_cylindrical_shadow_np(r[k], r_sun if r_sun.ndim == 1 else r_sun[k])
         for k in range(r.shape[0])],
        dtype=np.float64,
    )[..., None]
    return -P_SUN * flux_scale * cr * area_mass * e_sat_sun * shadow
