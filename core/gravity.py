"""
Zonal harmonic gravitational accelerations (J2 through J6).

Closed-form Cartesian expressions from Vallado, "Fundamentals of
Astrodynamics and Applications", 4th ed.

All functions operate on (B, 3) or (3,) float64 NumPy arrays in ECI.
"""

import numpy as np

from .constants import J2, J3, J4, J5, J6, MU_EARTH, R_EARTH


def j2_acceleration(r: np.ndarray, mu: float = MU_EARTH, re: float = R_EARTH) -> np.ndarray:
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = np.linalg.norm(r, axis=-1)
    r2 = r_mag * r_mag
    z2_r2 = (z * z) / r2
    f = -1.5 * J2 * mu * re * re / r_mag**5

    ax = f * x * (1.0 - 5.0 * z2_r2)
    ay = f * y * (1.0 - 5.0 * z2_r2)
    az = f * z * (3.0 - 5.0 * z2_r2)
    return np.stack([ax, ay, az], axis=-1)


def j3_acceleration(r: np.ndarray, mu: float = MU_EARTH, re: float = R_EARTH) -> np.ndarray:
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = np.linalg.norm(r, axis=-1)
    r2 = r_mag * r_mag
    f = -0.5 * J3 * mu * re**3 / r_mag**7

    ax = f * x * (15.0 * z - 35.0 * z**3 / r2)
    ay = f * y * (15.0 * z - 35.0 * z**3 / r2)
    # Gradient of -mu/r * J3 * (re/r)^3 * P3(z/r).
    # The axial component is not obtained by reusing the transverse factor.
    az = (0.5 * J3 * mu * re**3 / r_mag**7
          * (3.0 * r2 - 30.0 * z * z + 35.0 * z**4 / r2))
    return np.stack([ax, ay, az], axis=-1)


def j4_acceleration(r: np.ndarray, mu: float = MU_EARTH, re: float = R_EARTH) -> np.ndarray:
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = np.linalg.norm(r, axis=-1)
    r2 = r_mag * r_mag
    z2 = z * z
    z4 = z2 * z2
    f = 5.0 / 8.0 * J4 * mu * re**4 / r_mag**9

    ax = f * x * (3.0 * r2 - 42.0 * z2 + 63.0 * z4 / r2)
    ay = f * y * (3.0 * r2 - 42.0 * z2 + 63.0 * z4 / r2)
    az = f * z * (15.0 * r2 - 70.0 * z2 + 63.0 * z4 / r2)
    return np.stack([ax, ay, az], axis=-1)


def j5_acceleration(r: np.ndarray, mu: float = MU_EARTH, re: float = R_EARTH) -> np.ndarray:
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = np.linalg.norm(r, axis=-1)
    r2 = r_mag * r_mag
    z2 = z * z
    z4 = z2 * z2
    f = 3.0 / 8.0 * J5 * mu * re**5 / r_mag**11

    ax = f * x * z * (35.0 * r2 - 210.0 * z2 + 231.0 * z4 / r2)
    ay = f * y * z * (35.0 * r2 - 210.0 * z2 + 231.0 * z4 / r2)
    az = -f * (5.0 * r2 * r2 - 105.0 * r2 * z2 + 315.0 * z4 - 231.0 * z4 * z2 / r2)
    return np.stack([ax, ay, az], axis=-1)


def j6_acceleration(r: np.ndarray, mu: float = MU_EARTH, re: float = R_EARTH) -> np.ndarray:
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = np.linalg.norm(r, axis=-1)
    r2 = r_mag * r_mag
    z2 = z * z
    z4 = z2 * z2
    z6 = z4 * z2
    f = -1.0 / 16.0 * J6 * mu * re**6 / r_mag**13

    ax = f * x * (35.0 * r2 * r2 - 945.0 * r2 * z2 + 3465.0 * z4 - 3003.0 * z6 / r2)
    ay = f * y * (35.0 * r2 * r2 - 945.0 * r2 * z2 + 3465.0 * z4 - 3003.0 * z6 / r2)
    az = f * z * (245.0 * r2 * r2 - 2205.0 * r2 * z2 + 4851.0 * z4 - 3003.0 * z6 / r2)
    return np.stack([ax, ay, az], axis=-1)


def zonal_acceleration(
    r: np.ndarray,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    max_degree: int = 6,
) -> np.ndarray:
    """Sum of zonal harmonic accelerations up to specified degree (2..6)."""
    a = j2_acceleration(r, mu, re)
    if max_degree >= 3:
        a = a + j3_acceleration(r, mu, re)
    if max_degree >= 4:
        a = a + j4_acceleration(r, mu, re)
    if max_degree >= 5:
        a = a + j5_acceleration(r, mu, re)
    if max_degree >= 6:
        a = a + j6_acceleration(r, mu, re)
    return a
