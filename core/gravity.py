"""
Zonal harmonic gravitational accelerations (J2 through J6).

Closed-form Cartesian expressions from Vallado, "Fundamentals of
Astrodynamics and Applications", 4th ed.

All functions operate on (B, 3) or (3,) position tensors in ECI frame.
"""

import torch
from .constants import MU_EARTH, R_EARTH, J2, J3, J4, J5, J6


def j2_acceleration(r: torch.Tensor, mu: float = MU_EARTH, re: float = R_EARTH) -> torch.Tensor:
    """J2 zonal harmonic perturbation acceleration [m/s²]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = torch.norm(r, dim=-1)
    r2 = r_mag**2
    z2_r2 = z**2 / r2
    f = -1.5 * J2 * mu * re**2 / r_mag**5

    ax = f * x * (1.0 - 5.0 * z2_r2)
    ay = f * y * (1.0 - 5.0 * z2_r2)
    az = f * z * (3.0 - 5.0 * z2_r2)
    return torch.stack([ax, ay, az], dim=-1)


def j3_acceleration(r: torch.Tensor, mu: float = MU_EARTH, re: float = R_EARTH) -> torch.Tensor:
    """J3 zonal harmonic perturbation acceleration [m/s²]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = torch.norm(r, dim=-1)
    r2 = r_mag**2
    z2_r2 = z**2 / r2
    f = -0.5 * J3 * mu * re**3 / r_mag**7

    ax = f * x * (15.0 * z - 35.0 * z**3 / r2)
    ay = f * y * (15.0 * z - 35.0 * z**3 / r2)
    az = f * (6.0 * r2 - 45.0 * z**2 + 35.0 * z**4 / r2)
    return torch.stack([ax, ay, az], dim=-1)


def j4_acceleration(r: torch.Tensor, mu: float = MU_EARTH, re: float = R_EARTH) -> torch.Tensor:
    """J4 zonal harmonic perturbation acceleration [m/s²]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = torch.norm(r, dim=-1)
    r2 = r_mag**2
    z2 = z**2
    z4 = z2**2
    f = 5.0 / 8.0 * J4 * mu * re**4 / r_mag**9

    ax = f * x * (3.0 * r2 - 42.0 * z2 + 63.0 * z4 / r2)
    ay = f * y * (3.0 * r2 - 42.0 * z2 + 63.0 * z4 / r2)
    az = f * z * (15.0 * r2 - 70.0 * z2 + 63.0 * z4 / r2)
    return torch.stack([ax, ay, az], dim=-1)


def j5_acceleration(r: torch.Tensor, mu: float = MU_EARTH, re: float = R_EARTH) -> torch.Tensor:
    """J5 zonal harmonic perturbation acceleration [m/s²]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = torch.norm(r, dim=-1)
    r2 = r_mag**2
    z2 = z**2
    z4 = z2**2
    f = 3.0 / 8.0 * J5 * mu * re**5 / r_mag**11

    ax = f * x * z * (35.0 * r2 - 210.0 * z2 + 231.0 * z4 / r2)
    ay = f * y * z * (35.0 * r2 - 210.0 * z2 + 231.0 * z4 / r2)
    az = f * (5.0 * r2**2 - 105.0 * r2 * z2 + 315.0 * z4 - 231.0 * z4 * z2 / r2)
    return torch.stack([ax, ay, az], dim=-1)


def j6_acceleration(r: torch.Tensor, mu: float = MU_EARTH, re: float = R_EARTH) -> torch.Tensor:
    """J6 zonal harmonic perturbation acceleration [m/s²]."""
    x, y, z = r[..., 0], r[..., 1], r[..., 2]
    r_mag = torch.norm(r, dim=-1)
    r2 = r_mag**2
    z2 = z**2
    z4 = z2**2
    z6 = z2**3
    f = -1.0 / 16.0 * J6 * mu * re**6 / r_mag**13

    ax = f * x * (35.0 * r2**2 - 945.0 * r2 * z2 + 3465.0 * z4 - 3003.0 * z6 / r2)
    ay = f * y * (35.0 * r2**2 - 945.0 * r2 * z2 + 3465.0 * z4 - 3003.0 * z6 / r2)
    az = f * z * (245.0 * r2**2 - 2205.0 * r2 * z2 + 4851.0 * z4 - 3003.0 * z6 / r2)
    return torch.stack([ax, ay, az], dim=-1)


def zonal_acceleration(
    r: torch.Tensor,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    max_degree: int = 6,
) -> torch.Tensor:
    """
    Sum of zonal harmonic accelerations up to specified degree.

    Args:
        r: (3,) or (B, 3) ECI position [m]
        max_degree: 2..6

    Returns:
        acceleration [m/s²], same shape as r
    """
    a = torch.zeros_like(r)
    if max_degree >= 2:
        a = a + j2_acceleration(r, mu, re)
    if max_degree >= 3:
        a = a + j3_acceleration(r, mu, re)
    if max_degree >= 4:
        a = a + j4_acceleration(r, mu, re)
    if max_degree >= 5:
        a = a + j5_acceleration(r, mu, re)
    if max_degree >= 6:
        a = a + j6_acceleration(r, mu, re)
    return a
