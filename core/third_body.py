"""
Third-body perturbation accelerations from Sun and Moon.

Simplified lunar ephemeris (Brown's theory, ~0.5 deg accuracy) plus the
standard third-body gravitational perturbation formula. Pure NumPy hot
path; ``moon_position_eci`` returns a numpy (3,) float64 array.
"""

import math

import numpy as np

from .constants import DAYS_PER_CENTURY, DEG2RAD, JD_J2000, MU_MOON, MU_SUN


def moon_position_eci(jd: float) -> np.ndarray:
    """
    Simplified Moon position in ECI (geocentric), ~0.5 deg accuracy.

    Returns: (3,) numpy float64 array, position in meters.
    """
    T = (jd - JD_J2000) / DAYS_PER_CENTURY

    L_prime = (218.3164477 + 481267.88123421 * T
               - 0.0015786 * T * T
               + T**3 / 538841.0
               - T**4 / 65194000.0)

    l_prime = (134.9633964 + 477198.8675055 * T
               + 0.0087414 * T * T
               + T**3 / 69699.0
               - T**4 / 14712000.0)

    D = (297.8501921 + 445267.1114034 * T
         - 0.0018819 * T * T
         + T**3 / 545868.0
         - T**4 / 113065000.0)

    F = (93.2720950 + 483202.0175233 * T
         - 0.0036539 * T * T
         - T**3 / 3526000.0
         + T**4 / 863310000.0)

    l_sun = (357.5291092 + 35999.0502909 * T
             - 0.0001536 * T * T
             + T**3 / 24490000.0)

    l_prime_rad = l_prime * DEG2RAD
    D_rad = D * DEG2RAD
    F_rad = F * DEG2RAD
    l_sun_rad = l_sun * DEG2RAD

    longitude = L_prime + (
        6.288774 * math.sin(l_prime_rad)
        + 1.274027 * math.sin(2.0 * D_rad - l_prime_rad)
        + 0.658314 * math.sin(2.0 * D_rad)
        + 0.213618 * math.sin(2.0 * l_prime_rad)
        - 0.185116 * math.sin(l_sun_rad)
        - 0.114332 * math.sin(2.0 * F_rad)
    )

    latitude = (
        5.128122 * math.sin(F_rad)
        + 0.280602 * math.sin(l_prime_rad + F_rad)
        + 0.277693 * math.sin(l_prime_rad - F_rad)
        + 0.173237 * math.sin(2.0 * D_rad - F_rad)
        + 0.055413 * math.sin(2.0 * D_rad - l_prime_rad + F_rad)
        + 0.046271 * math.sin(2.0 * D_rad - l_prime_rad - F_rad)
    )

    r_km = 385000.56 + (
        -20905.355 * math.cos(l_prime_rad)
        - 3699.111 * math.cos(2.0 * D_rad - l_prime_rad)
        - 2955.968 * math.cos(2.0 * D_rad)
        - 569.925 * math.cos(2.0 * l_prime_rad)
    )

    r_m = r_km * 1000.0
    lon_rad = longitude * DEG2RAD
    lat_rad = latitude * DEG2RAD
    epsilon = (23.439291 - 0.0130042 * T) * DEG2RAD

    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_eps = math.cos(epsilon)
    sin_eps = math.sin(epsilon)

    x_ecl = r_m * cos_lat * cos_lon
    y_ecl = r_m * cos_lat * sin_lon
    z_ecl = r_m * sin_lat

    x_eci = x_ecl
    y_eci = y_ecl * cos_eps - z_ecl * sin_eps
    z_eci = y_ecl * sin_eps + z_ecl * cos_eps

    return np.array([x_eci, y_eci, z_eci], dtype=np.float64)


def third_body_acceleration(
    r_sat: np.ndarray,
    r_body: np.ndarray,
    mu_body: float,
) -> np.ndarray:
    """
    Third-body gravitational perturbation:
        a = mu_body * ((r_body - r_sat) / |r_body - r_sat|^3
                       - r_body / |r_body|^3)

    Supports (3,) or (B, 3) r_sat. ``r_body`` is (3,).
    """
    if r_sat.ndim == 1:
        dx = r_body[0] - r_sat[0]
        dy = r_body[1] - r_sat[1]
        dz = r_body[2] - r_sat[2]
        rel_mag = math.sqrt(dx * dx + dy * dy + dz * dz)
        body_mag = math.sqrt(
            r_body[0] * r_body[0] + r_body[1] * r_body[1] + r_body[2] * r_body[2]
        )
        rel_inv3 = 1.0 / (rel_mag ** 3)
        body_inv3 = 1.0 / (body_mag ** 3)
        return np.array(
            [
                mu_body * (dx * rel_inv3 - r_body[0] * body_inv3),
                mu_body * (dy * rel_inv3 - r_body[1] * body_inv3),
                mu_body * (dz * rel_inv3 - r_body[2] * body_inv3),
            ],
            dtype=np.float64,
        )

    r_rel = r_body - r_sat
    r_rel_mag = np.linalg.norm(r_rel, axis=-1, keepdims=True)
    r_body_mag = np.linalg.norm(r_body, axis=-1, keepdims=True)
    return mu_body * (r_rel / r_rel_mag**3 - r_body / r_body_mag**3)


def sun_moon_acceleration(r_sat: np.ndarray, jd: float) -> np.ndarray:
    """Combined Sun + Moon third-body acceleration."""
    from .srp import _sun_position_eci_np

    r_sun = _sun_position_eci_np(jd)
    r_moon = moon_position_eci(jd)

    return (
        third_body_acceleration(r_sat, r_sun, MU_SUN)
        + third_body_acceleration(r_sat, r_moon, MU_MOON)
    )
