"""
Third-body perturbation accelerations from Sun and Moon.

Simplified lunar ephemeris (Brown's theory, ~0.5 deg accuracy) and
standard third-body gravitational perturbation formula.

All functions support (3,) and (B, 3) position tensors in ECI frame.
"""

import torch
import math
from .constants import MU_SUN, MU_MOON, JD_J2000, DAYS_PER_CENTURY, DEG2RAD


def moon_position_eci(jd: float) -> torch.Tensor:
    """
    Simplified Moon position in ECI (geocentric) using Brown's theory.

    Accuracy is approximately 0.5 degrees, sufficient for third-body
    perturbation calculations in LEO/MEO orbit propagation.

    Args:
        jd: Julian date

    Returns:
        r_moon: (3,) tensor, Moon position in ECI [m]
    """
    # Centuries since J2000.0
    T = (jd - JD_J2000) / DAYS_PER_CENTURY

    # Fundamental arguments in degrees
    # Mean longitude of the Moon (L')
    L_prime = (218.3164477 + 481267.88123421 * T
               - 0.0015786 * T**2
               + T**3 / 538841.0
               - T**4 / 65194000.0)

    # Mean anomaly of the Moon (l')
    l_prime = (134.9633964 + 477198.8675055 * T
               + 0.0087414 * T**2
               + T**3 / 69699.0
               - T**4 / 14712000.0)

    # Mean elongation of the Moon (D)
    D = (297.8501921 + 445267.1114034 * T
         - 0.0018819 * T**2
         + T**3 / 545868.0
         - T**4 / 113065000.0)

    # Argument of latitude of the Moon (F)
    F = (93.2720950 + 483202.0175233 * T
         - 0.0036539 * T**2
         - T**3 / 3526000.0
         + T**4 / 863310000.0)

    # Mean anomaly of the Sun (l)
    l_sun = (357.5291092 + 35999.0502909 * T
             - 0.0001536 * T**2
             + T**3 / 24490000.0)

    # Convert to radians
    L_prime_rad = L_prime * DEG2RAD
    l_prime_rad = l_prime * DEG2RAD
    D_rad = D * DEG2RAD
    F_rad = F * DEG2RAD
    l_sun_rad = l_sun * DEG2RAD

    # Ecliptic longitude corrections (simplified, largest terms)
    longitude = L_prime + (
        6.288774 * math.sin(l_prime_rad)
        + 1.274027 * math.sin(2.0 * D_rad - l_prime_rad)
        + 0.658314 * math.sin(2.0 * D_rad)
        + 0.213618 * math.sin(2.0 * l_prime_rad)
        - 0.185116 * math.sin(l_sun_rad)
        - 0.114332 * math.sin(2.0 * F_rad)
    )

    # Ecliptic latitude corrections (simplified, largest terms)
    latitude = (
        5.128122 * math.sin(F_rad)
        + 0.280602 * math.sin(l_prime_rad + F_rad)
        + 0.277693 * math.sin(l_prime_rad - F_rad)
        + 0.173237 * math.sin(2.0 * D_rad - F_rad)
        + 0.055413 * math.sin(2.0 * D_rad - l_prime_rad + F_rad)
        + 0.046271 * math.sin(2.0 * D_rad - l_prime_rad - F_rad)
    )

    # Distance corrections (simplified, largest terms) in km
    # Mean distance ~385000 km
    r_km = 385000.56 + (
        -20905.355 * math.cos(l_prime_rad)
        - 3699.111 * math.cos(2.0 * D_rad - l_prime_rad)
        - 2955.968 * math.cos(2.0 * D_rad)
        - 569.925 * math.cos(2.0 * l_prime_rad)
    )

    # Convert to meters
    r_m = r_km * 1000.0

    # Convert ecliptic lon/lat to radians
    lon_rad = longitude * DEG2RAD
    lat_rad = latitude * DEG2RAD

    # Obliquity of the ecliptic
    epsilon = (23.439291 - 0.0130042 * T) * DEG2RAD

    # Ecliptic to ECI (equatorial) conversion
    cos_lon = math.cos(lon_rad)
    sin_lon = math.sin(lon_rad)
    cos_lat = math.cos(lat_rad)
    sin_lat = math.sin(lat_rad)
    cos_eps = math.cos(epsilon)
    sin_eps = math.sin(epsilon)

    # Position in ecliptic Cartesian
    x_ecl = r_m * cos_lat * cos_lon
    y_ecl = r_m * cos_lat * sin_lon
    z_ecl = r_m * sin_lat

    # Rotate from ecliptic to equatorial (ECI)
    x_eci = x_ecl
    y_eci = y_ecl * cos_eps - z_ecl * sin_eps
    z_eci = y_ecl * sin_eps + z_ecl * cos_eps

    return torch.tensor([x_eci, y_eci, z_eci], dtype=torch.float64)


def third_body_acceleration(
    r_sat: torch.Tensor,
    r_body: torch.Tensor,
    mu_body: float,
) -> torch.Tensor:
    """
    Third-body gravitational perturbation acceleration.

    Uses the standard formulation:
        a = mu_body * ((r_body - r_sat) / |r_body - r_sat|^3
                       - r_body / |r_body|^3)

    Args:
        r_sat: (3,) or (B, 3) satellite position in ECI [m]
        r_body: (3,) perturbing body position in ECI [m]
        mu_body: gravitational parameter of perturbing body [m^3/s^2]

    Returns:
        acceleration: same shape as r_sat [m/s^2]
    """
    # Vector from satellite to perturbing body
    r_rel = r_body - r_sat  # (3,) or (B, 3)

    r_rel_mag = torch.norm(r_rel, dim=-1, keepdim=True)
    r_body_mag = torch.norm(r_body, dim=-1, keepdim=True)

    a = mu_body * (r_rel / r_rel_mag**3 - r_body / r_body_mag**3)

    return a


def sun_moon_acceleration(
    r_sat: torch.Tensor,
    jd: float,
) -> torch.Tensor:
    """
    Combined Sun and Moon third-body perturbation acceleration.

    Args:
        r_sat: (3,) or (B, 3) satellite position in ECI [m]
        jd: Julian date at current epoch

    Returns:
        acceleration: same shape as r_sat [m/s^2]
    """
    from .srp import sun_position_eci

    r_sun = sun_position_eci(jd)
    r_moon = moon_position_eci(jd)

    a_sun = third_body_acceleration(r_sat, r_sun, MU_SUN)
    a_moon = third_body_acceleration(r_sat, r_moon, MU_MOON)

    return a_sun + a_moon
