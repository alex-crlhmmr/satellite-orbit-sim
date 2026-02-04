"""
Solar radiation pressure with cylindrical Earth shadow model.

Provides low-precision Sun ephemeris, shadow computation, and SRP
acceleration for satellite force modelling. All tensors use float64.
"""

import torch
import math

from .constants import (
    MU_SUN,
    AU,
    P_SUN,
    R_EARTH,
    R_SUN,
    DEG2RAD,
    JD_J2000,
    DAYS_PER_CENTURY,
)


def sun_position_eci(jd: float) -> torch.Tensor:
    """
    Low-precision solar ephemeris (Meeus, ~0.01 deg accuracy).

    Parameters
    ----------
    jd : float
        Julian date (TT/TDB).

    Returns
    -------
    torch.Tensor
        Sun position in ECI (GCRF) [m], shape (3,), float64.
    """
    # Centuries since J2000.0
    T = (jd - JD_J2000) / DAYS_PER_CENTURY

    # Mean longitude of the Sun [deg]
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T * T
    L0 = L0 % 360.0

    # Mean anomaly of the Sun [deg]
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T * T
    M_rad = math.radians(M % 360.0)

    # Equation of center [deg]
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(M_rad)
        + (0.019993 - 0.000101 * T) * math.sin(2.0 * M_rad)
        + 0.000289 * math.sin(3.0 * M_rad)
    )

    # Sun true longitude [deg]
    sun_lon = L0 + C
    sun_lon_rad = math.radians(sun_lon % 360.0)

    # Sun-Earth distance [AU]
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T
    nu = M_rad + math.radians(C)  # true anomaly
    R_au = (1.000001018 * (1.0 - e * e)) / (1.0 + e * math.cos(nu))

    # Obliquity of the ecliptic [deg]
    eps = 23.439291 - 0.0130042 * T - 1.64e-7 * T * T + 5.04e-7 * T * T * T
    eps_rad = math.radians(eps)

    # Sun position in ecliptic -> equatorial (ECI)
    R_m = R_au * AU  # convert to metres

    x = R_m * math.cos(sun_lon_rad)
    y = R_m * math.sin(sun_lon_rad) * math.cos(eps_rad)
    z = R_m * math.sin(sun_lon_rad) * math.sin(eps_rad)

    return torch.tensor([x, y, z], dtype=torch.float64)


def cylindrical_shadow(
    r_sat: torch.Tensor,
    r_sun: torch.Tensor,
) -> torch.Tensor:
    """
    Cylindrical Earth shadow model.

    Parameters
    ----------
    r_sat : torch.Tensor
        Satellite position in ECI [m]. Shape (3,) or (B, 3).
    r_sun : torch.Tensor
        Sun position in ECI [m]. Shape (3,) or (B, 3).

    Returns
    -------
    torch.Tensor
        Shadow factor per satellite: 1.0 = sunlit, 0.0 = umbra.
        Shape () or (B,).
    """
    r_sat = r_sat.to(torch.float64)
    r_sun = r_sun.to(torch.float64)

    single = r_sat.dim() == 1
    if single:
        r_sat = r_sat.unsqueeze(0)  # (1, 3)
    if r_sun.dim() == 1:
        r_sun = r_sun.unsqueeze(0)  # (1, 3) or broadcast-ready

    # Unit vector from Earth to Sun
    r_sun_mag = torch.norm(r_sun, dim=-1, keepdim=True)  # (B,1)
    e_sun = r_sun / r_sun_mag                             # (B,3)

    # Project satellite position onto Sun direction
    sat_proj = (r_sat * e_sun).sum(dim=-1)  # (B,) signed projection

    # Perpendicular distance from satellite to Earth-Sun line
    r_sat_parallel = sat_proj.unsqueeze(-1) * e_sun    # (B, 3)
    r_sat_perp = r_sat - r_sat_parallel                # (B, 3)
    perp_dist = torch.norm(r_sat_perp, dim=-1)         # (B,)

    # Shadow condition: satellite is behind Earth (proj < 0) AND
    # within the Earth's cylindrical shadow (perp_dist < R_EARTH).
    in_shadow = (sat_proj < 0.0) & (perp_dist < R_EARTH)

    shadow = torch.where(
        in_shadow,
        torch.tensor(0.0, dtype=torch.float64, device=r_sat.device),
        torch.tensor(1.0, dtype=torch.float64, device=r_sat.device),
    )

    if single:
        shadow = shadow.squeeze(0)

    return shadow


def srp_acceleration(
    r: torch.Tensor,
    r_sun: torch.Tensor,
    cr: float,
    area_mass: float,
) -> torch.Tensor:
    """
    Compute solar radiation pressure acceleration in ECI.

    Parameters
    ----------
    r : torch.Tensor
        Satellite position in ECI [m]. Shape (3,) or (B, 3).
    r_sun : torch.Tensor
        Sun position in ECI [m]. Shape (3,) or (B, 3).
    cr : float
        Radiation pressure coefficient (dimensionless).
    area_mass : float
        Area-to-mass ratio [m^2/kg].

    Returns
    -------
    torch.Tensor
        SRP acceleration in ECI [m/s^2], same shape as r.
    """
    r = r.to(torch.float64)
    r_sun = r_sun.to(torch.float64)

    single = r.dim() == 1
    if single:
        r = r.unsqueeze(0)
    if r_sun.dim() == 1:
        r_sun = r_sun.unsqueeze(0)

    # Vector from satellite to Sun
    r_sat_to_sun = r_sun - r  # (B, 3)
    dist_sat_sun = torch.norm(r_sat_to_sun, dim=-1, keepdim=True)  # (B, 1)
    e_sat_sun = r_sat_to_sun / dist_sat_sun  # (B, 3) unit vector

    # Sun-Earth distance for flux scaling
    r_sun_mag = torch.norm(r_sun, dim=-1, keepdim=True)  # (B, 1)

    # Shadow factor
    shadow = cylindrical_shadow(r, r_sun)  # (B,)
    if shadow.dim() == 0:
        shadow = shadow.unsqueeze(0)

    # SRP acceleration magnitude scales as (AU / |r_sun|)^2
    flux_scale = (AU / r_sun_mag) ** 2  # (B, 1)

    # a_srp = -P_SUN * (AU/|r_sun|)^2 * cr * (A/m) * e_sat_to_sun * shadow
    a_srp = (
        -P_SUN
        * flux_scale                       # (B, 1)
        * cr
        * area_mass
        * e_sat_sun                        # (B, 3)
        * shadow.unsqueeze(-1)             # (B, 1)
    )

    if single:
        a_srp = a_srp.squeeze(0)

    return a_srp
