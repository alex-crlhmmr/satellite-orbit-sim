"""
Exponential atmosphere drag model with 28 altitude bands (US Standard 1976).

Provides atmospheric density lookup and drag acceleration computation
for satellites in low-Earth orbit. All tensors use float64.
"""

import torch
import math

from .constants import ATMOSPHERE_BANDS, OMEGA_EARTH, R_EARTH, MU_EARTH


# Pre-compute band boundaries as plain lists for fast lookup.
# Each entry: (base_alt_m, base_density, scale_height_m)
_BAND_BASE_ALT_M = []
_BAND_BASE_DENSITY = []
_BAND_SCALE_HEIGHT_M = []

for _alt_km, _rho, _H_km in ATMOSPHERE_BANDS:
    _BAND_BASE_ALT_M.append(_alt_km * 1000.0)
    _BAND_BASE_DENSITY.append(_rho)
    _BAND_SCALE_HEIGHT_M.append(_H_km * 1000.0)

_NUM_BANDS = len(ATMOSPHERE_BANDS)
_MAX_ALT_M = 1000.0 * 1000.0  # 1000 km upper clamp


def atmospheric_density(altitude_m: torch.Tensor) -> torch.Tensor:
    """
    Compute atmospheric density using the 28-band US Standard 1976 model.

    Parameters
    ----------
    altitude_m : torch.Tensor
        Geometric altitude in meters. Arbitrary shape (scalar or batched).

    Returns
    -------
    torch.Tensor
        Density in kg/m^3, same shape as input.
    """
    altitude_m = altitude_m.to(torch.float64)
    # Clamp altitude to valid range [0, 1000 km]
    h = altitude_m.clamp(min=0.0, max=_MAX_ALT_M)

    # Build tensors from the band data for vectorised lookup
    base_alts = torch.tensor(_BAND_BASE_ALT_M, dtype=torch.float64,
                             device=altitude_m.device)       # (_NUM_BANDS,)
    base_densities = torch.tensor(_BAND_BASE_DENSITY, dtype=torch.float64,
                                  device=altitude_m.device)  # (_NUM_BANDS,)
    scale_heights = torch.tensor(_BAND_SCALE_HEIGHT_M, dtype=torch.float64,
                                 device=altitude_m.device)   # (_NUM_BANDS,)

    # For each altitude value find the band index: largest base_alt <= h.
    # h_flat: (N,), base_alts: (B,) -> compare (N, B)
    orig_shape = h.shape
    h_flat = h.reshape(-1)  # (N,)

    # (N, _NUM_BANDS): True where base_alt <= h
    mask = h_flat.unsqueeze(-1) >= base_alts.unsqueeze(0)  # (N, NUM_BANDS)

    # Sum along band axis gives count of bands with base_alt <= h;
    # subtract 1 to get index of the highest qualifying band.
    band_idx = mask.sum(dim=-1) - 1  # (N,)
    band_idx = band_idx.clamp(min=0, max=_NUM_BANDS - 1)

    h_base = base_alts[band_idx]          # (N,)
    rho_base = base_densities[band_idx]   # (N,)
    H = scale_heights[band_idx]           # (N,)

    rho = rho_base * torch.exp(-(h_flat - h_base) / H)

    return rho.reshape(orig_shape)


def drag_acceleration(
    r: torch.Tensor,
    v: torch.Tensor,
    cd: float,
    area_mass: float,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    omega: float = OMEGA_EARTH,
) -> torch.Tensor:
    """
    Compute atmospheric drag acceleration in the ECI frame.

    Parameters
    ----------
    r : torch.Tensor
        Position vector in ECI [m]. Shape (3,) or (B, 3).
    v : torch.Tensor
        Velocity vector in ECI [m/s]. Shape (3,) or (B, 3).
    cd : float
        Drag coefficient (dimensionless).
    area_mass : float
        Area-to-mass ratio [m^2/kg].
    mu : float
        Gravitational parameter [m^3/s^2] (unused, kept for API compat).
    re : float
        Earth equatorial radius [m].
    omega : float
        Earth rotation rate [rad/s].

    Returns
    -------
    torch.Tensor
        Drag acceleration in ECI [m/s^2], same shape as r.
    """
    r = r.to(torch.float64)
    v = v.to(torch.float64)

    single = r.dim() == 1
    if single:
        r = r.unsqueeze(0)  # (1, 3)
        v = v.unsqueeze(0)

    # Altitude = |r| - R_Earth
    r_mag = torch.norm(r, dim=-1)  # (B,)
    altitude = r_mag - re          # (B,)

    # Atmospheric density
    rho = atmospheric_density(altitude)  # (B,)

    # Relative velocity: subtract Earth rotation contribution.
    # omega_vec = [0, 0, omega]  =>  omega x r = [-omega*ry, omega*rx, 0]
    omega_cross_r = torch.zeros_like(r)
    omega_cross_r[..., 0] = -omega * r[..., 1]
    omega_cross_r[..., 1] =  omega * r[..., 0]
    # z-component is zero

    v_rel = v - omega_cross_r  # (B, 3)
    v_rel_mag = torch.norm(v_rel, dim=-1, keepdim=True)  # (B, 1)

    # Drag acceleration: a = -0.5 * rho * Cd * (A/m) * |v_rel| * v_rel
    a_drag = (
        -0.5
        * rho.unsqueeze(-1)        # (B, 1)
        * cd
        * area_mass
        * v_rel_mag                 # (B, 1)
        * v_rel                     # (B, 3)
    )

    if single:
        a_drag = a_drag.squeeze(0)

    return a_drag
