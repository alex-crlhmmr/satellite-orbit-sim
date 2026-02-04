"""
Orbital element conversions: Cartesian <-> Keplerian, anomaly solvers, ROE.

All functions use float64 tensors and support batched inputs.
"""

import torch
import math
from .constants import MU_EARTH, DEG2RAD
from .frames import perifocal_to_eci_matrix


def cartesian_to_keplerian(
    r: torch.Tensor,
    v: torch.Tensor,
    mu: float = MU_EARTH,
) -> dict:
    """
    Convert Cartesian state to Keplerian elements.

    Args:
        r: (3,) or (B, 3) position [m]
        v: (3,) or (B, 3) velocity [m/s]
        mu: gravitational parameter [m³/s²]

    Returns:
        dict with keys: a, e, i, raan, argp, nu (all tensors, angles in rad)
    """
    batched = r.dim() == 2
    if not batched:
        r = r.unsqueeze(0)
        v = v.unsqueeze(0)

    r_mag = torch.norm(r, dim=-1, keepdim=True)     # (B, 1)
    v_mag = torch.norm(v, dim=-1, keepdim=True)

    # Angular momentum
    h = torch.cross(r, v, dim=-1)                    # (B, 3)
    h_mag = torch.norm(h, dim=-1, keepdim=True)

    # Node vector
    k_hat = torch.zeros_like(r)
    k_hat[..., 2] = 1.0
    n = torch.cross(k_hat, h, dim=-1)                # (B, 3)
    n_mag = torch.norm(n, dim=-1, keepdim=True)

    # Eccentricity vector
    e_vec = ((v_mag**2 - mu / r_mag) * r - (r * v).sum(dim=-1, keepdim=True) * v) / mu
    e = torch.norm(e_vec, dim=-1)                     # (B,)

    # Semi-major axis (handles e≈1 edge case)
    energy = v_mag.squeeze(-1)**2 / 2.0 - mu / r_mag.squeeze(-1)
    a = -mu / (2.0 * energy)                          # (B,)

    # Inclination
    inc = torch.acos(torch.clamp(h[..., 2:3] / h_mag, -1.0, 1.0)).squeeze(-1)  # (B,)

    # RAAN
    n_mag_safe = torch.clamp(n_mag, min=1e-30)
    raan = torch.acos(torch.clamp(n[..., 0:1] / n_mag_safe, -1.0, 1.0)).squeeze(-1)
    # Quadrant check: if n_y < 0, RAAN = 2π - RAAN
    raan = torch.where(n[..., 1] < 0, 2.0 * math.pi - raan, raan)

    # Handle near-zero inclination (equatorial): set RAAN = 0
    equatorial = inc < 1e-10
    raan = torch.where(equatorial, torch.zeros_like(raan), raan)

    # Argument of periapsis
    e_safe = torch.clamp(e, min=1e-30)
    cos_argp = (n * e_vec).sum(dim=-1) / (n_mag.squeeze(-1) * e_safe)
    cos_argp = torch.clamp(cos_argp, -1.0, 1.0)
    argp = torch.acos(cos_argp)
    # Quadrant check: if e_z < 0, argp = 2π - argp
    argp = torch.where(e_vec[..., 2] < 0, 2.0 * math.pi - argp, argp)

    # Handle near-circular (e≈0): argp = 0
    circular = e < 1e-10
    argp = torch.where(circular, torch.zeros_like(argp), argp)

    # Handle equatorial: use longitude of periapsis
    cos_argp_eq = e_vec[..., 0] / e_safe
    cos_argp_eq = torch.clamp(cos_argp_eq, -1.0, 1.0)
    argp_eq = torch.acos(cos_argp_eq)
    argp_eq = torch.where(e_vec[..., 1] < 0, 2.0 * math.pi - argp_eq, argp_eq)
    argp = torch.where(equatorial & ~circular, argp_eq, argp)

    # True anomaly
    cos_nu = (e_vec * r).sum(dim=-1) / (e_safe * r_mag.squeeze(-1))
    cos_nu = torch.clamp(cos_nu, -1.0, 1.0)
    nu = torch.acos(cos_nu)
    # Quadrant check: if r·v < 0, nu = 2π - nu
    rdotv = (r * v).sum(dim=-1)
    nu = torch.where(rdotv < 0, 2.0 * math.pi - nu, nu)

    # Handle circular: use argument of latitude
    cos_u = (n * r).sum(dim=-1) / (n_mag.squeeze(-1) * r_mag.squeeze(-1))
    cos_u = torch.clamp(cos_u, -1.0, 1.0)
    u = torch.acos(cos_u)
    u = torch.where(r[..., 2] < 0, 2.0 * math.pi - u, u)
    nu = torch.where(circular & ~equatorial, u, nu)

    # Handle circular-equatorial: use true longitude
    cos_l = r[..., 0] / r_mag.squeeze(-1)
    cos_l = torch.clamp(cos_l, -1.0, 1.0)
    l = torch.acos(cos_l)
    l = torch.where(r[..., 1] < 0, 2.0 * math.pi - l, l)
    nu = torch.where(circular & equatorial, l, nu)

    result = {"a": a, "e": e, "i": inc, "raan": raan, "argp": argp, "nu": nu}

    if not batched:
        result = {k: v_.squeeze(0) for k, v_ in result.items()}

    return result


def keplerian_to_cartesian(
    a: torch.Tensor,
    e: torch.Tensor,
    i: torch.Tensor,
    raan: torch.Tensor,
    argp: torch.Tensor,
    nu: torch.Tensor,
    mu: float = MU_EARTH,
) -> tuple:
    """
    Convert Keplerian elements to Cartesian state.

    All inputs are tensors (scalar or batched).

    Returns:
        (r, v): position (3,) or (B,3) [m], velocity (3,) or (B,3) [m/s]
    """
    # Semi-latus rectum
    p = a * (1.0 - e**2)

    # Position and velocity in perifocal frame
    r_mag = p / (1.0 + e * torch.cos(nu))
    r_pqw_x = r_mag * torch.cos(nu)
    r_pqw_y = r_mag * torch.sin(nu)

    sqrt_mu_p = torch.sqrt(mu / p)
    v_pqw_x = -sqrt_mu_p * torch.sin(nu)
    v_pqw_y = sqrt_mu_p * (e + torch.cos(nu))

    batched = a.dim() >= 1 and a.shape[0] > 1

    if batched:
        r_pqw = torch.stack([r_pqw_x, r_pqw_y, torch.zeros_like(r_pqw_x)], dim=-1)
        v_pqw = torch.stack([v_pqw_x, v_pqw_y, torch.zeros_like(v_pqw_x)], dim=-1)
    else:
        r_pqw = torch.stack([r_pqw_x, r_pqw_y, torch.zeros_like(r_pqw_x)], dim=-1)
        v_pqw = torch.stack([v_pqw_x, v_pqw_y, torch.zeros_like(v_pqw_x)], dim=-1)

    # Rotation to ECI
    R = perifocal_to_eci_matrix(raan, i, argp)

    if batched:
        r_eci = (R @ r_pqw.unsqueeze(-1)).squeeze(-1)
        v_eci = (R @ v_pqw.unsqueeze(-1)).squeeze(-1)
    else:
        r_eci = R @ r_pqw
        v_eci = R @ v_pqw

    return r_eci, v_eci


def true_to_eccentric_anomaly(nu: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Convert true anomaly to eccentric anomaly [rad]."""
    E = 2.0 * torch.atan2(
        torch.sqrt(1.0 - e) * torch.sin(nu / 2.0),
        torch.sqrt(1.0 + e) * torch.cos(nu / 2.0),
    )
    return E % (2.0 * math.pi)


def eccentric_to_mean_anomaly(E: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Convert eccentric anomaly to mean anomaly [rad] (Kepler's equation)."""
    M = E - e * torch.sin(E)
    return M % (2.0 * math.pi)


def mean_to_eccentric_anomaly(
    M: torch.Tensor,
    e: torch.Tensor,
    tol: float = 1e-12,
    max_iter: int = 50,
) -> torch.Tensor:
    """
    Solve Kepler's equation M = E - e*sin(E) via Newton-Raphson.
    """
    # Initial guess
    E = M.clone()
    for _ in range(max_iter):
        f = E - e * torch.sin(E) - M
        fp = 1.0 - e * torch.cos(E)
        dE = f / fp
        E = E - dE
        if torch.all(torch.abs(dE) < tol):
            break
    return E % (2.0 * math.pi)


def eccentric_to_true_anomaly(E: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Convert eccentric anomaly to true anomaly [rad]."""
    nu = 2.0 * torch.atan2(
        torch.sqrt(1.0 + e) * torch.sin(E / 2.0),
        torch.sqrt(1.0 - e) * torch.cos(E / 2.0),
    )
    return nu % (2.0 * math.pi)


def true_to_mean_anomaly(nu: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Convert true anomaly to mean anomaly [rad]."""
    E = true_to_eccentric_anomaly(nu, e)
    return eccentric_to_mean_anomaly(E, e)


def mean_to_true_anomaly(M: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """Convert mean anomaly to true anomaly [rad]."""
    E = mean_to_eccentric_anomaly(M, e)
    return eccentric_to_true_anomaly(E, e)


def cartesian_to_roe(
    r_chief: torch.Tensor,
    v_chief: torch.Tensor,
    r_deputy: torch.Tensor,
    v_deputy: torch.Tensor,
    mu: float = MU_EARTH,
) -> torch.Tensor:
    """
    Compute quasi-nonsingular relative orbital elements (ROE).

    D'Amico formulation: δα = (δa, δλ, δex, δey, δix, δiy)

    where:
        δa  = (a_d - a_c) / a_c
        δλ  = (u_d - u_c) + (Ω_d - Ω_c) * cos(i_c)
        δex = e_d*cos(ω_d) - e_c*cos(ω_c)
        δey = e_d*sin(ω_d) - e_c*sin(ω_c)
        δix = i_d - i_c
        δiy = (Ω_d - Ω_c) * sin(i_c)

    Args:
        r_chief, v_chief: chief position/velocity (3,) or (B, 3) [m, m/s]
        r_deputy, v_deputy: deputy position/velocity (3,) or (B, 3) [m, m/s]

    Returns:
        roe: (6,) or (B, 6) quasi-nonsingular ROE (dimensionless except δλ in rad)
    """
    oe_c = cartesian_to_keplerian(r_chief, v_chief, mu)
    oe_d = cartesian_to_keplerian(r_deputy, v_deputy, mu)

    a_c, e_c, i_c = oe_c["a"], oe_c["e"], oe_c["i"]
    raan_c, argp_c, nu_c = oe_c["raan"], oe_c["argp"], oe_c["nu"]
    a_d, e_d, i_d = oe_d["a"], oe_d["e"], oe_d["i"]
    raan_d, argp_d, nu_d = oe_d["raan"], oe_d["argp"], oe_d["nu"]

    # Argument of latitude u = ω + ν
    u_c = argp_c + nu_c
    u_d = argp_d + nu_d

    # Relative elements
    da = (a_d - a_c) / a_c
    dl = _wrap_angle(u_d - u_c + (raan_d - raan_c) * torch.cos(i_c))
    dex = e_d * torch.cos(argp_d) - e_c * torch.cos(argp_c)
    dey = e_d * torch.sin(argp_d) - e_c * torch.sin(argp_c)
    dix = i_d - i_c
    diy = (raan_d - raan_c) * torch.sin(i_c)

    roe = torch.stack([da, dl, dex, dey, dix, diy], dim=-1)
    return roe


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle to [-π, π]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
