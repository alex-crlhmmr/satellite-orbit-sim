"""
Reference frame transformations: ECI, ECEF, RTN, perifocal.

All functions support batched (B, 3) tensors using float64.
"""

import math
from datetime import datetime

import torch

from .constants import DAYS_PER_CENTURY, DEG2RAD, JD_J2000, SECONDS_PER_DAY


def datetime_to_jd(dt: datetime) -> float:
    """Convert a Python datetime to Julian Date."""
    y = dt.year
    m = dt.month
    d = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0
                  + dt.microsecond / 3.6e9) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    A = int(y / 100)
    B = 2 - A + int(A / 4)
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + B - 1524.5


def jd_to_centuries_j2000(jd: float) -> float:
    """Convert Julian Date to centuries since J2000.0."""
    return (jd - JD_J2000) / DAYS_PER_CENTURY


def gmst_from_jd(jd: float) -> float:
    """
    Greenwich Mean Sidereal Time from Julian Date.
    Returns angle in radians [0, 2π).
    IAU 1982 model.
    """
    T = jd_to_centuries_j2000(jd)
    # GMST in seconds of time
    gmst_sec = (67310.54841
                + (876600.0 * 3600.0 + 8640184.812866) * T
                + 0.093104 * T**2
                - 6.2e-6 * T**3)
    # Convert to radians
    gmst_rad = (gmst_sec / 240.0) * DEG2RAD  # 1 sec of time = 1/240 degree
    return gmst_rad % (2.0 * math.pi)


def gmst_from_seconds(epoch_jd: float, elapsed_seconds: float) -> float:
    """GMST at epoch_jd + elapsed_seconds."""
    jd = epoch_jd + elapsed_seconds / SECONDS_PER_DAY
    return gmst_from_jd(jd)


def eci_to_ecef(r_eci: torch.Tensor, gmst: float) -> torch.Tensor:
    """
    Rotate ECI position(s) to ECEF.

    Args:
        r_eci: (3,) or (B, 3) position in ECI [m]
        gmst: Greenwich Mean Sidereal Time [rad]

    Returns:
        r_ecef: same shape as input, in ECEF [m]
    """
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)
    device = r_eci.device
    dtype = r_eci.dtype

    R = torch.tensor([
        [cos_g,  sin_g, 0.0],
        [-sin_g, cos_g, 0.0],
        [0.0,    0.0,   1.0],
    ], device=device, dtype=dtype)

    if r_eci.dim() == 1:
        return R @ r_eci
    else:
        return (R @ r_eci.unsqueeze(-1)).squeeze(-1)


def ecef_to_eci(r_ecef: torch.Tensor, gmst: float) -> torch.Tensor:
    """Rotate ECEF position(s) to ECI (inverse of eci_to_ecef)."""
    cos_g = math.cos(gmst)
    sin_g = math.sin(gmst)
    device = r_ecef.device
    dtype = r_ecef.dtype

    R = torch.tensor([
        [cos_g, -sin_g, 0.0],
        [sin_g,  cos_g, 0.0],
        [0.0,    0.0,   1.0],
    ], device=device, dtype=dtype)

    if r_ecef.dim() == 1:
        return R @ r_ecef
    else:
        return (R @ r_ecef.unsqueeze(-1)).squeeze(-1)


def eci_to_rtn(r_eci: torch.Tensor, v_eci: torch.Tensor) -> torch.Tensor:
    """
    Compute the ECI->RTN (radial, along-track, cross-track) DCM.

    Args:
        r_eci: (3,) or (B, 3) position [m]
        v_eci: (3,) or (B, 3) velocity [m/s]

    Returns:
        DCM: (3, 3) or (B, 3, 3) rotation matrix from ECI to RTN
    """
    batched = r_eci.dim() == 2
    if not batched:
        r_eci = r_eci.unsqueeze(0)
        v_eci = v_eci.unsqueeze(0)

    # R = r_hat (radial)
    r_hat = r_eci / torch.norm(r_eci, dim=-1, keepdim=True)

    # N = h_hat (cross-track, angular momentum direction)
    h = torch.cross(r_eci, v_eci, dim=-1)
    n_hat = h / torch.norm(h, dim=-1, keepdim=True)

    # T = N x R (along-track)
    t_hat = torch.cross(n_hat, r_hat, dim=-1)

    # DCM rows are R, T, N
    dcm = torch.stack([r_hat, t_hat, n_hat], dim=-2)  # (B, 3, 3)

    if not batched:
        dcm = dcm.squeeze(0)
    return dcm


def rtn_to_eci(r_eci: torch.Tensor, v_eci: torch.Tensor) -> torch.Tensor:
    """RTN->ECI DCM (transpose of ECI->RTN)."""
    dcm = eci_to_rtn(r_eci, v_eci)
    if dcm.dim() == 2:
        return dcm.T
    return dcm.transpose(-2, -1)


def perifocal_to_eci_matrix(
    raan: torch.Tensor,
    inc: torch.Tensor,
    argp: torch.Tensor,
) -> torch.Tensor:
    """
    Rotation matrix from perifocal (PQW) frame to ECI.

    Args:
        raan: Right ascension of ascending node [rad] — scalar or (B,)
        inc:  Inclination [rad]
        argp: Argument of periapsis [rad]

    Returns:
        R: (3, 3) or (B, 3, 3)
    """
    batched = raan.dim() >= 1 and raan.shape[0] > 1
    cos_O = torch.cos(raan)
    sin_O = torch.sin(raan)
    cos_i = torch.cos(inc)
    sin_i = torch.sin(inc)
    cos_w = torch.cos(argp)
    sin_w = torch.sin(argp)

    if not batched:
        R = torch.zeros(3, 3, device=raan.device, dtype=raan.dtype)
        R[0, 0] = cos_O * cos_w - sin_O * sin_w * cos_i
        R[0, 1] = -cos_O * sin_w - sin_O * cos_w * cos_i
        R[0, 2] = sin_O * sin_i
        R[1, 0] = sin_O * cos_w + cos_O * sin_w * cos_i
        R[1, 1] = -sin_O * sin_w + cos_O * cos_w * cos_i
        R[1, 2] = -cos_O * sin_i
        R[2, 0] = sin_w * sin_i
        R[2, 1] = cos_w * sin_i
        R[2, 2] = cos_i
    else:
        B = raan.shape[0]
        R = torch.zeros(B, 3, 3, device=raan.device, dtype=raan.dtype)
        R[:, 0, 0] = cos_O * cos_w - sin_O * sin_w * cos_i
        R[:, 0, 1] = -cos_O * sin_w - sin_O * cos_w * cos_i
        R[:, 0, 2] = sin_O * sin_i
        R[:, 1, 0] = sin_O * cos_w + cos_O * sin_w * cos_i
        R[:, 1, 1] = -sin_O * sin_w + cos_O * cos_w * cos_i
        R[:, 1, 2] = -cos_O * sin_i
        R[:, 2, 0] = sin_w * sin_i
        R[:, 2, 1] = cos_w * sin_i
        R[:, 2, 2] = cos_i
    return R
