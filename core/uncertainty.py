"""Uncertainty models used by high-fidelity covariance propagation."""

from __future__ import annotations

import numpy as np


def rtn_basis(state_gcrf: np.ndarray) -> np.ndarray:
    """Return the RTN-to-GCRF direction cosine matrix for a Cartesian state."""
    state = np.asarray(state_gcrf, dtype=np.float64)
    if state.shape != (6,):
        raise ValueError("state_gcrf must have shape (6,)")
    radial = state[:3] / np.linalg.norm(state[:3])
    normal = np.cross(state[:3], state[3:])
    normal_norm = np.linalg.norm(normal)
    if normal_norm == 0:
        raise ValueError("position and velocity cannot be collinear")
    normal /= normal_norm
    transverse = np.cross(normal, radial)
    return np.column_stack((radial, transverse, normal))


def white_acceleration_process_noise(
    state_gcrf: np.ndarray,
    duration_s: float,
    acceleration_psd_rtn: np.ndarray,
) -> np.ndarray:
    """Discretize independent continuous white RTN accelerations.

    ``acceleration_psd_rtn`` contains one-sided acceleration power spectral
    densities in m²/s³ for radial, transverse, and normal axes. The returned
    6x6 covariance is in GCRF Cartesian position/velocity ordering.
    """
    dt = float(duration_s)
    psd = np.asarray(acceleration_psd_rtn, dtype=np.float64)
    if dt < 0 or psd.shape != (3,) or np.any(psd < 0) or not np.isfinite(psd).all():
        raise ValueError("duration must be nonnegative and RTN PSD must be three finite nonnegative values")
    spectral = np.diag(psd)
    q_rtn = np.block([
        [spectral * dt**3 / 3.0, spectral * dt**2 / 2.0],
        [spectral * dt**2 / 2.0, spectral * dt],
    ])
    basis = rtn_basis(state_gcrf)
    transform = np.block([[basis, np.zeros((3, 3))], [np.zeros((3, 3)), basis]])
    q_gcrf = transform @ q_rtn @ transform.T
    return 0.5 * (q_gcrf + q_gcrf.T)

