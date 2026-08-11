"""Tests for data-calibratable covariance process noise."""

import numpy as np
import pytest

from core.uncertainty import rtn_basis, white_acceleration_process_noise

STATE = np.array([7.0e6, 1.0e5, -2.0e5, -100.0, 7500.0, 500.0])


def test_rtn_basis_is_right_handed_orthonormal():
    basis = rtn_basis(STATE)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    np.testing.assert_allclose(np.cross(basis[:, 0], basis[:, 1]), basis[:, 2], atol=1e-14)


def test_white_acceleration_process_noise_is_symmetric_psd():
    q = white_acceleration_process_noise(STATE, 60.0, np.array([1e-12, 4e-12, 2e-12]))
    np.testing.assert_allclose(q, q.T, atol=1e-20)
    assert np.linalg.eigvalsh(q).min() >= -1e-18
    assert q.shape == (6, 6)


def test_zero_process_noise_is_exactly_zero():
    q = white_acceleration_process_noise(STATE, 60.0, np.zeros(3))
    np.testing.assert_array_equal(q, np.zeros((6, 6)))


@pytest.mark.parametrize("duration,psd", [(-1.0, [1, 1, 1]), (1.0, [-1, 0, 0]), (1.0, [1, 2])])
def test_invalid_process_noise_rejected(duration, psd):
    with pytest.raises(ValueError):
        white_acceleration_process_noise(STATE, duration, np.asarray(psd))

