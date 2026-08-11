"""Tests for effective density/ballistic scale estimation."""

import numpy as np
import pytest

from core.density_estimation import DensityScaleFilter


def test_density_scale_filter_converges_and_reports_consistent_uncertainty():
    rng = np.random.default_rng(20240811)
    truth, measurement_sigma = 1.35, 0.08
    estimator = DensityScaleFilter(initial_scale=1.0, initial_variance=0.5**2,
                                   process_noise_psd=1e-7)
    normalized_innovations = []
    for index in range(200):
        estimator.predict(index * 60.0)
        sensitivity = 2.0 + 0.5 * np.sin(index / 10.0)
        observed = sensitivity * truth + rng.normal(0.0, measurement_sigma)
        result = estimator.update(observed, sensitivity, measurement_sigma**2)
        normalized_innovations.append(estimator.normalized_innovation_squared(result))
    assert abs(estimator.scale - truth) < 3.0 * result.sigma
    assert result.sigma < 0.03
    # A calibrated scalar innovation has E[NIS]=1; finite deterministic sample.
    assert 0.5 < np.mean(normalized_innovations[20:]) < 1.5


def test_density_scale_filter_random_walk_inflates_uncertainty():
    estimator = DensityScaleFilter(initial_variance=0.04, process_noise_psd=2e-4)
    estimate = estimator.predict(100.0)
    assert estimate.variance == pytest.approx(0.06)
    with pytest.raises(ValueError, match="monotonic"):
        estimator.predict(99.0)


def test_density_scale_filter_rejects_invalid_uncertainty():
    with pytest.raises(ValueError, match="nonnegative"):
        DensityScaleFilter(initial_variance=-1.0)
    estimator = DensityScaleFilter()
    with pytest.raises(ValueError, match="positive"):
        estimator.update(1.0, 1.0, 0.0)
