"""Sequential estimation of an effective atmospheric drag scale.

The state is intentionally called an *effective* scale: without independent
attitude/aerodynamic information, tracking data observe density times CdA/m.
The scalar random-walk filter exposes that uncertainty instead of folding a
best-fit correction invisibly into spacecraft area.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DensityScaleEstimate:
    epoch_s: float
    scale: float
    variance: float
    innovation: float | None = None
    innovation_variance: float | None = None

    @property
    def sigma(self) -> float:
        return math.sqrt(self.variance)


class DensityScaleFilter:
    """Scalar Kalman filter for a random-walk effective drag scale.

    Measurements follow ``observed_effect = sensitivity * scale + noise``.
    Sensitivity can come from a variational equation, finite-difference
    propagation, accelerometer model, or along-track orbit residual.
    """

    def __init__(self, initial_scale: float = 1.0,
                 initial_variance: float = 1.0,
                 process_noise_psd: float = 0.0,
                 minimum_scale: float = 0.0) -> None:
        if initial_variance < 0 or process_noise_psd < 0:
            raise ValueError("variances and process-noise PSD must be nonnegative")
        if initial_scale < minimum_scale:
            raise ValueError("initial scale is below minimum_scale")
        self.scale = float(initial_scale)
        self.variance = float(initial_variance)
        self.process_noise_psd = float(process_noise_psd)
        self.minimum_scale = float(minimum_scale)
        self.epoch_s = 0.0

    def predict(self, epoch_s: float) -> DensityScaleEstimate:
        epoch_s = float(epoch_s)
        dt = epoch_s - self.epoch_s
        if dt < 0:
            raise ValueError("density filter time must be monotonic")
        self.variance += self.process_noise_psd * dt
        self.epoch_s = epoch_s
        return DensityScaleEstimate(self.epoch_s, self.scale, self.variance)

    def update(self, observed_effect: float, sensitivity: float,
               measurement_variance: float) -> DensityScaleEstimate:
        observed_effect = float(observed_effect)
        sensitivity = float(sensitivity)
        measurement_variance = float(measurement_variance)
        if measurement_variance <= 0:
            raise ValueError("measurement_variance must be positive")
        innovation = observed_effect - sensitivity * self.scale
        innovation_variance = sensitivity * sensitivity * self.variance + measurement_variance
        gain = self.variance * sensitivity / innovation_variance
        self.scale = max(self.minimum_scale, self.scale + gain * innovation)
        # Joseph stabilized scalar covariance update.
        residual_gain = 1.0 - gain * sensitivity
        self.variance = (residual_gain * residual_gain * self.variance +
                         gain * gain * measurement_variance)
        return DensityScaleEstimate(
            self.epoch_s, self.scale, self.variance,
            innovation, innovation_variance,
        )

    def normalized_innovation_squared(self, estimate: DensityScaleEstimate) -> float:
        if estimate.innovation is None or estimate.innovation_variance is None:
            raise ValueError("estimate has no measurement innovation")
        return estimate.innovation * estimate.innovation / estimate.innovation_variance
