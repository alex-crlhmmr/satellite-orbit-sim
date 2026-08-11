"""Authoritative Earth orientation and solar ephemeris for visualization."""

from dataclasses import dataclass

import brahe as bh
import numpy as np


@dataclass(frozen=True)
class SceneEphemeris:
    """Geometry required to render an Earth-fixed body in GCRF."""

    itrf_to_gcrf: np.ndarray
    sun_position_gcrf_m: np.ndarray


_eop_initialized = False


def scene_ephemeris(epoch_jd: float) -> SceneEphemeris:
    """Return full IERS Earth orientation and DE440s solar position."""
    global _eop_initialized
    if not _eop_initialized:
        bh.initialize_eop()
        _eop_initialized = True
    epoch = bh.Epoch.from_jd(float(epoch_jd), bh.TimeSystem.UTC)
    return SceneEphemeris(
        itrf_to_gcrf=np.asarray(bh.rotation_itrf_to_gcrf(epoch), dtype=np.float64),
        sun_position_gcrf_m=np.asarray(
            bh.sun_position_spice(epoch, bh.EphemerisSource.DE440s), dtype=np.float64
        ),
    )
