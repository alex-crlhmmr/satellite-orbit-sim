"""
Atmosphere models and drag acceleration.

Two atmosphere implementations live here:

* ``USStd1976Atmosphere``  — exponential 28-band US Standard 1976 (legacy default).
* ``NRLMSISE2Atmosphere``  — official MSIS 2.x via pymsis (date / solar-flux /
  geomagnetic dependent, NRLMSISE-00 successor).

Both expose a ``density(r_eci, jd) -> float`` method (numpy r, float jd).
``make_atmosphere(cfg)`` builds the right one from a config dict.

The standalone helpers ``atmospheric_density`` (altitude-only) and the
internal ``_density_scalar/_density_array`` remain so the test suite and
any plain-altitude callers keep working.

Internal hot path is pure NumPy float64. ``atmospheric_density`` accepts
torch tensors for backward compatibility with the test suite.
"""

import math
from datetime import datetime, timedelta

import numpy as np
import torch

from .constants import (
    ATMOSPHERE_BANDS,
    FLATTENING,
    MU_EARTH,
    OMEGA_EARTH,
    R_EARTH,
)

_BAND_BASE_ALT_M = np.array(
    [_alt_km * 1000.0 for _alt_km, _, _ in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_BAND_BASE_DENSITY = np.array(
    [_rho for _, _rho, _ in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_BAND_SCALE_HEIGHT_M = np.array(
    [_H_km * 1000.0 for _, _, _H_km in ATMOSPHERE_BANDS],
    dtype=np.float64,
)
_NUM_BANDS = len(ATMOSPHERE_BANDS)
_MAX_ALT_M = 1000.0 * 1000.0


def _density_scalar(altitude_m: float) -> float:
    """Scalar fast path — used in the Propagator hot loop."""
    h = max(0.0, min(_MAX_ALT_M, altitude_m))
    # searchsorted returns the insertion index; we want the largest base_alt <= h,
    # so subtract one. Clamp into [0, _NUM_BANDS - 1].
    idx = int(np.searchsorted(_BAND_BASE_ALT_M, h, side="right")) - 1
    if idx < 0:
        idx = 0
    elif idx >= _NUM_BANDS:
        idx = _NUM_BANDS - 1
    h_base = _BAND_BASE_ALT_M[idx]
    rho_base = _BAND_BASE_DENSITY[idx]
    H = _BAND_SCALE_HEIGHT_M[idx]
    return rho_base * np.exp(-(h - h_base) / H)


def _density_array(altitude_m: np.ndarray) -> np.ndarray:
    """Vectorised numpy version for batched altitude arrays."""
    h = np.clip(altitude_m, 0.0, _MAX_ALT_M)
    idx = np.searchsorted(_BAND_BASE_ALT_M, h, side="right") - 1
    idx = np.clip(idx, 0, _NUM_BANDS - 1)
    h_base = _BAND_BASE_ALT_M[idx]
    rho_base = _BAND_BASE_DENSITY[idx]
    H = _BAND_SCALE_HEIGHT_M[idx]
    return rho_base * np.exp(-(h - h_base) / H)


def atmospheric_density(altitude_m):
    """
    Compute atmospheric density using the 28-band US Standard 1976 model.

    Accepts either a torch.Tensor or numpy.ndarray (or scalar). Returns
    the same type as the input. The Propagator hot path goes through
    the internal _density_scalar/_density_array helpers directly.
    """
    if isinstance(altitude_m, torch.Tensor):
        alt_np = altitude_m.detach().cpu().numpy().astype(np.float64, copy=False)
        if alt_np.ndim == 0:
            rho = np.float64(_density_scalar(float(alt_np)))
        else:
            rho = _density_array(alt_np)
        return torch.from_numpy(np.asarray(rho)).to(
            device=altitude_m.device, dtype=torch.float64
        ).reshape(altitude_m.shape)
    arr = np.asarray(altitude_m, dtype=np.float64)
    if arr.ndim == 0:
        return _density_scalar(float(arr))
    return _density_array(arr)


def drag_acceleration(
    r: np.ndarray,
    v: np.ndarray,
    cd: float,
    area_mass: float,
    mu: float = MU_EARTH,
    re: float = R_EARTH,
    omega: float = OMEGA_EARTH,
    atmosphere=None,
    jd: float = None,
) -> np.ndarray:
    """
    Atmospheric drag acceleration in ECI [m/s²].

    If ``atmosphere`` is provided, density is queried via
    ``atmosphere.density(r, jd)`` (full position / time / space-weather
    dependent). Otherwise the legacy altitude-only US Std 1976 model is
    used. Supports single (3,) and batched (B, 3) numpy arrays; the
    batched path always uses the altitude-only fallback.
    """
    if r.ndim == 1:
        if atmosphere is not None:
            rho = atmosphere.density(r, jd)
        else:
            r_mag = math.sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2])
            rho = _density_scalar(r_mag - re)
        wind = (atmosphere.wind_eci(r, jd)
                if atmosphere is not None and hasattr(atmosphere, "wind_eci")
                else np.zeros(3, dtype=np.float64))
        vrel0 = v[0] + omega * r[1] - wind[0]
        vrel1 = v[1] - omega * r[0] - wind[1]
        vrel2 = v[2] - wind[2]
        vrel_mag = math.sqrt(vrel0 * vrel0 + vrel1 * vrel1 + vrel2 * vrel2)
        k = -0.5 * rho * cd * area_mass * vrel_mag
        return np.array([k * vrel0, k * vrel1, k * vrel2], dtype=np.float64)

    # Batched (used by tests / batched RL — keeps altitude-only path)
    r_mag = np.linalg.norm(r, axis=-1)
    rho = _density_array(r_mag - re)
    omega_cross_r = np.zeros_like(r)
    omega_cross_r[..., 0] = -omega * r[..., 1]
    omega_cross_r[..., 1] = omega * r[..., 0]
    v_rel = v - omega_cross_r
    v_rel_mag = np.linalg.norm(v_rel, axis=-1, keepdims=True)
    return (-0.5 * cd * area_mass) * rho[..., None] * v_rel_mag * v_rel


# ---------------------------------------------------------------------------
# Atmosphere model objects
# ---------------------------------------------------------------------------

# JD of 1970-01-01 00:00 UTC (Unix epoch).
_JD_UNIX_EPOCH = 2440587.5


def _jd_to_datetime_utc(jd: float) -> datetime:
    """Julian date -> naive datetime interpreted as UTC.

    Returns a *naive* datetime because numpy.datetime64 (which pymsis
    consumes) has no timezone representation. The value is in UTC.
    """
    seconds = (jd - _JD_UNIX_EPOCH) * 86400.0
    return datetime(1970, 1, 1) + timedelta(seconds=seconds)


class USStd1976Atmosphere:
    """Legacy altitude-only exponential model (US Standard 1976)."""

    name = "ussa76"

    def density(self, r_eci: np.ndarray, jd: float = None) -> float:
        alt = math.sqrt(r_eci[0] ** 2 + r_eci[1] ** 2 + r_eci[2] ** 2) - R_EARTH
        return _density_scalar(alt)

    def wind_eci(self, r_eci: np.ndarray, jd: float = None) -> np.ndarray:
        return np.zeros(3, dtype=np.float64)


class NRLMSISE2Atmosphere:
    """
    NRLMSISE-2 (a.k.a. MSIS 2.x) atmosphere via pymsis.

    Parameters
    ----------
    f107  : daily 10.7 cm solar radio flux [sfu], previous day. Default 150.
    f107a : 81-day-centered average F10.7 [sfu].                 Default 150.
    ap    : daily geomagnetic Ap index.                          Default 4.

    Geocentric latitude / longitude / altitude are derived from the ECI
    position and the current Julian date (GMST rotation around Z). For
    the purpose of atmospheric density this is accurate to << 1 km at
    LEO altitudes — far below the model's own uncertainty.
    """

    name = "nrlmsise2"

    def __init__(self, f107: float = 150.0, f107a: float = 150.0,
                 ap: float = 4.0, space_weather=None) -> None:
        from pymsis import calculate as _calc  # imported lazily

        self.f107 = float(f107)
        self.f107a = float(f107a)
        self.ap = float(ap)
        self.space_weather = space_weather
        self._calc = _calc

        # Pre-allocate the small arrays pymsis expects.
        self._f107_arr = np.array([self.f107], dtype=np.float64)
        self._f107a_arr = np.array([self.f107a], dtype=np.float64)
        self._ap_arr = np.array([[self.ap] * 7], dtype=np.float64)

    def density(self, r_eci: np.ndarray, jd: float) -> float:
        # ECI -> ECEF via GMST rotation about Z; then geocentric lat/lon.
        from .frames import gmst_from_jd

        gmst = gmst_from_jd(jd)
        cg = math.cos(gmst)
        sg = math.sin(gmst)
        x_e = cg * r_eci[0] + sg * r_eci[1]
        y_e = -sg * r_eci[0] + cg * r_eci[1]
        z_e = r_eci[2]
        lat_rad, alt_m = _ecef_to_geodetic(x_e, y_e, z_e)
        alt_km = alt_m / 1000.0
        lat_deg = math.degrees(lat_rad)
        lon_deg = math.degrees(math.atan2(y_e, x_e))

        instant = _jd_to_datetime_utc(jd)
        if self.space_weather is not None:
            record = self.space_weather.at(instant)
            self._f107_arr[0] = record.f107
            self._f107a_arr[0] = record.f107a
            self._ap_arr[0, :] = record.ap
        date = np.array([instant], dtype="datetime64[us]")
        out = self._calc(
            date,
            np.array([lon_deg], dtype=np.float64),
            np.array([lat_deg], dtype=np.float64),
            np.array([alt_km], dtype=np.float64),
            f107s=self._f107_arr,
            f107as=self._f107a_arr,
            aps=self._ap_arr,
        )
        rho = float(np.asarray(out[..., 0]).reshape(-1)[0])
        # NaN can occur if pymsis is asked for an out-of-range altitude
        # (e.g. r below surface during a bad initial state). Fall back to
        # USSA76 in that case rather than poisoning the integrator.
        if not math.isfinite(rho) or rho < 0.0:
            return _density_scalar(alt_km * 1000.0)
        return rho

    def wind_eci(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        # MSIS is a density/composition model and supplies no neutral winds.
        return np.zeros(3, dtype=np.float64)


class WindAdjustedAtmosphere:
    """Attach a specified Earth-fixed neutral-wind vector to a density model.

    This is primarily an interface and sensitivity-analysis tool. A constant
    vector is not a substitute for HWM or measured winds.
    """

    def __init__(self, base, wind_ecef_mps) -> None:
        wind = np.asarray(wind_ecef_mps, dtype=np.float64)
        if wind.shape != (3,) or not np.isfinite(wind).all():
            raise ValueError("wind_ecef_mps must be a finite three-vector")
        self.base = base
        self.wind_ecef_mps = wind
        self.name = f"{base.name}+specified_wind"

    def density(self, r_eci: np.ndarray, jd: float) -> float:
        return self.base.density(r_eci, jd)

    def wind_eci(self, r_eci: np.ndarray, jd: float) -> np.ndarray:
        from .frames import gmst_from_jd
        angle = gmst_from_jd(jd)
        cosine, sine = math.cos(angle), math.sin(angle)
        x, y, z = self.wind_ecef_mps
        return np.array([cosine * x - sine * y,
                         sine * x + cosine * y, z], dtype=np.float64)


def _ecef_to_geodetic(x: float, y: float, z: float) -> tuple[float, float]:
    """WGS-84 ECEF to geodetic latitude and ellipsoidal altitude."""
    a = R_EARTH
    f = FLATTENING
    e2 = f * (2.0 - f)
    p = math.hypot(x, y)
    if p < 1e-9:
        b = a * (1.0 - f)
        return math.copysign(math.pi / 2.0, z), abs(z) - b
    lat = math.atan2(z, p * (1.0 - e2))
    alt = 0.0
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        next_lat = math.atan2(z, p * (1.0 - e2 * n / (n + alt)))
        if abs(next_lat - lat) < 1e-14:
            lat = next_lat
            break
        lat = next_lat
    sin_lat = math.sin(lat)
    n = a / math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return lat, alt


def make_atmosphere(config: dict):
    """
    Build an atmosphere model from a config dict.

    Recognised keys (all optional):
        model:  'nrlmsise2' (default) | 'ussa76'
        f107:   daily F10.7 [sfu]                 (NRLMSISE only)
        f107a:  81-day mean F10.7 [sfu]           (NRLMSISE only)
        ap:     geomagnetic Ap index              (NRLMSISE only)
    """
    if not isinstance(config, dict):
        config = {}
    model = str(config.get("model", "nrlmsise2")).lower()
    result = None
    if model in ("ussa76", "us_std_1976", "usstd76", "exponential"):
        result = USStd1976Atmosphere()
    elif model in ("nrlmsise2", "nrlmsise", "msis", "msis2"):
        try:
            space_weather = None
            if "space_weather_file" in config:
                from .space_weather import SpaceWeatherSeries
                space_weather = SpaceWeatherSeries.from_csv(
                    config["space_weather_file"],
                    max_age_days=config.get("space_weather_max_age_days", 1),
                    expected_sha256=config.get("space_weather_sha256"),
                )
            result = NRLMSISE2Atmosphere(
                f107=config.get("f107", 150.0),
                f107a=config.get("f107a", 150.0),
                ap=config.get("ap", 4.0),
                space_weather=space_weather,
            )
        except ImportError as exc:
            raise RuntimeError(
                "NRLMSISE-2 was requested but pymsis is not installed"
            ) from exc
    if result is None:
        raise ValueError(f"unknown atmosphere model: {model}")
    if "wind_ecef_mps" in config:
        result = WindAdjustedAtmosphere(result, config["wind_ecef_mps"])
    return result
