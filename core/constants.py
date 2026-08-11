"""
WGS-84, IAU, and derived constants for orbital mechanics.

All values in SI units (m, kg, s, rad) unless noted.
"""

import math

import torch

# --- WGS-84 Earth Constants ---
MU_EARTH = 3.986004418e14          # Gravitational parameter [m³/s²]
R_EARTH = 6.378137e6               # Equatorial radius [m]
R_EARTH_POLAR = 6.356752314245e6   # Polar radius [m]
OMEGA_EARTH = 7.2921150e-5         # Earth rotation rate [rad/s]
FLATTENING = 1.0 / 298.257223563   # WGS-84 flattening

# --- Zonal Harmonics (unnormalized) ---
J2 = 1.08263e-3
J3 = -2.53881e-6
J4 = -1.61988e-6
J5 = -2.27141e-7
J6 = 5.40788e-7

# --- Solar Constants ---
MU_SUN = 1.32712440018e20         # Sun gravitational parameter [m³/s²]
AU = 1.495978707e11               # Astronomical unit [m]
P_SUN = 4.56e-6                   # Solar radiation pressure at 1 AU [N/m²]
R_SUN = 6.957e8                   # Solar radius [m]

# --- Lunar Constants ---
MU_MOON = 4.9028e12               # Moon gravitational parameter [m³/s²]

# --- Time Constants ---
JD_J2000 = 2451545.0              # Julian date of J2000.0 epoch
MJD_J2000 = 51544.5               # Modified Julian date of J2000.0
SECONDS_PER_DAY = 86400.0
DAYS_PER_CENTURY = 36525.0

# --- Unit Conversions ---
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
KM2M = 1000.0
M2KM = 0.001

# --- Default Satellite Properties ---
DEFAULT_CD = 2.2                   # Drag coefficient
DEFAULT_CR = 1.5                   # Reflectivity coefficient
DEFAULT_AREA_MASS = 0.01           # Area-to-mass ratio [m²/kg]
DEFAULT_MASS = 100.0               # Satellite mass [kg]

# --- Atmospheric Drag ---
# US Standard Atmosphere 1976 - 28 altitude bands
# Each tuple: (base_altitude_km, base_density_kg_m3, scale_height_km)
ATMOSPHERE_BANDS = [
    (0,     1.225,        7.249),
    (25,    3.899e-2,     6.349),
    (30,    1.774e-2,     6.682),
    (40,    3.972e-3,     7.554),
    (50,    1.057e-3,     8.382),
    (60,    3.206e-4,     7.714),
    (70,    8.770e-5,     6.549),
    (80,    1.905e-5,     5.799),
    (90,    3.396e-6,     5.382),
    (100,   5.297e-7,     5.877),
    (110,   9.661e-8,     7.263),
    (120,   2.438e-8,     9.473),
    (130,   8.484e-9,     12.636),
    (140,   3.845e-9,     16.149),
    (150,   2.070e-9,     22.523),
    (180,   5.464e-10,    29.740),
    (200,   2.789e-10,    37.105),
    (250,   7.248e-11,    45.546),
    (300,   2.418e-11,    53.628),
    (350,   9.518e-12,    53.298),
    (400,   3.725e-12,    58.515),
    (450,   1.585e-12,    60.828),
    (500,   6.967e-13,    63.822),
    (600,   1.454e-13,    71.835),
    (700,   3.614e-14,    88.667),
    (800,   1.170e-14,    124.64),
    (900,   5.245e-15,    181.05),
    (1000,  3.019e-15,    268.00),
]


def get_tensors(device: str = "cpu", dtype: torch.dtype = torch.float64) -> dict:
    """Return all constants as tensors on the specified device."""
    return {
        "mu_earth": torch.tensor(MU_EARTH, device=device, dtype=dtype),
        "r_earth": torch.tensor(R_EARTH, device=device, dtype=dtype),
        "omega_earth": torch.tensor(OMEGA_EARTH, device=device, dtype=dtype),
        "j2": torch.tensor(J2, device=device, dtype=dtype),
        "j3": torch.tensor(J3, device=device, dtype=dtype),
        "j4": torch.tensor(J4, device=device, dtype=dtype),
        "j5": torch.tensor(J5, device=device, dtype=dtype),
        "j6": torch.tensor(J6, device=device, dtype=dtype),
        "mu_sun": torch.tensor(MU_SUN, device=device, dtype=dtype),
        "au": torch.tensor(AU, device=device, dtype=dtype),
        "p_sun": torch.tensor(P_SUN, device=device, dtype=dtype),
        "mu_moon": torch.tensor(MU_MOON, device=device, dtype=dtype),
    }
