"""
Validation test suite for the orbital simulation core physics engine.

Tests cover element conversions, anomaly solvers, propagation invariants,
perturbation effects, frame transforms, relative orbital elements,
atmospheric density, Sun ephemeris, and batched operations.
"""

import math
import numpy as np
import pytest
import torch

from core.constants import (
    MU_EARTH,
    R_EARTH,
    J2,
    AU,
    DEG2RAD,
    JD_J2000,
    SECONDS_PER_DAY,
)
from core.elements import (
    cartesian_to_keplerian,
    keplerian_to_cartesian,
    true_to_eccentric_anomaly,
    eccentric_to_mean_anomaly,
    mean_to_eccentric_anomaly,
    eccentric_to_true_anomaly,
    cartesian_to_roe,
)
from core.frames import (
    eci_to_ecef,
    ecef_to_eci,
    eci_to_rtn,
    gmst_from_jd,
)
from core.atmosphere import atmospheric_density, drag_acceleration, make_atmosphere
from core.atmosphere import _ecef_to_geodetic
from core.gravity import (
    j2_acceleration, j3_acceleration, j4_acceleration,
    j5_acceleration, j6_acceleration,
)
from core.srp import sun_position_eci
from core.propagator import Propagator
from core.aerodynamics import BoxWingGeometry, lvlh_body_to_eci


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_propagator(**overrides) -> Propagator:
    """Create a Propagator with all perturbations off by default."""
    config = {
        "mu": MU_EARTH,
        "dt": 10.0,
        "enable_j2": False,
        "max_j_degree": 2,
        "enable_drag": False,
        "cd": 2.2,
        "area_mass": 0.01,
        "enable_srp": False,
        "cr": 1.5,
        "enable_third_body": False,
        "epoch_jd": JD_J2000,
        "device": "cpu",
        "dtype": torch.float64,
    }
    config.update(overrides)
    return Propagator(config)


def _keplerian_state(a_m, e, i_deg, raan_deg, argp_deg, nu_deg):
    """Return (r, v) tensors from classical orbital elements."""
    a = torch.tensor(a_m, dtype=torch.float64)
    ecc = torch.tensor(e, dtype=torch.float64)
    inc = torch.tensor(i_deg * DEG2RAD, dtype=torch.float64)
    raan = torch.tensor(raan_deg * DEG2RAD, dtype=torch.float64)
    argp = torch.tensor(argp_deg * DEG2RAD, dtype=torch.float64)
    nu = torch.tensor(nu_deg * DEG2RAD, dtype=torch.float64)
    r, v = keplerian_to_cartesian(a, ecc, inc, raan, argp, nu)
    return r, v


def _state_vector(r, v):
    """Combine position and velocity into a (6,) state vector."""
    return torch.cat([r, v], dim=-1)


def _specific_energy(r, v, mu=MU_EARTH):
    """Compute vis-viva specific orbital energy."""
    r_mag = torch.norm(r, dim=-1)
    v_mag = torch.norm(v, dim=-1)
    return 0.5 * v_mag ** 2 - mu / r_mag


def _angular_momentum(r, v):
    """Return angular momentum vector h = r x v."""
    return torch.cross(r, v, dim=-1)


def test_lvlh_attitude_and_box_projected_area():
    r = np.array([7.0e6, 0.0, 0.0])
    v = np.array([0.0, 7500.0, 0.0])
    dcm = lvlh_body_to_eci(r, v)
    np.testing.assert_allclose(dcm.T @ dcm, np.eye(3), atol=1e-15)
    np.testing.assert_allclose(dcm[:, 0], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(dcm[:, 2], [-1.0, 0.0, 0.0])
    geometry = BoxWingGeometry.from_config({
        "box_dimensions_m": [2.0, 3.0, 4.0],
        "panels": [{"normal_body": [1.0, 0.0, 0.0], "area_m2": 5.0}],
    })
    assert geometry.projected_area(np.array([1.0, 0.0, 0.0])) == 17.0
    assert geometry.projected_area(np.array([0.0, 1.0, 0.0])) == 8.0


def test_geometry_changes_drag_with_attitude_relative_flow():
    geometry = BoxWingGeometry.from_config(
        {"box_dimensions_m": [2.0, 1.0, 1.0], "panels": []}
    )
    r = np.array([R_EARTH + 200e3, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    along = geometry.area_mass_ratio_lvlh(r, v, v, 100.0)
    nadir = geometry.area_mass_ratio_lvlh(r, v, -r, 100.0)
    assert nadir == 2.0 * along


def test_specified_neutral_wind_changes_relative_drag_velocity():
    r = np.array([R_EARTH + 200e3, 0.0, 0.0])
    v = np.array([0.0, 7800.0, 0.0])
    calm = make_atmosphere({"model": "ussa76"})
    # At J2000 GMST, construct ECEF wind which rotates to +ECI Y.
    angle = gmst_from_jd(JD_J2000)
    wind_ecef = [100.0 * np.sin(angle), 100.0 * np.cos(angle), 0.0]
    windy = make_atmosphere({"model": "ussa76", "wind_ecef_mps": wind_ecef})
    calm_drag = np.linalg.norm(drag_acceleration(r, v, 2.2, 0.01,
                                                atmosphere=calm, jd=JD_J2000))
    windy_drag = np.linalg.norm(drag_acceleration(r, v, 2.2, 0.01,
                                                 atmosphere=windy, jd=JD_J2000))
    assert windy_drag < calm_drag


def test_unknown_atmosphere_never_silently_falls_back():
    with pytest.raises(ValueError, match="unknown atmosphere model"):
        make_atmosphere({"model": "dtm2020_typo"})


# ===========================================================================
# 1. Keplerian element round-trip
# ===========================================================================

class TestElementRoundtrip:
    """Convert Keplerian -> Cartesian -> Keplerian and verify closure."""

    def test_element_roundtrip(self):
        a_m = 7000e3  # 7000 km
        e_val = 0.01
        i_deg = 45.0
        raan_deg = 60.0
        argp_deg = 30.0
        nu_deg = 120.0

        a = torch.tensor(a_m, dtype=torch.float64)
        ecc = torch.tensor(e_val, dtype=torch.float64)
        inc = torch.tensor(i_deg * DEG2RAD, dtype=torch.float64)
        raan = torch.tensor(raan_deg * DEG2RAD, dtype=torch.float64)
        argp = torch.tensor(argp_deg * DEG2RAD, dtype=torch.float64)
        nu = torch.tensor(nu_deg * DEG2RAD, dtype=torch.float64)

        r, v = keplerian_to_cartesian(a, ecc, inc, raan, argp, nu)
        recovered = cartesian_to_keplerian(r, v)

        pairs = [
            ("a", a),
            ("e", ecc),
            ("i", inc),
            ("raan", raan),
            ("argp", argp),
            ("nu", nu),
        ]

        for name, expected in pairs:
            got = recovered[name]
            rel_err = torch.abs(got - expected) / torch.clamp(torch.abs(expected), min=1e-30)
            assert rel_err.item() < 1e-6, (
                f"Element '{name}' relative error {rel_err.item():.2e} exceeds 1e-6 "
                f"(expected {expected.item():.8e}, got {got.item():.8e})"
            )


# ===========================================================================
# 2. Anomaly round-trip
# ===========================================================================

class TestAnomalyRoundtrip:
    """True -> Eccentric -> Mean -> Eccentric -> True must close."""

    @pytest.mark.parametrize("ecc", [0.0, 0.1, 0.5, 0.9])
    def test_anomaly_roundtrip(self, ecc):
        nu_values = torch.linspace(0.01, 2.0 * math.pi - 0.01, 20, dtype=torch.float64)
        e_tensor = torch.tensor(ecc, dtype=torch.float64)

        for nu_orig in nu_values:
            E = true_to_eccentric_anomaly(nu_orig, e_tensor)
            M = eccentric_to_mean_anomaly(E, e_tensor)
            E2 = mean_to_eccentric_anomaly(M, e_tensor)
            nu_recovered = eccentric_to_true_anomaly(E2, e_tensor)

            # Compare modulo 2*pi
            diff = (nu_recovered - nu_orig) % (2.0 * math.pi)
            diff = torch.min(diff, 2.0 * math.pi - diff)
            assert diff.item() < 1e-10, (
                f"Anomaly round-trip failed for e={ecc}, nu={nu_orig.item():.4f}: "
                f"error = {diff.item():.2e}"
            )


# ===========================================================================
# 3. Two-body energy and angular momentum conservation
# ===========================================================================

class TestTwoBodyConservation:
    """Propagate circular LEO for 100 orbits — energy and h must be conserved."""

    def test_two_body_energy_conservation(self):
        alt_m = 400e3
        a_m = R_EARTH + alt_m
        r, v = _keplerian_state(a_m, 0.001, 28.5, 0.0, 0.0, 0.0)
        state0 = _state_vector(r, v)

        # Orbital period and total duration
        T_period = 2.0 * math.pi * math.sqrt(a_m ** 3 / MU_EARTH)
        duration = 100.0 * T_period

        prop = _make_propagator(dt=10.0)
        final_state, _ = prop.propagate(state0, duration)

        r0, v0 = state0[:3], state0[3:]
        rf, vf = final_state[:3], final_state[3:]

        # Specific energy conservation
        E0 = _specific_energy(r0, v0).item()
        Ef = _specific_energy(rf, vf).item()
        rel_energy = abs(Ef - E0) / abs(E0)
        assert rel_energy < 1e-6, (
            f"Energy relative error {rel_energy:.2e} exceeds 1e-6"
        )

        # Angular momentum conservation
        h0_mag = torch.norm(_angular_momentum(r0, v0)).item()
        hf_mag = torch.norm(_angular_momentum(rf, vf)).item()
        rel_h = abs(hf_mag - h0_mag) / abs(h0_mag)
        assert rel_h < 1e-6, (
            f"Angular momentum relative error {rel_h:.2e} exceeds 1e-6"
        )


# ===========================================================================
# 4. Kepler period — final position matches initial
# ===========================================================================

class TestKeplerPeriod:
    """After one period T, position should return to the start."""

    def test_kepler_period(self):
        a_m = R_EARTH + 500e3
        r, v = _keplerian_state(a_m, 0.001, 51.6, 0.0, 0.0, 0.0)
        state0 = _state_vector(r, v)

        T_period = 2.0 * math.pi * math.sqrt(a_m ** 3 / MU_EARTH)
        prop = _make_propagator(dt=5.0)
        final_state, _ = prop.propagate(state0, T_period)

        pos_err = torch.norm(final_state[:3] - state0[:3]).item()
        assert pos_err < 1.0, (
            f"Position error after one period: {pos_err:.2f} m (limit 1 m)"
        )


# ===========================================================================
# 5. J2 RAAN drift
# ===========================================================================

class TestJ2RAANDrift:
    """Verify analytical J2 RAAN precession rate against numerical propagation."""

    def test_j2_raan_drift(self):
        alt_m = 600e3
        a_m = R_EARTH + alt_m
        i_deg = 51.6
        i_rad = i_deg * DEG2RAD

        r, v = _keplerian_state(a_m, 0.001, i_deg, 45.0, 0.0, 0.0)
        state0 = _state_vector(r, v)

        # Analytical RAAN drift rate
        n = math.sqrt(MU_EARTH / a_m ** 3)
        p = a_m * (1.0 - 0.001 ** 2)  # semi-latus rectum
        raan_dot_analytical = -1.5 * n * J2 * (R_EARTH / p) ** 2 * math.cos(i_rad)

        duration = SECONDS_PER_DAY  # 1 day
        prop = _make_propagator(dt=10.0, enable_j2=True, max_j_degree=2)
        final_state, _ = prop.propagate(state0, duration)

        oe0 = cartesian_to_keplerian(state0[:3], state0[3:])
        oef = cartesian_to_keplerian(final_state[:3], final_state[3:])

        raan0 = oe0["raan"].item()
        raanf = oef["raan"].item()

        # Handle wrapping
        d_raan = raanf - raan0
        if d_raan > math.pi:
            d_raan -= 2.0 * math.pi
        elif d_raan < -math.pi:
            d_raan += 2.0 * math.pi

        raan_dot_numerical = d_raan / duration
        expected_drift = raan_dot_analytical

        rel_err = abs(raan_dot_numerical - expected_drift) / abs(expected_drift)
        assert rel_err < 0.01, (
            f"J2 RAAN drift relative error {rel_err:.4f} exceeds 1%. "
            f"Numerical: {math.degrees(raan_dot_numerical * SECONDS_PER_DAY):.4f} deg/day, "
            f"Analytical: {math.degrees(expected_drift * SECONDS_PER_DAY):.4f} deg/day"
        )


# ===========================================================================
# 6. Sun-synchronous orbit
# ===========================================================================

class TestSunSynchronous:
    """A sun-sync orbit at 600 km should precess ~0.9856 deg/day with J2."""

    def test_sun_synchronous(self):
        alt_m = 600e3
        a_m = R_EARTH + alt_m
        # Sun-sync inclination for 600 km (approximately 97.8 deg)
        i_ss_deg = 97.8
        i_ss_rad = i_ss_deg * DEG2RAD

        r, v = _keplerian_state(a_m, 0.001, i_ss_deg, 0.0, 0.0, 0.0)
        state0 = _state_vector(r, v)

        duration = SECONDS_PER_DAY
        prop = _make_propagator(dt=10.0, enable_j2=True, max_j_degree=2)
        final_state, _ = prop.propagate(state0, duration)

        oe0 = cartesian_to_keplerian(state0[:3], state0[3:])
        oef = cartesian_to_keplerian(final_state[:3], final_state[3:])

        raan0 = oe0["raan"].item()
        raanf = oef["raan"].item()

        d_raan = raanf - raan0
        if d_raan > math.pi:
            d_raan -= 2.0 * math.pi
        elif d_raan < -math.pi:
            d_raan += 2.0 * math.pi

        d_raan_deg = math.degrees(d_raan)
        expected_deg = 0.9856  # deg/day eastward

        rel_err = abs(d_raan_deg - expected_deg) / expected_deg
        assert rel_err < 0.05, (
            f"Sun-sync RAAN drift {d_raan_deg:.4f} deg/day differs from "
            f"expected {expected_deg} deg/day by {rel_err * 100:.1f}%"
        )


# ===========================================================================
# 7. Frame transforms
# ===========================================================================

class TestFrameTransforms:
    """ECI<->ECEF round-trip and RTN orthogonality."""

    def test_eci_ecef_roundtrip(self):
        r_eci = torch.tensor([7000e3, 1000e3, 500e3], dtype=torch.float64)
        gmst = gmst_from_jd(JD_J2000 + 0.25)  # arbitrary epoch

        r_ecef = eci_to_ecef(r_eci, gmst)
        r_back = ecef_to_eci(r_ecef, gmst)

        err = torch.norm(r_back - r_eci).item()
        assert err < 1e-9, (
            f"ECI->ECEF->ECI round-trip error {err:.2e} m exceeds 1e-9 m"
        )

    def test_rtn_orthogonality(self):
        r, v = _keplerian_state(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0)
        dcm = eci_to_rtn(r, v)

        # R @ R^T should be identity
        product = dcm @ dcm.T
        identity = torch.eye(3, dtype=torch.float64)
        err = torch.norm(product - identity).item()
        assert err < 1e-12, (
            f"RTN DCM is not orthogonal: ||R R^T - I|| = {err:.2e}"
        )


# ===========================================================================
# 8. Relative Orbital Elements
# ===========================================================================

class TestROEComputation:
    """ROE for identical orbits should be zero; known offsets should appear."""

    def test_roe_identical_orbits(self):
        r, v = _keplerian_state(R_EARTH + 500e3, 0.01, 45.0, 30.0, 60.0, 90.0)
        roe = cartesian_to_roe(r, v, r, v)
        assert torch.all(torch.abs(roe) < 1e-12), (
            f"ROE for identical orbits should be zero, got {roe}"
        )

    def test_roe_da_offset(self):
        a_chief = R_EARTH + 500e3
        da = 100.0  # 100 m offset in semi-major axis
        a_deputy = a_chief + da

        r_c, v_c = _keplerian_state(a_chief, 0.01, 45.0, 30.0, 60.0, 90.0)
        r_d, v_d = _keplerian_state(a_deputy, 0.01, 45.0, 30.0, 60.0, 90.0)

        roe = cartesian_to_roe(r_c, v_c, r_d, v_d)
        expected_da = da / a_chief

        rel_err = abs(roe[0].item() - expected_da) / expected_da
        assert rel_err < 1e-4, (
            f"ROE delta-a relative error {rel_err:.2e}: "
            f"expected {expected_da:.6e}, got {roe[0].item():.6e}"
        )


# ===========================================================================
# 9. Atmospheric density
# ===========================================================================

class TestAtmosphericDensity:
    """Verify density magnitude at sea level and at 400 km."""

    def test_density_sea_level(self):
        alt = torch.tensor(0.0, dtype=torch.float64)
        rho = atmospheric_density(alt).item()
        assert abs(rho - 1.225) / 1.225 < 0.01, (
            f"Sea-level density {rho:.4f} kg/m^3 deviates from 1.225 by > 1%"
        )

    def test_density_400km(self):
        alt = torch.tensor(400e3, dtype=torch.float64)
        rho = atmospheric_density(alt).item()
        assert 1e-12 < rho < 1e-11, (
            f"Density at 400 km = {rho:.2e} kg/m^3 outside [1e-12, 1e-11] range"
        )

    def test_density_decreases_with_altitude(self):
        altitudes = torch.tensor([0.0, 100e3, 200e3, 400e3, 800e3], dtype=torch.float64)
        densities = atmospheric_density(altitudes)
        for k in range(len(altitudes) - 1):
            assert densities[k].item() > densities[k + 1].item(), (
                f"Density at {altitudes[k].item() / 1e3:.0f} km "
                f"({densities[k].item():.2e}) is not greater than at "
                f"{altitudes[k + 1].item() / 1e3:.0f} km ({densities[k + 1].item():.2e})"
            )


# ===========================================================================
# 10. Sun position
# ===========================================================================

class TestSunPosition:
    """Sun position at J2000 epoch: magnitude ~ 1 AU, direction ~ vernal equinox."""

    def test_sun_position(self):
        # J2000.0 is January 1, 2000 12:00 TT.
        # The Sun is near RA ~ 281 deg (Capricorn) at that date, not at
        # the vernal equinox (which would be near March 20).
        # We test at the vernal equinox of 2000 (~March 20.44, JD 2451624.94).
        jd_vernal = 2451624.94  # approx March 20, 2000

        r_sun = sun_position_eci(jd_vernal)
        r_mag = torch.norm(r_sun).item()

        # Magnitude should be ~1 AU (within 2%)
        rel_err = abs(r_mag - AU) / AU
        assert rel_err < 0.02, (
            f"Sun distance {r_mag:.4e} m differs from 1 AU by {rel_err * 100:.1f}%"
        )

        # At vernal equinox the Sun should be roughly along +X (ECI),
        # meaning the x-component dominates and is positive.
        r_hat = r_sun / r_mag
        assert r_hat[0].item() > 0.9, (
            f"Sun direction x-component {r_hat[0].item():.3f} is not dominant "
            f"at vernal equinox (expected > 0.9)"
        )

    def test_sun_magnitude_at_j2000(self):
        r_sun = sun_position_eci(JD_J2000)
        r_mag = torch.norm(r_sun).item()
        rel_err = abs(r_mag - AU) / AU
        assert rel_err < 0.02, (
            f"Sun distance at J2000 {r_mag:.4e} m differs from 1 AU by {rel_err * 100:.1f}%"
        )


# ===========================================================================
# 11. Batched operations
# ===========================================================================

class TestBatchedOperations:
    """Batched element conversions must match individual conversions."""

    def test_batched_element_conversions(self):
        B = 4
        a_vals = [R_EARTH + h for h in [400e3, 500e3, 600e3, 700e3]]
        e_vals = [0.001, 0.01, 0.05, 0.1]
        i_vals = [28.5, 45.0, 63.4, 90.0]
        raan_vals = [0.0, 45.0, 90.0, 180.0]
        argp_vals = [0.0, 30.0, 60.0, 270.0]
        nu_vals = [0.0, 90.0, 180.0, 270.0]

        # --- Individual conversions ---
        r_list, v_list = [], []
        for k in range(B):
            r_k, v_k = _keplerian_state(
                a_vals[k], e_vals[k], i_vals[k],
                raan_vals[k], argp_vals[k], nu_vals[k],
            )
            r_list.append(r_k)
            v_list.append(v_k)

        # --- Batched conversion (Keplerian -> Cartesian) ---
        a_batch = torch.tensor(a_vals, dtype=torch.float64)
        e_batch = torch.tensor(e_vals, dtype=torch.float64)
        i_batch = torch.tensor([x * DEG2RAD for x in i_vals], dtype=torch.float64)
        raan_batch = torch.tensor([x * DEG2RAD for x in raan_vals], dtype=torch.float64)
        argp_batch = torch.tensor([x * DEG2RAD for x in argp_vals], dtype=torch.float64)
        nu_batch = torch.tensor([x * DEG2RAD for x in nu_vals], dtype=torch.float64)

        r_batched, v_batched = keplerian_to_cartesian(
            a_batch, e_batch, i_batch, raan_batch, argp_batch, nu_batch,
        )

        for k in range(B):
            r_err = torch.norm(r_batched[k] - r_list[k]).item()
            v_err = torch.norm(v_batched[k] - v_list[k]).item()
            assert r_err < 1e-6, (
                f"Batch element {k}: position error {r_err:.2e} m"
            )
            assert v_err < 1e-6, (
                f"Batch element {k}: velocity error {v_err:.2e} m/s"
            )

        # --- Batched inverse (Cartesian -> Keplerian) ---
        oe_batched = cartesian_to_keplerian(r_batched, v_batched)

        for k in range(B):
            oe_single = cartesian_to_keplerian(r_list[k], v_list[k])
            for key in ["a", "e", "i", "raan", "argp", "nu"]:
                diff = abs(oe_batched[key][k].item() - oe_single[key].item())
                scale = max(abs(oe_single[key].item()), 1e-30)
                rel = diff / scale
                assert rel < 1e-10, (
                    f"Batch element {k}, '{key}': batched vs single relative diff = {rel:.2e}"
                )

    def test_single_item_batch(self):
        values = [torch.tensor([x], dtype=torch.float64) for x in (
            R_EARTH + 500e3, 0.01, 0.5, 0.2, 0.3, 0.4,
        )]
        r, v = keplerian_to_cartesian(*values)
        assert r.shape == (1, 3)
        assert v.shape == (1, 3)


class TestForceModelDefinitions:
    @pytest.mark.parametrize("degree,j_value,fn", [
        (2, 1.08263e-3, j2_acceleration),
        (3, -2.53881e-6, j3_acceleration),
        (4, -1.61988e-6, j4_acceleration),
        (5, -2.27141e-7, j5_acceleration),
        (6, 5.40788e-7, j6_acceleration),
    ])
    def test_zonal_acceleration_is_potential_gradient(self, degree, j_value, fn):
        from numpy.polynomial.legendre import legval
        import numpy as np

        r = np.array([6.8e6, -1.1e6, 2.2e6], dtype=np.float64)

        def potential(pos):
            radius = np.linalg.norm(pos)
            p_n = legval(pos[2] / radius, [0.0] * degree + [1.0])
            return (-MU_EARTH / radius * j_value
                    * (R_EARTH / radius) ** degree * p_n)

        h = 0.1
        reference = np.empty(3)
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = h
            reference[axis] = (potential(r + delta) - potential(r - delta)) / (2 * h)

        np.testing.assert_allclose(fn(r), reference, rtol=2e-7, atol=1e-12)

    @pytest.mark.parametrize("latitude_deg", [0.0, 30.0, 51.6, 75.0, 90.0])
    def test_wgs84_geodetic_roundtrip(self, latitude_deg):
        import numpy as np
        from core.constants import FLATTENING

        lat = math.radians(latitude_deg)
        e2 = FLATTENING * (2.0 - FLATTENING)
        n = R_EARTH / math.sqrt(1.0 - e2 * math.sin(lat) ** 2)
        altitude = 400e3
        x = (n + altitude) * math.cos(lat)
        z = (n * (1.0 - e2) + altitude) * math.sin(lat)
        got_lat, got_alt = _ecef_to_geodetic(x, 0.0, z)
        assert abs(got_lat - lat) < 1e-12
        assert abs(got_alt - altitude) < 1e-5
