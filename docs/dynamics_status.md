# Dynamics verification status

The simulator has two deliberately separate dynamics profiles.

## Research profile (default)

`propagator.backend: high_fidelity` delegates standardized models to Brahe
1.7+, rather than maintaining local copies of reference algorithms. It uses:

- EGM2008 spherical harmonics, configurable degree/order (20x20 by default)
- IERS Bulletin A Earth orientation and full GCRF/ITRF rotation
- WGS-84 geodetic coordinates and measured space weather with NRLMSISE-00
- JPL DE440s Sun/Moon ephemerides and a conical eclipse model
- independent drag and illuminated areas
- adaptive RKF78 integration with configured absolute/relative tolerances
- IERS solid Earth and pole tides, and relativistic acceleration
- state transition matrix and 6x6 covariance propagation
- optional continuous white-acceleration process noise in RTN; it defaults to
  zero until a representative tracking-data campaign calibrates its PSD
- a reusable scalar random-walk Kalman estimator for effective density/ballistic
  scale, including propagated variance and normalized-innovation diagnostics;
  it is not enabled by default until a measurement source and calibrated noise
  protocol are configured

Brahe caches EOP, space-weather, and ephemeris data under `~/.cache/brahe`.
The first research-profile start therefore needs internet access; later starts
use the cache.

## Legacy profile

`propagator.backend: legacy` retains the compact local RK4 implementation for
education and the experimental thrust/RL interface. J3 and J5 were corrected
against potential gradients, and all J2--J6 terms now have finite-difference
regression tests. This profile is not a precision orbit-determination engine.
It supports an optional box-wing projected-area model evaluated at each force
call under an explicit +X along-track/+Z nadir LVLH attitude law. This removes
the constant-area assumption for controlled simulations, but is not a
substitute for mission attitude telemetry. A fixed body-to-ECI quaternion is
also supported for controlled cases; time-tagged attitude telemetry is still
an explicit next integration.

The local MSIS adapter can use a checksum-locked daily CSV of F10.7, centered
F10.7A, and Ap values. It rejects checksum mismatches and stale/missing dates.
Fixed indices remain available for deterministic sensitivity scenarios. A
specified Earth-fixed neutral-wind vector can be applied for force sensitivity
testing, but it is not presented as a substitute for HWM or measured winds.

## Explicitly not yet claimed

Earth albedo/infrared radiation pressure, horizontal winds, estimated
empirical accelerations, operational integration of density-scale measurements, and a
completed independent GMAT/Orekit truth-data campaign are not implemented.
JB2008 and DTM2020 adapters are also not implemented. The real-data validation
does include a frozen-arc comparison of exponential, Harris-Priester, and
NRLMSISE-00 models, with the ballistic parameter refitted separately and that
identifiability limitation reported explicitly.
The research backend propagates covariance, but it is not yet a measurement
filter. These remain separate validation work; their absence is surfaced here
instead of being hidden behind a "high fidelity" label.
