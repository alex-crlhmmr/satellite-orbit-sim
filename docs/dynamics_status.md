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

Brahe caches EOP, space-weather, and ephemeris data under `~/.cache/brahe`.
The first research-profile start therefore needs internet access; later starts
use the cache.

## Legacy profile

`propagator.backend: legacy` retains the compact local RK4 implementation for
education and the experimental thrust/RL interface. J3 and J5 were corrected
against potential gradients, and all J2--J6 terms now have finite-difference
regression tests. This profile is not a precision orbit-determination engine.

## Explicitly not yet claimed

Earth albedo/infrared radiation pressure, horizontal winds, estimated
empirical accelerations, stochastic density scale-factor estimation, and a
completed independent GMAT/Orekit truth-data campaign are not implemented.
The research backend propagates covariance, but it is not yet a measurement
filter. These remain separate validation work; their absence is surfaced here
instead of being hidden behind a "high fidelity" label.
