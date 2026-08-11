# Technical reference

This document describes the supported implementation. Scientific evidence and
remaining limitations are maintained separately in
[dynamics_status.md](dynamics_status.md) and [`validation/`](../validation/README.md).

## State and frames

The public propagation state is a six-element Cartesian vector

```text
[x, y, z, vx, vy, vz]
```

in GCRF/ECI metres and metres per second. Epochs are UTC Julian dates at the
Python boundary. The research backend uses IERS Earth-orientation data for
GCRF/ITRF conversion. Telemetry positions and velocities remain ECI.

Initial conditions are osculating Keplerian elements converted by
`core.elements.keplerian_to_cartesian`. The simulator uses float64 internally.
PyTorch tensors form the application-facing state API, while the numerical
propagation engines operate on NumPy/Brahe data.

## Propagation backends

### High fidelity (default)

`core.high_fidelity.HighFidelityPropagator` wraps Brahe's numerical orbit
propagator. Its configured force model contains:

- EGM2008 spherical harmonics at configurable degree/order (20×20 default)
- NRLMSISE-00 drag with measured CSSI space weather
- independent constant drag and illuminated SRP areas
- DE440s Sun and Moon third-body accelerations
- conical eclipse geometry
- IERS solid-Earth and pole tides
- relativistic acceleration
- full Earth rotation in force/frame evaluation

Integration uses adaptive RKF78 with `abs_tol`, `rel_tol`, and `max_step`.
`propagator.dt` is the application step/output interval; it is not the internal
RKF78 step. The backend propagates a state-transition matrix and covariance.
Optional continuous white-acceleration process noise is specified in RTN and
defaults to zero until calibrated.

The high-fidelity backend currently accepts no commanded nonzero thrust and no
time-varying projected-area model. Supplying legacy geometry keys inside its
configuration raises an error.

### Legacy

`core.propagator.Propagator` is the transparent local backend. It uses
fixed-step RK4 and selectable two-body, J2–J6 zonal, drag, SRP and analytical
Sun/Moon perturbations. It is intended for education and experimental thrust/RL
work, not precision orbit determination.

Legacy drag can use:

- MSIS 2.x via `pymsis`
- the 28-band US Standard Atmosphere 1976 control
- an explicit Earth-fixed neutral-wind sensitivity vector
- a box-wing projected area under LVLH or fixed-quaternion attitude

Unknown atmosphere models, missing requested MSIS dependencies, stale
space-weather records and unsupported attitude laws fail explicitly.

## Aerodynamic convention

For density `rho`, drag coefficient `Cd`, projected area `A`, mass `m`, and
velocity relative to the rotating/windy atmosphere `v_rel`, the local engine
uses

```text
a_drag = -0.5 rho Cd (A/m) |v_rel| v_rel
```

The box-wing projected area is the orthographic sum of the convex bus and
two-sided panel contributions. It does not model self-shadowing, detailed
gas-surface accommodation or rarefied-flow coefficients. Therefore effective
`CdA/m` estimates must not be interpreted as independently recovered physical
area or density.

## Density-scale estimation

`core.density_estimation.DensityScaleFilter` is a scalar random-walk Kalman
filter for an effective density/ballistic scale. It carries variance and
normalized-innovation-squared diagnostics. It is validated directly against
GRACE-FO accelerometer-derived density but is not automatically enabled in
normal propagation because `main.py` has no density measurement source.

## Configuration contract

The canonical schema is [config/default.yaml](../config/default.yaml):

```yaml
propagator:
  backend: high_fidelity
  dt: 10.0
  high_fidelity:
    gravity_degree: 20
    gravity_order: 20
    enable_drag: true
    enable_srp: true
    enable_third_body: true
    enable_tides: true
    enable_relativity: true
    abs_tol: 1.0e-6
    rel_tol: 1.0e-11
    max_step: 60.0
  legacy:
    enable_j2: true
    max_j_degree: 6
```

Backend-specific keys are not interchangeable. The top-level `atmosphere`
section belongs to the legacy backend; Brahe owns the default backend's
NRLMSISE-00 and space-weather implementation.

Select the common backend override directly, or merge a minimal YAML override
onto the canonical defaults:

```bash
satellite-orbit-sim --backend legacy
satellite-orbit-sim --config config/legacy.yaml
```

## Constellations

An optional `satellites` list describes independently propagated spacecraft.
Each entry inherits the top-level `orbit` and `satellite` values and may
override either mapping. Every member owns an independent Cartesian state,
propagator, covariance and render trail. Mutual gravity, inter-satellite links,
collision response and formation-control logic are not implicitly modeled.

The renderer draws every member while `render.camera_target` selects the one
used by tracking and onboard cameras. Telemetry schema version 2 includes each
member and its `active` status under `satellites`; the target member remains at
the top level for compatibility. High-fidelity members currently advance
sequentially, so propagation cost is approximately linear in constellation
size.

## Rendering and cameras

The renderer uses moderngl with an EGL standalone context and an off-screen RGB
plus depth framebuffer. Earth geometry is the WGS-84 reference ellipsoid. The
model transform uses Brahe's full IERS ITRF-to-GCRF rotation, and direct solar
illumination uses the same DE440s ephemeris family as the research dynamics.
The day and night rasters are documented NASA composites, not epoch-specific
weather or radiance. They provide colour only: no terrain, clouds, weather or
stars are procedurally invented. The geometry-grounded shader has no artificial
night-side daylight floor or uncalibrated atmospheric rim. It is not a
radiometrically calibrated camera model: the two composite rasters do not share
physical radiance units, and atmospheric scattering/refraction is not yet
rendered.

Earth, trail and satellite GPU objects are persistent across frames; only
their dynamic contents and uniforms are updated before framebuffer readback.
Asset provenance, acquisition periods and checksums are recorded in
[`assets/README.md`](../assets/README.md).

Supported modes:

- `tracking`: external satellite tracking
- `fixed`: inertial overview
- `split`: tracking and fixed side by side
- `ground_track`: nadir overview following the ground track
- `nadir`: onboard Earth-pointing view
- `horizon`: onboard velocity/horizon view
- `onboard`: nadir and horizon side by side

Each frame binds and clears the off-screen framebuffer before configuring the
single or split viewport, preventing stale depth data across camera changes.

## Streaming

The server exposes three listeners by default:

- TCP 9100: binary JPEG video
- TCP 9101: binary JSON telemetry
- HTTP 8080: browser viewer

The binary frame header is 21 bytes:

```text
4B magic "ORBT" | 1B channel | 4B payload length |
4B sequence | 8B simulation time | payload
```

Per-client latest-value channels prevent slow consumers from blocking
propagation. Telemetry is produced even when rendering is disabled. The default
bind address exposes the unauthenticated service to the LAN; bind to loopback
and use SSH forwarding on untrusted networks.

## Validation hierarchy

Evidence is intentionally layered:

1. analytic invariants and finite-difference force checks;
2. independent Brahe/Orekit propagation comparisons;
3. Sentinel-1A and Swarm-A precise-orbit residuals;
4. frozen atmosphere-model ablations;
5. direct GRACE-FO accelerometer-density comparison;
6. uncertainty consistency through normalized innovations.

Dataset files live outside Git. Manifests pin filenames, provenance and SHA-256
hashes. Training, validation and untouched test intervals are committed before
evaluation. Committed reports are integrity-tested and regenerated in CI.

## Installation and dependency groups

`pyproject.toml` is authoritative and supports Python 3.12. The base install is
physics-only. Optional groups are:

- `legacy`: pymsis
- `render`: moderngl and Pillow
- `viewer`: OpenCV
- `rl`: Gymnasium
- `dev`: pytest and Ruff
- `all`: all runtime features

`uv.lock` pins the ordinary macOS/Linux development graph. Jetson users must
install the JetPack-compatible NVIDIA PyTorch wheel before installing project
extras; the generic lock is not a replacement for NVIDIA's platform matrix.

## Experimental RL code

`env.OrbitalEnv` remains Gymnasium-compatible and uses the local legacy engine
because commanded thrust is not supported by the research backend. It is not
connected to `main.py`, not part of the supported simulator entry point, and
must not be used as evidence for high-fidelity propagation.
