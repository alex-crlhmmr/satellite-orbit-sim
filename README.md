# Satellite Orbit Sim

Validated low-Earth-orbit propagation with headless 3D rendering and LAN
telemetry/video streaming. The supported default is a high-fidelity Brahe
backend; a compact local propagator remains available for education and the
experimental thrust/RL work.

## Default dynamics

The `high_fidelity` backend uses:

- EGM2008 spherical harmonics, 20×20 by default
- IERS Earth orientation and full GCRF/ITRF transformations
- NRLMSISE-00 with measured space weather
- JPL DE440s Sun/Moon ephemerides
- conical eclipses and solar radiation pressure
- solid Earth and pole tides
- relativistic acceleration
- adaptive RKF78 integration and state-transition/covariance propagation

The exact supported and missing effects are tracked in
[docs/dynamics_status.md](docs/dynamics_status.md). Do not infer support from
settings belonging to the legacy backend: backend-specific configuration is
separated and unsupported high-fidelity keys fail loudly.

## Evidence

- Independent Brahe/Orekit comparisons over five orbit regimes and five
  force-model profiles: [validation/](validation/README.md)
- Frozen Sentinel-1A and Swarm-A precise-orbit benchmarks:
  [validation/real_data/](validation/real_data/README.md)
- GRACE-FO accelerometer-density benchmark with an untouched storm test:
  [validation/density/](validation/density/README.md)
- 87 tests covering physics, configuration, streaming, visualization geometry,
  uncertainty and data
  protocols

On the frozen GRACE-FO October storm interval, raw NRLMSISE-00 density MAPE is
92.56%, an April-trained static scale reaches 59.01%, and the validation-selected
one-step estimator reaches 11.98%. This is a direct density benchmark, not a
claim that ordinary satellites measure density onboard.

## Installation

Python 3.12 is the supported version. `pyproject.toml` is canonical and
`uv.lock` pins the reproducible development environment.

### macOS or ordinary Linux

```bash
# Reproducible development environment
uv sync --all-extras

# Or with pip
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all,dev]"
```

Physics-only installations can omit the optional UI and experimental code:

```bash
python -m pip install -e .
```

Available extras are `legacy`, `render`, `viewer`, `rl`, `dev`, and `all`.

### NVIDIA Jetson

Use Python 3.12 only if it is supported by the installed JetPack release.
Install NVIDIA's JetPack-compatible PyTorch wheel first; do not allow pip or
uv to replace it with a generic CUDA wheel. Then install this project:

```bash
python -m pip install -e ".[all]"
sudo apt-get install gstreamer1.0-nice gstreamer1.0-plugins-bad
```

Confirm `python -c "import torch; print(torch.__version__)"` still reports the
NVIDIA-provided build. Jetson/PyTorch compatibility is controlled by NVIDIA,
so a universal wheel is intentionally not pinned here.
Hardware-H.264 WebRTC additionally requires JetPack's `nvvidconv` and
`nvv4l2h264enc` GStreamer elements. The browser retains MJPEG as a compatibility
fallback when WebRTC is disabled or genuinely unavailable.

## Run

On the simulation host:

```bash
python main.py
# or: satellite-orbit-sim
```

The default starts the renderer and exposes:

- browser viewer: `http://<HOST_IP>:8080`
- binary video: TCP 9100
- binary telemetry: TCP 9101

Anyone on the same trusted LAN can access the browser URL because the default
bind address is `0.0.0.0`. There is no authentication or TLS. On an untrusted
network, set `stream.bind_host: 127.0.0.1` and tunnel it:

```bash
ssh -L 8080:localhost:8080 user@HOST_IP
```

Useful commands:

```bash
python main.py --steps 5000
python main.py --no-render --no-stream --steps 1440
satellite-orbit-sim --backend legacy
satellite-orbit-sim --config config/legacy.yaml
satellite-orbit-sim --config config/constellation.yaml --camera fixed
satellite-orbit-sim --config config/constellation.yaml --target deputy-1
python main.py --camera tracking
python main.py --camera fixed
python main.py --camera split
python main.py --camera ground_track
python main.py --camera nadir
python main.py --camera horizon
python main.py --camera onboard

satellite-orbit-viewer --host HOST_IP
satellite-orbit-viewer --host HOST_IP --headless --save-telemetry
```

## Configuration

The default file is [config/default.yaml](config/default.yaml). `--config`
accepts a minimal YAML override and recursively merges it onto those defaults;
[`config/legacy.yaml`](config/legacy.yaml) is the backend-switching example.
Major sections:

- `orbit`: initial osculating Keplerian elements
- `satellite`: mass, coefficients, drag area and illuminated area
- `satellites`: optional constellation list with per-object `id`, `orbit`, and
  `satellite` overrides; singular blocks supply inherited defaults
- `propagator.high_fidelity`: supported Brahe research settings
- `propagator.legacy`: local RK4/J2–J6 and optional box-wing settings
- `atmosphere`: legacy-backend MSIS-2/USSA76 settings only
- `render` and `stream`: UI and network settings
- `environment`: experimental RL episode settings

Each constellation member owns an independent state, propagator, covariance
and trail. Telemetry schema version 2 retains the camera target's top-level
fields for compatibility and adds `satellites`, `satellite_count` and
`target_id`. Set `render.camera_target` to choose the tracked member. The
high-fidelity backend currently propagates members sequentially, so benchmark
the intended constellation size and update rate on deployment hardware.

The default satellite is a generic 100 kg, 1 m² ISS-orbit scenario—not a
specific flight vehicle. Change the mass, areas and coefficients for the
spacecraft being studied.

## Quality checks

```bash
uv run pytest -q
uv run ruff check .
```

CI runs unit tests, independent cross-validation, precise-orbit benchmarks,
atmosphere comparisons and the GRACE-FO density gate. Large source datasets are
downloaded outside the repository and verified against committed SHA-256
manifests.

## Repository map

```text
core/          Dynamics, atmosphere, aerodynamics and uncertainty
render/        EGL/moderngl renderer and camera modes
stream/        Streaming protocol, server, browser UI and optional viewer CLI
env/           Experimental Gymnasium environment (optional)
validation/    Independent and real-data evidence pipelines
tests/         Unit and evidence-integrity tests
config/        Runtime configuration
main.py        Supported simulation entry point
```

Earth visualization uses a WGS-84 ellipsoid, full IERS Earth orientation and
DE440s solar geometry. Bundled NASA surface composites and their time/provenance
limits are documented in [assets/README.md](assets/README.md). No terrain,
clouds, weather or star field is fabricated.

## Scope

This is a research-grade LEO orbit simulator, not a complete spacecraft digital
twin or flight-qualified navigation system. Attitude dynamics, thermal/power
subsystems, validated neutral winds, albedo/infrared pressure, maneuver support
in the research backend and operational measurement filtering are not claimed.
The RL package is experimental and is not connected to `main.py`.
