# Satellite Orbit Sim

High-fidelity LEO satellite orbital simulation with real-time 3D rendering and TCP streaming. Built to run headless on NVIDIA Jetson Orin and stream video + telemetry to remote clients.

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.0%2B-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Features

- **High-fidelity orbital mechanics** — J2-J6 zonal harmonics, atmospheric drag (US Standard 1976), solar radiation pressure with cylindrical shadow, Sun/Moon third-body perturbations
- **RK4 propagator** — Fixed-step integration in float64 with individually togglable force models
- **3D headless rendering** — Earth with NASA Blue Marble day/night textures, orbit trail, satellite marker via moderngl + EGL (no display server needed)
- **Real-time streaming** — TCP server broadcasts JPEG video frames and JSON telemetry to connected clients
- **Remote viewer** — OpenCV-based client with telemetry overlay, frame capture, and headless recording mode
- **RL environment** — Gymnasium-compatible env with relative orbital element (ROE) observations and continuous RTN thrust actions
- **Multiple camera modes** — Tracking (follows satellite), fixed inertial, or split-screen (both)

## Architecture

```
core/           Physics engine (gravity, drag, SRP, third-body, propagator)
env/            Gymnasium RL environment and reward functions
render/         moderngl/EGL 3D renderer with GLSL shaders
stream/         TCP streaming server and protocol
config/         YAML configuration
assets/         NASA Blue Marble textures
tests/          Validation test suite (19 tests)
main.py         Simulation entry point
viewer.py       Remote viewer client
```

## Quick Start

### On the Jetson (or any Linux machine with GPU)

```bash
pip install -r requirements.txt
python main.py
```

This starts the simulation with default config (ISS-like orbit at 400 km, 51.6° inclination) and begins streaming on ports 9100 (video) and 9101 (telemetry).

### Remote Viewing

From another machine on the network:

```bash
python viewer.py --host <JETSON_IP>
```

If ports are firewalled, use SSH tunneling:

```bash
ssh -L 9100:localhost:9100 -L 9101:localhost:9101 user@<JETSON_IP>
python viewer.py --host localhost
```

### Options

```bash
# Simulation
python main.py --camera split          # Split-screen: tracking + inertial views
python main.py --camera fixed          # Fixed inertial camera
python main.py --steps 5000            # Run for 5000 steps then stop
python main.py --no-render --no-stream # Propagation only (headless, no streaming)
python main.py --config path/to/config.yaml

# Viewer
python viewer.py --host localhost --headless          # Terminal-only (no OpenCV window)
python viewer.py --host localhost --save-frames        # Save every 30th frame as PNG
python viewer.py --host localhost --save-telemetry     # Record telemetry to JSONL
```

## Physics Models

| Model | Implementation | Reference |
|-------|---------------|-----------|
| Gravity | J2-J6 zonal harmonics, closed-form Cartesian | Vallado, *Fundamentals of Astrodynamics* |
| Drag | 28-band exponential atmosphere (US Std 1976) | Vallado Table 8-4 |
| SRP | Solar radiation pressure + cylindrical shadow | Montenbruck & Gill |
| Third-body | Sun (Meeus) + Moon (Brown's theory) | Meeus, *Astronomical Algorithms* |
| Integrator | RK4 fixed-step, dt=10s, float64 | — |

## Validation

Run the test suite:

```bash
python -m pytest tests/ -v
```

Key results from 19 tests:
- Two-body energy conservation: relative error < 4e-11 over 100 orbits
- Kepler period return: < 0.05 m position error
- J2 RAAN drift: < 1% error vs analytical prediction
- Sun-synchronous orbit: correct ~0.9856°/day RAAN precession

## Configuration

All parameters are set in `config/default.yaml`:

- **Orbit** — Keplerian elements (SMA, eccentricity, inclination, RAAN, AoP, true anomaly)
- **Satellite** — Mass, drag coefficient, reflectivity, area-to-mass ratio, max thrust
- **Propagator** — Timestep, enable/disable individual perturbations, max J degree
- **Render** — Resolution, FPS, camera mode, trail length, texture paths
- **Stream** — Video/telemetry ports, JPEG quality
- **Environment** — Step duration, max steps, reward type, fuel penalty

## RL Environment

The Gymnasium environment (`env/orbital_env.py`) supports station-keeping, orbit-raising, and deorbit tasks:

```python
import gymnasium as gym
from env.orbital_env import OrbitalEnv

env = OrbitalEnv(config)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # RTN thrust [-1, 1]^3
    obs, reward, terminated, truncated, info = env.step(action)
```

**Observation space:** Relative orbital elements (6), altitude, orbital period, eclipse flag, normalized time

**Action space:** Continuous thrust in RTN frame, `Box(-1, 1, shape=(3,))`, scaled by `max_thrust_n`

## Streaming Protocol

Binary TCP protocol with 21-byte header:

```
[4B magic "ORBT"][1B channel][4B payload_len][4B seq][8B sim_time][payload]
```

- Channel 0x01: JPEG video frame
- Channel 0x02: JSON telemetry (orbital elements, altitude, speed, position/velocity)

## Hardware

Developed and tested on:
- NVIDIA Jetson Orin 64GB (JetPack 6, CUDA 12.6)
- ARM v8, 8 cores
- Headless GPU rendering via EGL

## Dependencies

- PyTorch >= 2.0 (CPU sufficient for single-satellite; CUDA for batch RL)
- moderngl >= 5.8 (with EGL backend)
- Gymnasium >= 0.29
- NumPy, Pillow, PyYAML, OpenCV
