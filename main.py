"""
Orbital Simulation — Main Entry Point

Runs the simulation loop: propagate orbit, render frames, stream video + telemetry.

The reinforcement-learning environment is experimental and is not wired into
this entry point.
"""

import argparse
import asyncio
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from core.constants import DEG2RAD, M2KM, MU_EARTH, R_EARTH
from core.elements import cartesian_to_keplerian, keplerian_to_cartesian
from core.frames import datetime_to_jd
from core.propagator import Propagator
from render.ephemeris import scene_ephemeris


def _merge_config(base: dict, override: dict) -> dict:
    """Recursively merge a configuration override without mutating either input."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_yaml_mapping(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as config_file:
            content = yaml.safe_load(config_file)
    except FileNotFoundError as exc:
        raise ValueError(f"configuration file not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML configuration in {path}: {exc}") from exc
    if content is None:
        return {}
    if not isinstance(content, dict):
        raise ValueError(f"configuration root must be a mapping: {path}")
    return content


def load_config(path: str | Path | None = None) -> dict:
    """Load defaults and apply an optional, minimal YAML override file."""
    default_path = Path(__file__).parent / "config" / "default.yaml"
    config = _read_yaml_mapping(default_path)
    if path is not None:
        config = _merge_config(config, _read_yaml_mapping(Path(path).expanduser()))
    return config


def build_initial_state(config: dict) -> torch.Tensor:
    """Build initial state vector from orbital elements in config."""
    orb = config["orbit"]
    dtype = torch.float64

    a = torch.tensor(orb["semi_major_axis_m"], dtype=dtype)
    e = torch.tensor(orb["eccentricity"], dtype=dtype)
    i = torch.tensor(orb["inclination_deg"] * DEG2RAD, dtype=dtype)
    raan = torch.tensor(orb["raan_deg"] * DEG2RAD, dtype=dtype)
    argp = torch.tensor(orb["arg_periapsis_deg"] * DEG2RAD, dtype=dtype)
    nu = torch.tensor(orb["true_anomaly_deg"] * DEG2RAD, dtype=dtype)

    r, v = keplerian_to_cartesian(a, e, i, raan, argp, nu)
    return torch.cat([r, v])


def build_propagator(config: dict, epoch_jd: float) -> Propagator:
    """Build the selected backend, rejecting keys it cannot honor."""
    sat = config["satellite"]
    prop = config["propagator"]
    root_allowed = {"backend", "dt", "high_fidelity", "legacy"}
    root_unknown = set(prop) - root_allowed
    if root_unknown:
        raise ValueError(
            "unknown propagator keys: " + ", ".join(sorted(root_unknown))
        )
    backend = str(prop.get("backend", "high_fidelity")).lower()
    if backend not in {"high_fidelity", "legacy"}:
        raise ValueError(f"unknown propagator backend: {backend}")
    common = {
        "dt": prop["dt"], "epoch_jd": epoch_jd, "mass": sat["mass_kg"],
        "cd": sat["drag_coefficient"], "cr": sat["reflectivity_coefficient"],
        "area_mass": sat["area_to_mass_ratio"], "device": "cpu",
        "dtype": torch.float64,
    }
    if backend == "high_fidelity":
        hf = prop.get("high_fidelity", {})
        hf_allowed = {
            "gravity_degree", "gravity_order", "enable_drag", "enable_srp",
            "enable_third_body", "enable_relativity", "enable_tides", "abs_tol",
            "rel_tol", "max_step", "initial_covariance",
            "process_noise_acceleration_psd_rtn",
        }
        unsupported = {"drag_geometry", "drag_attitude", "attitude_quaternion_xyzw"}
        supplied = unsupported.intersection(hf)
        if supplied:
            raise ValueError(
                "high_fidelity backend does not support: " + ", ".join(sorted(supplied))
            )
        unknown = set(hf) - hf_allowed
        if unknown:
            raise ValueError(
                "unknown high_fidelity keys: " + ", ".join(sorted(unknown))
            )
        hf_config = common | hf
        for source, target in (("drag_area_m2", "drag_area_m2"),
                               ("srp_area_m2", "srp_area_m2")):
            if sat.get(source) is not None:
                hf_config[target] = sat[source]
        from core.high_fidelity import HighFidelityPropagator
        return HighFidelityPropagator(hf_config)

    legacy = prop.get("legacy", {})
    legacy_allowed = {
        "enable_j2", "max_j_degree", "enable_drag", "enable_srp",
        "enable_third_body", "drag_geometry", "drag_attitude",
        "attitude_quaternion_xyzw", "epoch_jd",
    }
    legacy_unknown = set(legacy) - legacy_allowed
    if legacy_unknown:
        raise ValueError(
            "unknown legacy keys: " + ", ".join(sorted(legacy_unknown))
        )
    prop_config = {
        "mu": MU_EARTH,
        **common,
        "enable_j2": legacy.get("enable_j2", True),
        "max_j_degree": legacy.get("max_j_degree", 6),
        "enable_drag": legacy.get("enable_drag", True),
        "enable_srp": legacy.get("enable_srp", True),
        "enable_third_body": legacy.get("enable_third_body", True),
        "atmosphere": config.get("atmosphere", {}),
    }
    prop_config.update({key: legacy[key] for key in
                        ("drag_geometry", "drag_attitude", "attitude_quaternion_xyzw")
                        if key in legacy})
    return Propagator(prop_config)


def build_telemetry(state: torch.Tensor, sim_time: float, epoch_jd: float) -> dict:
    """Build telemetry dictionary from current state."""
    r = state[:3]
    v = state[3:6]
    r_mag = torch.norm(r).item()
    v_mag = torch.norm(v).item()
    altitude_km = (r_mag - R_EARTH) * M2KM

    oe = cartesian_to_keplerian(r, v)

    return {
        "sim_time_s": sim_time,
        "position_eci_m": r.tolist(),
        "velocity_eci_ms": v.tolist(),
        "altitude_km": altitude_km,
        "speed_ms": v_mag,
        "semi_major_axis_km": oe["a"].item() * M2KM,
        "eccentricity": oe["e"].item(),
        "inclination_deg": oe["i"].item() / DEG2RAD,
        "raan_deg": oe["raan"].item() / DEG2RAD,
        "argp_deg": oe["argp"].item() / DEG2RAD,
        "true_anomaly_deg": oe["nu"].item() / DEG2RAD,
    }


async def run_simulation(config: dict):
    """Main simulation loop."""
    # Epoch
    ep = config["epoch"]
    epoch_dt = datetime(ep["year"], ep["month"], ep["day"],
                        ep["hour"], ep["minute"], ep["second"])
    epoch_jd = datetime_to_jd(epoch_dt)

    # Initial state
    state = build_initial_state(config)
    propagator = build_propagator(config, epoch_jd)

    # Rendering setup
    renderer = None
    render_cfg = config.get("render", {})
    if render_cfg.get("enabled", False):
        try:
            from render.renderer import Renderer
            renderer = Renderer(
                width=render_cfg.get("width", 1280),
                height=render_cfg.get("height", 720),
                config=render_cfg,
            )
            camera_mode = render_cfg.get("camera_mode", "tracking")
            renderer.set_camera_mode(camera_mode)
            print(f"Renderer initialized: {render_cfg.get('width', 1280)}x{render_cfg.get('height', 720)}, camera={camera_mode}")
        except Exception as e:
            print(f"Renderer init failed (continuing without rendering): {e}")

    # Streaming setup
    server = None
    stream_cfg = config.get("stream", {})
    if stream_cfg.get("enabled", False):
        try:
            from stream.server import StreamingServer
            server = StreamingServer(
                bind_host=stream_cfg.get("bind_host", "0.0.0.0"),
                video_port=stream_cfg.get("video_port", 9100),
                telemetry_port=stream_cfg.get("telemetry_port", 9101),
                http_port=stream_cfg.get("http_port", 8080),
                jpeg_quality=stream_cfg.get("jpeg_quality", 85),
            )
            await server.start()
            vp = stream_cfg.get("video_port", 9100)
            tp = stream_cfg.get("telemetry_port", 9101)
            hp = stream_cfg.get("http_port", 8080)
            print(f"Streaming server started: binary {vp}/{tp}, http {hp}  (browser: http://localhost:{hp})")
        except Exception as e:
            server = None
            print(f"Streaming server init failed (continuing without streaming): {e}")

    # Trail buffer
    trail_length = render_cfg.get("trail_length", 500)
    trail = deque(maxlen=trail_length)

    # Simulation parameters
    env_dt = config.get("environment", {}).get("env_dt", 60.0)
    render_fps = render_cfg.get("fps", 30)
    wall_frame_interval = 1.0 / render_fps if render_fps > 0 else 1.0 / 30.0
    # How many sim steps to run between rendered frames
    # At ~58ms/step, 2 steps per frame ≈ 8-10 FPS streamed
    steps_per_frame = max(1, int(wall_frame_interval / 0.07))  # ~0.07s per step

    sim_time = 0.0
    step_count = 0
    seq = 0

    max_steps = config.get("environment", {}).get("max_steps", 0)  # 0 = unlimited

    print(f"Starting simulation: epoch={epoch_dt.isoformat()}, dt={propagator.dt}s, env_dt={env_dt}s")
    print(f"Initial altitude: {(torch.norm(state[:3]).item() - R_EARTH) * M2KM:.1f} km")
    print(f"Render: {render_fps}fps target, {steps_per_frame} sim steps per frame")
    print("Press Ctrl+C to stop")

    try:
        while True:
            wall_start = time.monotonic()

            # Run propagation steps (yield to event loop between steps
            # so asyncio can process new client connections)
            for _ in range(steps_per_frame):
                state, _ = propagator.propagate(state, env_dt, t0=sim_time)
                sim_time += env_dt
                step_count += 1
                trail.append(state[:3].numpy().copy())

                # Yield to event loop for connection handling
                await asyncio.sleep(0)

                if max_steps > 0 and step_count >= max_steps:
                    break
                altitude = torch.norm(state[:3]).item() - R_EARTH
                if altitude < 100e3:
                    break

            # Render a video frame when EGL is available.
            frame = None
            if renderer is not None:
                scene = scene_ephemeris(epoch_jd + sim_time / 86400.0)

                trail_array = np.array(list(trail)) if len(trail) > 1 else None
                frame = renderer.render_frame(
                    sat_positions=state[:3].numpy(),
                    sun_pos=scene.sun_position_gcrf_m,
                    earth_rotation=scene.itrf_to_gcrf,
                    trail_positions=trail_array,
                    sat_velocity=state[3:6].numpy(),
                )

            # Telemetry is physics-only and remains available in no-render mode
            # or when renderer initialisation fails.
            if server is not None:
                telemetry = build_telemetry(state, sim_time, epoch_jd)
                if frame is not None:
                    await server.send_video_frame(
                        frame, seq, sim_time,
                    )
                await server.send_telemetry(telemetry, seq, sim_time)
                seq += 1

            # Print status periodically
            if step_count % 100 == 0:
                alt_km = (torch.norm(state[:3]).item() - R_EARTH) * M2KM
                vc, tc = (0, 0) if server is None else server.client_count()
                wall_fps = 1.0 / max(time.monotonic() - wall_start, 1e-6)
                print(f"  Step {step_count}: t={sim_time:.0f}s, alt={alt_km:.1f}km, "
                      f"{wall_fps:.1f}fps, clients=({vc}v,{tc}t)")

            # Check termination
            if max_steps > 0 and step_count >= max_steps:
                print(f"Reached max_steps ({max_steps})")
                break

            altitude = torch.norm(state[:3]).item() - R_EARTH
            if altitude < 100e3:
                print(f"Satellite reentered at altitude {altitude * M2KM:.1f} km")
                break

            # Pace: don't go faster than target FPS wall-clock
            wall_elapsed = time.monotonic() - wall_start
            if wall_elapsed < wall_frame_interval:
                await asyncio.sleep(wall_frame_interval - wall_elapsed)

    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    finally:
        if server is not None:
            await server.stop()
        if renderer is not None:
            renderer.cleanup()

    print(f"Final state after {sim_time:.0f}s ({sim_time/3600:.1f}h):")
    telemetry = build_telemetry(state, sim_time, epoch_jd)
    for k, v in telemetry.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, list):
            print(f"  {k}: [{', '.join(f'{x:.1f}' for x in v)}]")
        else:
            print(f"  {k}: {v}")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LEO Satellite Orbital Simulation")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML override merged onto config/default.yaml",
    )
    parser.add_argument(
        "--backend", choices=["high_fidelity", "legacy"], default=None,
        help="Override the propagator backend selected by the configuration",
    )
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable streaming")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max_steps")
    parser.add_argument("--camera", type=str, default=None,
                        choices=["tracking", "fixed", "split", "ground_track",
                                 "nadir", "horizon", "onboard"],
                        help="Camera mode: 'tracking' (follows satellite), 'fixed' (inertial overview), 'split' (tracking + fixed), 'ground_track' (top-down nadir overview), 'nadir' (onboard camera pointed at Earth), 'horizon' (onboard camera pointed at horizon along velocity), 'onboard' (nadir + horizon split)")
    return parser


def main():
    args = build_argument_parser().parse_args()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if args.backend is not None:
        config.setdefault("propagator", {})["backend"] = args.backend
    if args.camera is not None:
        config.setdefault("render", {})["camera_mode"] = args.camera
    if args.no_render:
        config.setdefault("render", {})["enabled"] = False
    if args.no_stream:
        config.setdefault("stream", {})["enabled"] = False
    if args.steps is not None:
        config.setdefault("environment", {})["max_steps"] = args.steps

    try:
        asyncio.run(run_simulation(config))
    except KeyboardInterrupt:
        # asyncio.run() cancels the main task before re-raising Ctrl+C, so the
        # coroutine-local handler cannot reliably suppress this traceback.
        print("\nSimulation stopped by user")


if __name__ == "__main__":
    main()
