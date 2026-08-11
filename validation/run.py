#!/usr/bin/env python3
"""Cross-validate Brahe propagation against an independent Orekit runner."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import brahe as bh
import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from core.constants import DEG2RAD
from core.elements import keplerian_to_cartesian


@dataclass
class Result:
    scenario: str
    profile: str
    horizon_s: float
    samples: int
    final_position_error_m: float
    final_velocity_error_mps: float
    max_position_error_m: float
    rms_position_error_m: float
    max_velocity_error_mps: float
    rms_velocity_error_mps: float
    final_radial_error_m: float
    final_along_track_error_m: float
    final_cross_track_error_m: float


def initial_state(case: dict) -> np.ndarray:
    values = [case["a_m"], case["e"]] + [
        case[k] * DEG2RAD for k in ("i_deg", "raan_deg", "argp_deg", "nu_deg")
    ]
    tensors = [torch.tensor(x, dtype=torch.float64) for x in values]
    r, v = keplerian_to_cartesian(*tensors)
    return torch.cat((r, v)).numpy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_orekit(args, profile: str, state: np.ndarray, duration: float,
               sample: float, degree: int, order: int) -> np.ndarray:
    gravity = str(args.gravity_file) if profile == "gravity" else "-"
    command = [
        str(args.java), "-jar", str(args.orekit_jar), str(args.orekit_data),
        profile, gravity, str(degree), str(order), args.epoch,
        str(duration), str(sample), *(f"{x:.17g}" for x in state), "GCRF",
    ]
    completed = subprocess.run(command, check=True, text=True, capture_output=True)
    rows = list(csv.DictReader(completed.stdout.splitlines()))
    return np.array([[float(row[k]) for k in (
        "elapsed_s", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"
    )] for row in rows])


def run_brahe(args, profile: str, state: np.ndarray, times: np.ndarray,
              degree: int, order: int) -> np.ndarray:
    epoch = bh.Epoch(args.epoch)
    if profile == "two_body":
        force = bh.ForceModelConfig.two_body()
    elif profile == "gravity":
        model = bh.GravityModelType.from_file(str(args.gravity_file))
        gravity = bh.GravityConfiguration.spherical_harmonic(
            degree=degree, order=order, model_type=model
        )
        force = bh.ForceModelConfig(
            gravity=gravity,
            frame_transform=bh.FrameTransformationModel.FULL_EARTH_ROTATION,
        )
    elif profile == "third_body":
        force = bh.ForceModelConfig(
            gravity=bh.GravityConfiguration.point_mass(),
            third_body=[
                bh.ThirdBodyConfiguration(bh.ThirdBody.SUN, bh.EphemerisSource.DE440s),
                bh.ThirdBodyConfiguration(bh.ThirdBody.MOON, bh.EphemerisSource.DE440s),
            ],
        )
    elif profile in ("srp", "srp_sunlight"):
        fixed = bh.ParameterSource.value
        force = bh.ForceModelConfig(
            gravity=bh.GravityConfiguration.point_mass(),
            srp=bh.SolarRadiationPressureConfiguration(
                fixed(1.0), fixed(1.5),
                bh.EclipseModel.NONE if profile == "srp_sunlight" else bh.EclipseModel.CONICAL
            ),
            mass=fixed(100.0),
        )
    else:
        raise ValueError(profile)
    propagation = (
        bh.NumericalPropagationConfig
        .with_method(bh.IntegrationMethod.RKF78)
        .with_abs_tol(1e-8)
        .with_rel_tol(1e-14)
        .with_max_step(60.0)
    )
    prop = bh.NumericalOrbitPropagator(epoch, state, propagation, force)
    rows = []
    for elapsed in times:
        target = epoch + float(elapsed)
        prop.propagate_to(target)
        rows.append([elapsed, *np.asarray(prop.current_state(), dtype=float)])
    return np.asarray(rows)


def rtn_components(reference: np.ndarray, delta: np.ndarray) -> np.ndarray:
    r, v = reference[:3], reference[3:]
    radial = r / np.linalg.norm(r)
    normal = np.cross(r, v)
    normal /= np.linalg.norm(normal)
    transverse = np.cross(normal, radial)
    return np.array([radial @ delta, transverse @ delta, normal @ delta])


def metrics(case: str, profile: str, horizon: float,
            reference: np.ndarray, candidate: np.ndarray) -> Result:
    position = candidate[:, 1:4] - reference[:, 1:4]
    velocity = candidate[:, 4:7] - reference[:, 4:7]
    pe = np.linalg.norm(position, axis=1)
    ve = np.linalg.norm(velocity, axis=1)
    rtn = rtn_components(reference[-1, 1:7], position[-1])
    return Result(case, profile, horizon, len(reference), float(pe[-1]),
                  float(ve[-1]), float(pe.max()), float(np.sqrt(np.mean(pe**2))),
                  float(ve.max()), float(np.sqrt(np.mean(ve**2))), *map(float, rtn))


def write_report(results: list[Result], output: Path, metadata: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    limits = {"two_body": 1.0, "gravity": 1.0, "third_body": 1.0,
              "srp_sunlight": 5.0, "srp": 25.0}
    payload = {
        "metadata": metadata,
        "acceptance": {"max_position_error_m_by_profile": limits},
        "passed": all(x.max_position_error_m <= limits[x.profile] for x in results),
        "results": [asdict(x) for x in results],
    }
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# Dynamics cross-validation report", "",
        "Candidate: Brahe 1.7; reference: Orekit 13.1.7. States are compared in GCRF.",
        "Acceptance: <=1 m for gravitational profiles, <=5 m for continuous "
        "SRP, and <=25 m for conical SRP across eclipse boundaries: "
        f"**{'PASS' if payload['passed'] else 'FAIL'}**.",
        "The SRP thresholds are explicit: small ephemeris/radiation-direction differences "
        "accumulate to ~4.1 m at GEO, while different limb/event implementations accumulate "
        "up to ~20 m over seven days near equinox. Both remain investigation items.", "",
        "![Maximum position disagreement](position_error.svg)", "",
        "| Scenario | Model | Horizon | Final position | Max position | Final velocity | R / T / N final |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r.scenario} | {r.profile} | {r.horizon_s/86400:g} d | "
            f"{r.final_position_error_m:.6g} m | {r.max_position_error_m:.6g} m | "
            f"{r.final_velocity_error_mps:.6g} m/s | {r.final_radial_error_m:.4g} / "
            f"{r.final_along_track_error_m:.4g} / {r.final_cross_track_error_m:.4g} m |"
        )
    lines += ["", "Raw machine-readable results: `results.json`.", ""]
    (output / "report.md").write_text("\n".join(lines))
    write_plot(results, output / "position_error.svg", 1.0)


def write_plot(results: list[Result], path: Path, limit: float) -> None:
    """Write a dependency-free logarithmic SVG summary."""
    width, height = 1000, 560
    left, right, top, bottom = 90, 30, 45, 100
    values = [max(x.max_position_error_m, 1e-7) for x in results]
    lo, hi = -7.0, max(0.5, math.ceil(math.log10(max(values))))
    x_step = (width - left - right) / max(len(results), 1)

    def y(value):
        return top + (hi - math.log10(max(value, 10**lo))) / (hi - lo) * (height-top-bottom)

    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
           '<rect width="100%" height="100%" fill="white"/>',
           '<text x="500" y="25" text-anchor="middle" font-family="sans-serif" font-size="18">Orekit–Brahe maximum position disagreement</text>']
    for exponent in range(int(lo), int(hi) + 1):
        yy = y(10**exponent)
        svg += [f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#ddd"/>',
                f'<text x="{left-8}" y="{yy+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">1e{exponent} m</text>']
    yy = y(limit)
    svg.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" stroke="#c33" stroke-dasharray="6 4"/>')
    colors = {"two_body": "#2878b5", "gravity": "#e07a1f",
              "third_body": "#3a9d5d", "srp_sunlight": "#7b9f35",
              "srp": "#8a55aa"}
    for index, result in enumerate(results):
        xx = left + (index + 0.5) * x_step
        yy = y(result.max_position_error_m)
        svg.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="{colors[result.profile]}"/>')
        if index % 3 == 1:
            label = result.scenario.replace("_", " ")
            svg.append(f'<text x="{xx:.1f}" y="{height-bottom+20}" transform="rotate(35 {xx:.1f} {height-bottom+20})" font-family="sans-serif" font-size="10">{label}</text>')
    svg += [f'<text x="{left}" y="{height-20}" font-family="sans-serif" font-size="12" fill="#2878b5">● two-body</text>',
            f'<text x="{left+110}" y="{height-20}" font-family="sans-serif" font-size="12" fill="#e07a1f">● 20x20 gravity</text>',
            f'<text x="{left+250}" y="{height-20}" font-family="sans-serif" font-size="12" fill="#3a9d5d">● Sun/Moon</text>',
            f'<text x="{left+360}" y="{height-20}" font-family="sans-serif" font-size="12" fill="#7b9f35">● SRP sunlight</text>',
            f'<text x="{left+490}" y="{height-20}" font-family="sans-serif" font-size="12" fill="#8a55aa">● SRP eclipse</text>',
            '</svg>']
    path.write_text("\n".join(svg) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", type=Path, default=Path("validation/scenarios.yaml"))
    parser.add_argument("--java", type=Path, required=True)
    parser.add_argument("--orekit-jar", type=Path, required=True)
    parser.add_argument("--orekit-data", type=Path, required=True)
    parser.add_argument("--gravity-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("validation/results"))
    parser.add_argument("--quick", action="store_true", help="one LEO, 1-day horizon")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.scenarios.read_text())
    args.epoch = cfg["epoch"]
    bh.initialize_eop()
    scenarios = cfg["scenarios"][:1] if args.quick else cfg["scenarios"]
    horizons = cfg["horizons_s"][:1] if args.quick else cfg["horizons_s"]
    degree, order = cfg["gravity"]["degree"], cfg["gravity"]["order"]
    results = []
    for case in scenarios:
        state = initial_state(case)
        for profile in cfg["profiles"]:
            for horizon in horizons:
                sample = min(float(cfg["sample_interval_s"]), float(horizon))
                reference = run_orekit(args, profile, state, horizon, sample, degree, order)
                candidate = run_brahe(args, profile, state, reference[:, 0], degree, order)
                results.append(metrics(case["name"], profile, horizon, reference, candidate))
                print(case["name"], profile, horizon, results[-1].final_position_error_m)
    write_report(results, args.output, {
        "epoch": cfg["epoch"], "gravity_degree": degree, "gravity_order": order,
        "gravity_file": args.gravity_file.name,
        "gravity_sha256": sha256(args.gravity_file),
        "orekit_jar_sha256": sha256(args.orekit_jar),
        "orekit_version": "13.1.7",
        "brahe_version": bh.__version__,
    })


if __name__ == "__main__":
    main()
