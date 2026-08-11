#!/usr/bin/env python3
"""Evidence-gated drag calibration against Sentinel-1A precise orbits."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import brahe as bh
import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from validation.real_data.sentinel_eof import OrbitArc, read_eof
from validation.real_data.fetch import sha256


@dataclass
class ArcMetrics:
    filename: str
    split: str
    regime: str
    model: str
    drag_area_m2: float
    final_position_m: float
    rms_position_m: float
    p95_position_m: float
    final_radial_m: float
    final_along_track_m: float
    final_cross_track_m: float
    rms_along_track_m: float


def force_model(config: dict, drag_area: float) -> bh.ForceModelConfig:
    fixed = bh.ParameterSource.value
    spacecraft = config["spacecraft"]
    drag = None if drag_area == 0 else bh.DragConfiguration(
        bh.AtmosphericModel.NRLMSISE00, fixed(drag_area),
        fixed(spacecraft["drag_coefficient"]),
    )
    srp = bh.SolarRadiationPressureConfiguration(
        fixed(spacecraft["srp_area_m2"]),
        fixed(spacecraft["reflectivity_coefficient"]),
        bh.EclipseModel.CONICAL,
    )
    tides = bh.TidesConfiguration(
        bh.PermanentTideConfig.AUTO,
        bh.SolidTideConfig(frequency_dependent=True, pole_tide=True),
        None, bh.EphemerisSource.DE440s,
    )
    return bh.ForceModelConfig(
        gravity=bh.GravityConfiguration(degree=20, order=20),
        drag=drag, srp=srp,
        third_body=[
            bh.ThirdBodyConfiguration(bh.ThirdBody.SUN, bh.EphemerisSource.DE440s),
            bh.ThirdBodyConfiguration(bh.ThirdBody.MOON, bh.EphemerisSource.DE440s),
        ],
        relativity=True, mass=fixed(spacecraft["mass_kg"]), tides=tides,
        frame_transform=bh.FrameTransformationModel.FULL_EARTH_ROTATION,
    )


def propagate(truth: OrbitArc, config: dict, drag_area: float) -> tuple[np.ndarray, np.ndarray]:
    truth_gcrf = truth.states_gcrf()
    initial_epoch = bh.Epoch(truth.epochs[0])
    propagation = (
        bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF78)
        .with_abs_tol(1e-6).with_rel_tol(1e-11).with_max_step(60.0)
    )
    propagator = bh.NumericalOrbitPropagator(
        initial_epoch, truth_gcrf[0], propagation, force_model(config, drag_area)
    )
    epochs = [bh.Epoch(timestamp) for timestamp in truth.epochs]
    propagator.propagate_to(epochs[-1])
    candidate = np.asarray(propagator.states_gcrf(epochs), dtype=np.float64)
    return truth_gcrf, candidate


def rtn_errors(truth: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    errors = candidate[:, :3] - truth[:, :3]
    result = []
    for state, error in zip(truth, errors):
        radial = state[:3] / np.linalg.norm(state[:3])
        normal = np.cross(state[:3], state[3:])
        normal /= np.linalg.norm(normal)
        transverse = np.cross(normal, radial)
        result.append([radial @ error, transverse @ error, normal @ error])
    return np.asarray(result)


def summarize(entry: dict, model: str, area: float,
              truth: np.ndarray, candidate: np.ndarray) -> ArcMetrics:
    position = np.linalg.norm(candidate[:, :3] - truth[:, :3], axis=1)
    rtn = rtn_errors(truth, candidate)
    return ArcMetrics(
        entry["filename"], entry["split"], entry["regime"], model, area,
        float(position[-1]), float(np.sqrt(np.mean(position**2))),
        float(np.percentile(position, 95)), *map(float, rtn[-1]),
        float(np.sqrt(np.mean(rtn[:, 1]**2))),
    )


def fit_effective_area(training: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                       calibration_area: float, bounds: list[float]) -> tuple[float, list[float]]:
    """Robust median of per-arc linearized least-squares area estimates.

    Orbit-maintenance maneuvers are not described by a ballistic drag model.
    A median prevents a maneuver-contaminated training arc from dominating the
    estimate, without looking at validation or test outcomes.
    """
    estimates = []
    for truth, no_drag, calibrated in training:
        response = ((calibrated[:, :3] - no_drag[:, :3]) / calibration_area).reshape(-1)
        observation = (truth[:, :3] - no_drag[:, :3]).reshape(-1)
        estimates.append(float(response @ observation / (response @ response)))
    area = float(np.median(estimates))
    return float(np.clip(area, bounds[0], bounds[1])), estimates


def write_report(output: Path, payload: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = ["# Sentinel-1A drag validation", "",
             f"Evidence gate: **{'PASS' if payload['passed'] else 'FAIL'}**", "",
             f"Fitted effective drag area: {payload['fitted_drag_area_m2']:.4f} m²", "",
             f"Equivalent density/ballistic scale versus the 10 m² nominal baseline: "
             f"{payload['effective_drag_scale']:.4f}", "",
             "> This is an effective drag scale, not a recovered physical surface area. "
             "It absorbs density, attitude, coefficient, and unmodelled-force errors.", "",
             "| Arc | Split | Regime | Model | RMS 3D | RMS along-track | Final 3D |",
             "|---|---|---|---:|---:|---:|---:|"]
    for row in payload["metrics"]:
        arc = row["filename"].split("_V", 1)[-1][:8]
        lines.append(f"| {arc} | {row['split']} | {row['regime']} | {row['model']} | "
                     f"{row['rms_position_m']:.3f} m | {row['rms_along_track_m']:.3f} m | "
                     f"{row['final_position_m']:.3f} m |")
    lines += ["", "Parameters are fitted only on `train`; validation and storm test arcs are locked.", ""]
    (output / "report.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("validation/real_data/manifest.yaml"))
    parser.add_argument("--protocol", type=Path, default=Path("validation/real_data/protocol.yaml"))
    parser.add_argument("--output", type=Path, default=Path("validation/real_data/results"))
    args = parser.parse_args()
    manifest, protocol = yaml.safe_load(args.manifest.read_text()), yaml.safe_load(args.protocol.read_text())
    bh.initialize_eop(); bh.initialize_sw()
    arcs = []
    for entry in manifest["files"]:
        source = args.data / entry["filename"]
        if sha256(source) != entry["sha256"]:
            raise RuntimeError(f"manifest checksum mismatch: {source}")
        arc = read_eof(source).downsample(
            protocol["sample_interval_s"], protocol["arc_duration_s"]
        )
        arcs.append((entry, arc))

    calibration_area = protocol["spacecraft"]["calibration_area_m2"]
    training = []
    for entry, arc in arcs:
        if entry["split"] != "train":
            continue
        truth, no_drag = propagate(arc, protocol, 0.0)
        _, calibrated = propagate(arc, protocol, calibration_area)
        training.append((truth, no_drag, calibrated))
    limits = protocol["acceptance"]
    fitted_area, training_area_estimates = fit_effective_area(
        training, calibration_area, limits["fitted_drag_area_bounds_m2"]
    )

    metrics = []
    nominal_area = protocol["spacecraft"]["nominal_drag_area_m2"]
    for entry, arc in arcs:
        for model, area in (("nominal", nominal_area), ("fitted", fitted_area)):
            truth, candidate = propagate(arc, protocol, area)
            metrics.append(summarize(entry, model, area, truth, candidate))

    def mean_rms(split: str, model: str) -> float:
        values = [x.rms_position_m for x in metrics if x.split == split and x.model == model]
        return float(np.mean(values))
    validation_gain = 1.0 - mean_rms("validation", "fitted") / mean_rms("validation", "nominal")
    storm_change = mean_rms("test", "fitted") / mean_rms("test", "nominal") - 1.0
    passed = (validation_gain >= limits["validation_rms_improvement_min_fraction"] and
              storm_change <= limits["storm_test_rms_degradation_max_fraction"] and
              limits["fitted_drag_area_bounds_m2"][0] < fitted_area < limits["fitted_drag_area_bounds_m2"][1])
    payload = {
        "protocol_version": protocol["protocol_version"], "passed": bool(passed),
        "fitted_drag_area_m2": fitted_area, "validation_rms_improvement_fraction": validation_gain,
        "effective_drag_scale": fitted_area / nominal_area,
        "dataset_files": [{"filename": x["filename"], "sha256": x["sha256"],
                           "split": x["split"], "regime": x["regime"]}
                          for x in manifest["files"]],
        "training_area_estimates_m2": training_area_estimates,
        "storm_test_rms_change_fraction": storm_change, "metrics": [asdict(x) for x in metrics],
    }
    write_report(args.output, payload)
    print(json.dumps({k: payload[k] for k in payload if k != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
