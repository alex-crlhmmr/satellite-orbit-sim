#!/usr/bin/env python3
"""Force-level density validation against GRACE-FO accelerometer products."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import brahe as bh
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.density_estimation import DensityScaleFilter
from validation.density.gracefo import DensityObservation, read_density_zip
from validation.real_data.fetch import sha256


def _as_date(value) -> date:
    return value if isinstance(value, date) else date.fromisoformat(value)


def _in_ranges(day: date, ranges) -> bool:
    return any(_as_date(start) <= day < _as_date(stop) for start, stop in ranges)


def load_observations(data: Path, manifest: dict, protocol: dict):
    stride = protocol["sample_interval_s"] // manifest["dataset"]["sampling_s"]
    result = {name: [] for name in protocol["splits"]}
    for entry in manifest["files"]:
        source = data / entry["filename"]
        if sha256(source) != entry["sha256"]:
            raise RuntimeError(f"manifest checksum mismatch: {source}")
        for observation in read_density_zip(source, stride):
            for split, ranges in protocol["splits"].items():
                if _in_ranges(observation.epoch_gps.date(), ranges):
                    result[split].append(observation)
                    break
    if any(not values for values in result.values()):
        raise RuntimeError("one or more density splits are empty")
    return result


def model_density(observation: DensityObservation) -> float:
    instant = observation.epoch_gps
    epoch = bh.Epoch.from_date(instant.year, instant.month, instant.day,
                               bh.TimeSystem.GPS)
    epoch += (instant.hour * 3600 + instant.minute * 60 +
              instant.second + instant.microsecond / 1e6)
    geodetic = np.array([observation.longitude_deg, observation.latitude_deg,
                         observation.altitude_m])
    return float(bh.density_nrlmsise00_geod(epoch, geodetic))


def mape(truth: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.abs(prediction / truth - 1.0)))


def run_online(observations, modeled, initial_scale, initial_variance,
               process_noise, relative_sigma):
    estimator = DensityScaleFilter(initial_scale, initial_variance, process_noise)
    start = observations[0].epoch_gps
    predictions, nis = [], []
    for observation, sensitivity in zip(observations, modeled):
        elapsed = (observation.epoch_gps - start).total_seconds()
        estimator.predict(elapsed)
        predictions.append(sensitivity * estimator.scale)
        variance = (relative_sigma * observation.density_kg_m3) ** 2
        estimate = estimator.update(observation.density_kg_m3, sensitivity, variance)
        nis.append(estimator.normalized_innovation_squared(estimate))
    return np.asarray(predictions), np.asarray(nis)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path,
                        default=Path("validation/density/manifest.yaml"))
    parser.add_argument("--protocol", type=Path,
                        default=Path("validation/density/protocol.yaml"))
    parser.add_argument("--output", type=Path,
                        default=Path("validation/density/results"))
    args = parser.parse_args()
    manifest = yaml.safe_load(args.manifest.read_text())
    protocol = yaml.safe_load(args.protocol.read_text())
    bh.initialize_eop()
    weather_entry = manifest["space_weather"]
    weather_file = args.data / weather_entry["filename"]
    if sha256(weather_file) != weather_entry["sha256"]:
        raise RuntimeError(f"space-weather checksum mismatch: {weather_file}")
    weather_provider = bh.FileSpaceWeatherProvider.from_file(
        str(weather_file), extrapolate="Error"
    )
    bh.set_global_space_weather_provider(weather_provider)
    observations = load_observations(args.data, manifest, protocol)
    modeled = {split: np.asarray([model_density(item) for item in values])
               for split, values in observations.items()}
    truth = {split: np.asarray([item.density_kg_m3 for item in values])
             for split, values in observations.items()}

    training_ratios = truth["train"] / modeled["train"]
    static_scale = float(np.median(training_ratios))
    initial_variance = float(max(np.var(training_ratios), 1e-6))
    relative_sigma = protocol["observation_relative_sigma"]
    candidates = []
    for process_noise in protocol["process_noise_candidates_per_s"]:
        prediction, nis = run_online(
            observations["validation"], modeled["validation"], static_scale,
            initial_variance, process_noise, relative_sigma,
        )
        candidates.append((mape(truth["validation"], prediction),
                           float(process_noise), float(np.mean(nis))))
    validation_mape, selected_noise, validation_nis = min(candidates)

    metrics = {}
    online = {}
    for split in ("validation", "test"):
        raw_mape = mape(truth[split], modeled[split])
        static_mape = mape(truth[split], modeled[split] * static_scale)
        prediction, nis = run_online(
            observations[split], modeled[split], static_scale, initial_variance,
            selected_noise, relative_sigma,
        )
        online_mape = mape(truth[split], prediction)
        metrics[split] = {
            "samples": len(truth[split]), "raw_mape": raw_mape,
            "static_mape": static_mape, "online_mape": online_mape,
            "static_improvement_fraction": 1.0 - static_mape / raw_mape,
            "online_improvement_fraction": 1.0 - online_mape / raw_mape,
            "online_mean_nis": float(np.mean(nis)),
        }
        online[split] = prediction
    limits = protocol["acceptance"]
    nis_low, nis_high = limits["online_test_mean_nis_bounds"]
    test = metrics["test"]
    passed = (
        test["static_improvement_fraction"] >= limits["static_test_mape_improvement_min_fraction"]
        and test["online_improvement_fraction"] >= limits["online_test_mape_improvement_min_fraction"]
        and nis_low <= test["online_mean_nis"] <= nis_high
    )
    payload = {
        "protocol_version": protocol["protocol_version"], "passed": bool(passed),
        "dataset": manifest["dataset"], "files": manifest["files"],
        "space_weather": manifest["space_weather"],
        "training_static_scale": static_scale,
        "training_ratio_median_absolute_deviation": float(
            np.median(np.abs(training_ratios - static_scale))
        ),
        "selected_process_noise_psd_per_s": selected_noise,
        "validation_selection_mape": validation_mape,
        "validation_selection_mean_nis": validation_nis,
        "metrics": metrics,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        "# GRACE-FO force-level density validation", "",
        f"Evidence gate: **{'PASS' if passed else 'FAIL'}**", "",
        f"Training-only static scale: `{static_scale:.6f}`", "",
        f"Validation-selected process-noise PSD: `{selected_noise:.2e} s⁻¹`", "",
        "| Split | Raw MAPE | Static MAPE | Online one-step MAPE | Mean NIS |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("validation", "test"):
        row = metrics[split]
        lines.append(
            f"| {split} | {row['raw_mape']:.2%} | {row['static_mape']:.2%} | "
            f"{row['online_mape']:.2%} | {row['online_mean_nis']:.3f} |"
        )
    lines += ["", "Online predictions are evaluated before assimilating the observation at that epoch.",
              "Process noise is selected on validation only; October is untouched test data.", ""]
    (args.output / "report.md").write_text("\n".join(lines))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
