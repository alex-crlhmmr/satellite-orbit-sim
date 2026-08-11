#!/usr/bin/env python3
"""Run atmosphere models through identical frozen real-orbit protocols."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+",
                        default=["exponential", "harris_priester", "nrlmsise00"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_protocol = yaml.safe_load(args.protocol.read_text())
    summaries = []
    for model in args.models:
        protocol = dict(base_protocol)
        protocol["atmosphere_model"] = model
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as stream:
            yaml.safe_dump(protocol, stream)
            temporary_protocol = Path(stream.name)
        model_output = args.output / model
        try:
            subprocess.run([
                sys.executable, "validation/real_data/benchmark.py",
                "--data", str(args.data), "--manifest", str(args.manifest),
                "--protocol", str(temporary_protocol), "--output", str(model_output),
            ], check=True, stdout=subprocess.DEVNULL)
        finally:
            temporary_protocol.unlink(missing_ok=True)
        result = json.loads((model_output / "results.json").read_text())
        def mean_rms(split: str) -> float:
            values = [row["rms_position_m"] for row in result["metrics"]
                      if row["split"] == split and row["model"] == "fitted"]
            return sum(values) / len(values)
        summaries.append({
            "model": model,
            "passed": result["passed"],
            "fitted_cda_over_mass_m2_per_kg": result["fitted_cda_over_mass_m2_per_kg"],
            "validation_rms_improvement_fraction": result["validation_rms_improvement_fraction"],
            "storm_test_rms_change_fraction": result["storm_test_rms_change_fraction"],
            "fitted_validation_mean_rms_m": mean_rms("validation"),
            "fitted_test_mean_rms_m": mean_rms("test"),
        })
    payload = {"frozen_manifest": str(args.manifest), "models": summaries}
    (args.output / "comparison.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
