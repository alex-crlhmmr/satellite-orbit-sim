"""Strict streaming reader for TU Delft GRACE-FO density ZIP products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class DensityObservation:
    epoch_gps: datetime
    altitude_m: float
    longitude_deg: float
    latitude_deg: float
    density_kg_m3: float


def read_density_zip(path: Path, stride: int = 1):
    if stride < 1:
        raise ValueError("stride must be positive")
    with ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith(".txt")]
        if len(names) != 1:
            raise ValueError("density archive must contain exactly one text product")
        with archive.open(names[0]) as stream:
            accepted_index = 0
            previous = None
            for raw in stream:
                line = raw.decode("ascii").strip()
                if not line or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) != 12 or fields[2] != "GPS":
                    raise ValueError("unexpected GRACE-FO density record format")
                epoch = datetime.fromisoformat(f"{fields[0]}T{fields[1]}")
                if previous is not None and (epoch - previous).total_seconds() != 10:
                    raise ValueError("density records are not a continuous 10-second series")
                previous = epoch
                density_flag = float(fields[10])
                if density_flag != 0.0:
                    continue
                if accepted_index % stride == 0:
                    density = float(fields[8])
                    if density <= 0:
                        raise ValueError("nominal density must be positive")
                    yield DensityObservation(epoch, float(fields[3]), float(fields[4]),
                                             float(fields[5]), density)
                accepted_index += 1
