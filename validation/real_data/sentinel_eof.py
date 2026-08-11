"""Strict parser and frame conversion for Copernicus POD EOF products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET

import brahe as bh
import numpy as np


@dataclass(frozen=True)
class OrbitArc:
    epochs: tuple[datetime, ...]
    states_itrf: np.ndarray

    def downsample(self, interval_s: int, duration_s: int = 86400) -> "OrbitArc":
        source_step = round((self.epochs[1] - self.epochs[0]).total_seconds())
        if interval_s % source_step:
            raise ValueError("requested interval must be a multiple of source sampling")
        stride = interval_s // source_step
        count = duration_s // interval_s + 1
        return OrbitArc(self.epochs[:stride * count:stride],
                        self.states_itrf[:stride * count:stride].copy())

    def states_gcrf(self) -> np.ndarray:
        rows = []
        for epoch, state in zip(self.epochs, self.states_itrf):
            rows.append(bh.state_itrf_to_gcrf(bh.Epoch(epoch), state))
        return np.asarray(rows)


def read_eof(path: Path) -> OrbitArc:
    root = ET.parse(path).getroot()
    header_frame = root.findtext("./Earth_Explorer_Header/Variable_Header/Ref_Frame")
    header_time = root.findtext("./Earth_Explorer_Header/Variable_Header/Time_Reference")
    if header_frame != "EARTH_FIXED" or header_time != "UTC":
        raise ValueError(f"unsupported EOF frame/time: {header_frame}/{header_time}")
    epochs, states = [], []
    for osv in root.findall(".//OSV"):
        timestamp = osv.findtext("UTC")
        if timestamp is None or not timestamp.startswith("UTC="):
            raise ValueError("missing UTC timestamp")
        epochs.append(datetime.fromisoformat(timestamp[4:]).replace(tzinfo=timezone.utc))
        states.append([float(osv.findtext(key)) for key in ("X", "Y", "Z", "VX", "VY", "VZ")])
    if len(epochs) < 2 or any((b-a).total_seconds() != 10 for a, b in zip(epochs, epochs[1:])):
        raise ValueError("EOF orbit vectors are not a continuous 10-second series")
    array = np.asarray(states, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("EOF contains non-finite state values")
    return OrbitArc(tuple(epochs), array)

