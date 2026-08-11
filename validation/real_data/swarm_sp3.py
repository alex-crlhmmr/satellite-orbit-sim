"""Strict reader for ESA Swarm reduced-dynamic SP3 ZIP products."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np

from validation.real_data.sentinel_eof import OrbitArc


def _timestamp(fields: list[str]) -> datetime:
    year, month, day, hour, minute = map(int, fields[:5])
    second = float(fields[5])
    whole = int(second)
    return datetime(year, month, day, hour, minute, whole,
                    round((second - whole) * 1e6), tzinfo=timezone.utc)


def read_sp3_zip(path: Path) -> OrbitArc:
    """Read one Swarm COM SP3 archive and return SI states in ITRF/UTC.

    SP3 epochs use GPS time.  The accompanying ESA header states the UTC
    validity start, so the conversion offset is derived from the product
    itself instead of freezing a leap-second constant in code.
    """
    with ZipFile(path) as archive:
        names = archive.namelist()
        sp3_names = [name for name in names if name.lower().endswith(".sp3")]
        hdr_names = [name for name in names if name.lower().endswith(".hdr")]
        if len(sp3_names) != 1 or len(hdr_names) != 1:
            raise ValueError("Swarm archive must contain exactly one SP3 and one HDR")
        header = ET.fromstring(archive.read(hdr_names[0]))
        validity = header.findtext("./Fixed_Header/Validity_Period/Validity_Start")
        if not validity or not validity.startswith("UTC="):
            raise ValueError("Swarm header has no UTC validity start")
        utc_start = datetime.fromisoformat(validity[4:]).replace(tzinfo=timezone.utc)
        lines = archive.read(sp3_names[0]).decode("ascii").splitlines()

    if not lines or "GPS" not in next((line for line in lines if line.startswith("%c")), ""):
        raise ValueError("SP3 time system is not GPS")
    epochs_gps: list[datetime] = []
    positions: list[list[float]] = []
    velocities: list[list[float]] = []
    for line in lines:
        if line.startswith("*"):
            epochs_gps.append(_timestamp(line[1:].split()))
        elif line.startswith("P"):
            positions.append([float(line[4:18]), float(line[18:32]), float(line[32:46])])
        elif line.startswith("V"):
            velocities.append([float(line[4:18]), float(line[18:32]), float(line[32:46])])
    if len(epochs_gps) < 2 or not (len(epochs_gps) == len(positions) == len(velocities)):
        raise ValueError("SP3 has incomplete position/velocity records")
    gps_utc = epochs_gps[0] - utc_start
    if not 0 <= gps_utc.total_seconds() <= 60:
        raise ValueError("implausible GPS-UTC offset in Swarm product")
    epochs = tuple(epoch - gps_utc for epoch in epochs_gps)
    if any(abs((b - a).total_seconds() - 10.0) > 1e-6 for a, b in zip(epochs, epochs[1:])):
        raise ValueError("SP3 orbit vectors are not a continuous 10-second series")
    # SP3 positions are km and velocities are decimetres/second.
    states = np.column_stack((np.asarray(positions) * 1000.0,
                              np.asarray(velocities) * 0.1))
    if not np.isfinite(states).all():
        raise ValueError("SP3 contains non-finite state values")
    return OrbitArc(epochs, states)
