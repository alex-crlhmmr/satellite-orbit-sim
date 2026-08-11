"""Versionable space-weather inputs for empirical atmosphere models."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
from pathlib import Path


@dataclass(frozen=True)
class DailySpaceWeather:
    date: date
    f107: float
    f107a: float
    ap: float


class SpaceWeatherSeries:
    """Strict daily F10.7/F10.7A/Ap table loaded from a checksummable CSV."""

    def __init__(self, records: list[DailySpaceWeather], source_sha256: str,
                 max_age_days: int = 1) -> None:
        if not records:
            raise ValueError("space-weather series is empty")
        if max_age_days < 0:
            raise ValueError("max_age_days must be nonnegative")
        dates = [record.date for record in records]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise ValueError("space-weather dates must be unique and increasing")
        self._records = {record.date: record for record in records}
        self._dates = dates
        self.source_sha256 = source_sha256
        self.max_age_days = int(max_age_days)

    @classmethod
    def from_csv(cls, path: str | Path, max_age_days: int = 1,
                 expected_sha256: str | None = None) -> "SpaceWeatherSeries":
        path = Path(path)
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError(f"space-weather checksum mismatch: {digest}")
        rows = []
        for row in csv.DictReader(raw.decode("utf-8").splitlines()):
            record = DailySpaceWeather(
                datetime.strptime(row["date"], "%Y-%m-%d").date(),
                float(row["f107"]), float(row["f107a"]), float(row["ap"]),
            )
            if record.f107 <= 0 or record.f107a <= 0 or record.ap < 0:
                raise ValueError("space-weather indices are outside physical bounds")
            rows.append(record)
        return cls(rows, digest, max_age_days)

    def at(self, instant: datetime) -> DailySpaceWeather:
        target = instant.date()
        candidates = [day for day in self._dates if day <= target]
        if not candidates:
            raise LookupError(f"no space-weather record on or before {target}")
        selected = candidates[-1]
        age = (target - selected).days
        if age > self.max_age_days:
            raise LookupError(
                f"space-weather record for {target} is stale by {age} days"
            )
        return self._records[selected]
