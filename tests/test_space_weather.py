"""Tests for reproducible empirical-atmosphere inputs."""

from datetime import datetime
import hashlib

import pytest

from core.space_weather import SpaceWeatherSeries


def _write_table(path):
    content = ("date,f107,f107a,ap\n"
               "2024-05-10,190.0,150.0,80.0\n"
               "2024-05-11,200.0,151.0,120.0\n")
    path.write_text(content)
    return hashlib.sha256(content.encode()).hexdigest()


def test_space_weather_checksum_lookup_and_staleness(tmp_path):
    source = tmp_path / "indices.csv"
    digest = _write_table(source)
    series = SpaceWeatherSeries.from_csv(source, expected_sha256=digest,
                                         max_age_days=1)
    record = series.at(datetime(2024, 5, 12, 12))
    assert record.f107 == 200.0
    assert record.ap == 120.0
    assert series.source_sha256 == digest
    with pytest.raises(LookupError, match="stale"):
        series.at(datetime(2024, 5, 13))


def test_space_weather_rejects_checksum_and_order_errors(tmp_path):
    source = tmp_path / "indices.csv"
    _write_table(source)
    with pytest.raises(ValueError, match="checksum"):
        SpaceWeatherSeries.from_csv(source, expected_sha256="0" * 64)
    source.write_text("date,f107,f107a,ap\n2024-05-11,100,100,4\n2024-05-10,100,100,4\n")
    with pytest.raises(ValueError, match="increasing"):
        SpaceWeatherSeries.from_csv(source)
