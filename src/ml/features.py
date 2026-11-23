"""The features of a forecasting model: lags of the target, and the calendar of ts.

lag_k is the value of the target k time steps earlier (k * interval before ts). A calendar feature is written as
its sine and cosine, so 23:00 is close to 00:00 and Sunday close to Monday.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.exceptions import ChainError
from src.models.rows import Row

CALENDAR: dict[str, tuple[int, Any]] = {
    "hour": (24, lambda time: time.hour),
    "weekday": (7, lambda time: time.weekday()),
    "month": (12, lambda time: time.month - 1),
}


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    lags: tuple[int, ...]
    calendar: tuple[str, ...]
    interval: timedelta

    @property
    def names(self) -> list[str]:
        lag_names = [f"lag_{lag}" for lag in self.lags]
        return lag_names + [f"{name}_{part}" for name in self.calendar for part in ("sin", "cos")]

    def features(self, time: datetime, history: Mapping[datetime, float]) -> list[float] | None:
        """The features of the row at `time`, or None if a lag is missing from the history."""
        values = []
        for lag in self.lags:
            value = history.get(time - lag * self.interval)
            if value is None:
                return None
            values.append(float(value))
        for name in self.calendar:
            period, part_of = CALENDAR[name]
            angle = 2 * math.pi * part_of(time) / period
            values += [math.sin(angle), math.cos(angle)]
        return values

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "lags": list(self.lags),
            "calendar": list(self.calendar),
            "interval_minutes": self.interval.total_seconds() / 60,
        }


def history(items: list[Row], target: str) -> dict[datetime, float]:
    """ts -> the value of the target, for the rows that have both."""
    if items and not any(target in row for row in items):
        raise ChainError(f"the rows have no field {target}")
    values = {}
    for row in items:
        time, value = row.get("ts"), row.get(target)
        if not isinstance(time, datetime):
            raise ChainError("the rows need a time (ts) to build lag features")
        if isinstance(value, int | float) and not isinstance(value, bool):
            values[time] = float(value)
    return values


def training_set(spec: FeatureSpec, items: list[Row]) -> tuple[list[list[float]], list[float], list[datetime]]:
    """X, y and the times of the rows that have every lag and a target value, in time order."""
    known = history(items, spec.target)
    features, targets, times = [], [], []
    for time in sorted(known):
        row_features = spec.features(time, known)
        if row_features is not None:
            features.append(row_features)
            targets.append(known[time])
            times.append(time)
    return features, targets, times
