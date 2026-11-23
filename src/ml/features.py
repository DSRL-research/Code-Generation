"""The inputs of a model: columns of the rows, made by the steps of the chain before train (lag, map; DESIGN.md
section 17). ColumnSpec says which; column_table and row_inputs read them as numbers."""

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.models.rows import Row


@dataclass(frozen=True)
class ColumnSpec:
    """The inputs of a model: the `features` columns of the rows. `interval` is the time step of the rows it was
    trained on (None if unknown)."""

    target: str
    features: tuple[str, ...]
    interval: timedelta | None

    def to_json(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "features": list(self.features),
            "interval_minutes": self.interval.total_seconds() / 60 if self.interval is not None else None,
        }


def column_table(
    items: list[Row], features: tuple[str, ...], target: str
) -> tuple[list[list[float]], list[float], list[datetime], int]:
    """X, y, the times (ts) of the rows used, and how many rows were skipped because the target or a feature is
    missing or not a finite number. True and False count as 1 and 0."""
    table, targets, times, skipped = [], [], [], 0
    for row in items:
        values = row_inputs(row, (*features, target))
        if values is None:
            skipped += 1
            continue
        table.append(values[:-1])
        targets.append(values[-1])
        if isinstance(row.get("ts"), datetime):
            times.append(row["ts"])
    return table, targets, times, skipped


def row_inputs(row: Row, names: tuple[str, ...]) -> list[float] | None:
    """The values of `names` in a row as numbers, or None if one is missing or not a finite number."""
    values = [_number(row.get(name)) for name in names]
    return None if any(value is None for value in values) else values  # type: ignore[return-value]


def _number(value: Any) -> float | None:
    if isinstance(value, int | float) and math.isfinite(value):
        return float(value)
    return None
