from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext

Series = tuple[Any, ...]


@dataclass(frozen=True)
class LagOperation(Operation):
    """Adds earlier values of a field as new fields: <field>_lag_<k> is its value k time steps before the row.

    JSON: {"op": "lag", "field": "kw", "lags": [1, 2, 24], "by": "mg"}
    The time step is the interval of the rows, so reduce them with a window first: with 1h rows, kw_lag_24 is the
    value 24 hours earlier. Rows that lack one of the earlier values (the first ones, or after a gap) are dropped,
    so the expressions of later steps never meet a missing lag. `by` keeps series apart (e.g. microgrids put
    together with union): a row only gets values from the rows with the same `by`.
    """

    name: ClassVar[str] = "lag"
    field: str
    lags: tuple[int, ...]
    by: tuple[str, ...] = ()

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        if rows.interval is None:
            raise ChainError("the time step of the rows is unknown; reduce them with a window first, e.g. 1h")
        if rows.items and not any(self.field in row for row in rows.items):
            raise ChainError(f"the rows have no field {self.field}")
        values = self._values(rows.items)
        kept = []
        for row in rows.items:
            lagged = self._lagged(row, values[self._series(row)], rows.interval)
            if lagged is not None:
                kept.append({**row, **lagged})
        return rows.with_items(kept)

    def _lagged(self, row: Row, earlier: dict[datetime, Any], interval: timedelta) -> dict[str, Any] | None:
        """The new fields of a row, or None if one of the earlier values is missing."""
        lagged = {}
        for lag in self.lags:
            value = earlier.get(row["ts"] - lag * interval)
            if value is None:
                return None
            lagged[f"{self.field}_lag_{lag}"] = value
        return lagged

    def _values(self, items: list[Row]) -> dict[Series, dict[datetime, Any]]:
        """For each series, time -> the value of the field (None counts as missing)."""
        values: dict[Series, dict[datetime, Any]] = {}
        seen: set[tuple[Series, datetime]] = set()
        for row in items:
            time = row.get("ts")
            if not isinstance(time, datetime):
                raise ChainError("lag needs a time (ts) in every row")
            series = self._series(row)
            if (series, time) in seen:
                hint = "" if self.by else ', or name the field that tells the series apart with "by"'
                raise ChainError(f"two rows have the time {time.isoformat()}; lag needs one row per time{hint}")
            seen.add((series, time))
            values.setdefault(series, {})[time] = row.get(self.field)
        return values

    def _series(self, row: Row) -> Series:
        for name in self.by:
            if name not in row:
                raise ChainError(f"unknown field {name}")
        key = tuple(row[name] for name in self.by)
        try:
            hash(key)
        except TypeError:
            raise ChainError(f"by cannot be a list or an object: {key}") from None
        return key
