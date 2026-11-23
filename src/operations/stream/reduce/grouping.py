"""How reduce groups its rows: by time window, by keys, or all rows in one group (DESIGN.md section 5)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from src.exceptions import ChainError
from src.expressions import Expression
from src.models.rows import Row
from src.operations.stream.reduce.window import Window

GroupKey = tuple[Any, ...]


class Grouping(ABC):
    """One way of grouping. reduce asks it for the key of each row, and whether to keep each group."""

    @property
    @abstractmethod
    def key_names(self) -> tuple[str, ...]:
        """The fields that hold the key in an output row."""

    @abstractmethod
    def key_of(self, row: Row) -> GroupKey:
        """The key of the group the row belongs to."""

    def keeps(self, group: list[Row], interval: timedelta | None) -> bool:
        """Whether the output has a row for this group. `interval` is the time step of the incoming rows."""
        return True

    def interval_after(self) -> timedelta | None:
        """The time step of the output rows."""
        return None


@dataclass(frozen=True)
class WindowGrouping(Grouping):
    """{"window": "15m"}: tumbling windows; the key is ts, the start of the window.

    Only complete windows are kept: a window must hold (window / interval) different times. The interval comes
    from the source as fetched, so a filter that thins the rows makes windows incomplete; it cannot turn single
    raw readings into "means".
    """

    window: Window

    @property
    def key_names(self) -> tuple[str, ...]:
        return ("ts",)

    def key_of(self, row: Row) -> GroupKey:
        time = row.get("ts")
        if not isinstance(time, datetime):
            raise ChainError("a reduce with a window needs rows with a time (ts), and a row has none")
        return (self.window.start_of(time),)

    def keeps(self, group: list[Row], interval: timedelta | None) -> bool:
        if interval is None:
            return False
        needed = self.window.length // interval
        return len({row["ts"] for row in group}) >= needed

    def interval_after(self) -> timedelta | None:
        return self.window.length


@dataclass(frozen=True)
class GroupKeyField:
    """One key of a reduce with "by": a field, or an expression whose value is stored under `name`."""

    name: str
    expression: Expression | None = None  # None: the value of the field `name`

    def value(self, row: Row) -> Any:
        if self.expression is not None:
            return self.expression.evaluate(row)
        if self.name not in row:
            raise ChainError(f"unknown field {self.name}")
        return row[self.name]


@dataclass(frozen=True)
class KeyGrouping(Grouping):
    """{"by": "hour"}, {"by": ["weekday", "hour"]} or {"by": {"slot": "ts.slot()"}}: one group per key."""

    keys: tuple[GroupKeyField, ...]

    @property
    def key_names(self) -> tuple[str, ...]:
        return tuple(key.name for key in self.keys)

    def key_of(self, row: Row) -> GroupKey:
        values = tuple(key.value(row) for key in self.keys)
        try:
            hash(values)
        except TypeError:
            raise ChainError(f"a key of reduce cannot be a list or an object: {values!r}") from None
        return values


class AllRows(Grouping):
    """Neither "window" nor "by": all the rows form one group, and the output is one row (no rows if no input)."""

    @property
    def key_names(self) -> tuple[str, ...]:
        return ()

    def key_of(self, row: Row) -> GroupKey:
        return ()
