"""Time windows of reduce: "15m", "1h", "1d", "7d" (DESIGN.md section 5)."""

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from src.exceptions import ChainError
from src.models.time import Timestamp

_WINDOW = re.compile(r"([1-9][0-9]*)([mhd])")
_DAY = timedelta(days=1)
_UNITS = {"m": timedelta(minutes=1), "h": timedelta(hours=1), "d": _DAY}
_ALLOWED_DAYS = (1, 7)

_FIRST_MONDAY = datetime(2000, 1, 3)
"""Windows are counted from a Monday at midnight: windows that divide a day start at midnight, 7d windows on
Mondays."""


@dataclass(frozen=True)
class Window:
    """A tumbling time window."""

    text: str
    length: timedelta

    @classmethod
    def parse(cls, text: str) -> "Window":
        """Minutes or hours that divide a day (15m, 30m, 1h, 4h, ...), or 1d, or 7d."""
        match = _WINDOW.fullmatch(text)
        if match is None:
            raise ChainError(f"cannot read the window {text!r}; write it like 15m, 1h or 1d")
        count, unit = int(match[1]), match[2]
        length = count * _UNITS[unit]
        if unit == "d" and count not in _ALLOWED_DAYS:
            raise ChainError(f"a window in days must be 1d or 7d, not {text}")
        if unit != "d" and _DAY % length:
            raise ChainError(f"a window must divide a day (15m, 30m, 1h, 4h, ...), and {text} does not")
        return cls(text, length)

    @property
    def minutes(self) -> int:
        return int(self.length.total_seconds() // 60)

    def start_of(self, time: datetime) -> Timestamp:
        """The start of the window that holds `time`."""
        windows_before = (time - _FIRST_MONDAY) // self.length
        return Timestamp.of(_FIRST_MONDAY + windows_before * self.length)
