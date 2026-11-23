"""reduce: grouping, aggregates and windows, each in its own module.

- window.py      Window: "15m", "1h", "1d", "7d"
- grouping.py    WindowGrouping, KeyGrouping, AllRows
- criteria.py    mean, sum, count, min, max
- aggregates.py  the entries of "aggregate" and what they compute
- operation.py   ReduceOperation
"""

from src.operations.stream.reduce.aggregates import parse_aggregate
from src.operations.stream.reduce.grouping import AllRows, GroupKeyField, KeyGrouping, WindowGrouping
from src.operations.stream.reduce.operation import ReduceOperation
from src.operations.stream.reduce.window import Window

__all__ = [
    "AllRows",
    "GroupKeyField",
    "KeyGrouping",
    "ReduceOperation",
    "Window",
    "WindowGrouping",
    "parse_aggregate",
]
