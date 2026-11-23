"""Stream operators: they reshape rows (DESIGN.md section 5).

filter, map, reduce (a package), join, union, top, collect.
"""

from src.operations.stream.collect import CollectOperation
from src.operations.stream.filter import FilterOperation
from src.operations.stream.join import JoinOperation
from src.operations.stream.map import MapOperation
from src.operations.stream.reduce import ReduceOperation
from src.operations.stream.top import TopOperation
from src.operations.stream.union import UnionOperation

STREAM_OPERATIONS = (
    FilterOperation,
    MapOperation,
    ReduceOperation,
    JoinOperation,
    UnionOperation,
    TopOperation,
    CollectOperation,
)

__all__ = [
    "STREAM_OPERATIONS",
    "CollectOperation",
    "FilterOperation",
    "JoinOperation",
    "MapOperation",
    "ReduceOperation",
    "TopOperation",
    "UnionOperation",
]
