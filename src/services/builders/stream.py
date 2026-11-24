from pydantic import BaseModel

from src.models.steps.stream import (
    CollectJson,
    FilterJson,
    JoinJson,
    LagJson,
    MapJson,
    ReduceJson,
    TopJson,
    UnionJson,
)
from src.operations import (
    CollectOperation,
    FilterOperation,
    JoinOperation,
    LagOperation,
    MapOperation,
    Operation,
    ReduceOperation,
    TopOperation,
    UnionOperation,
)
from src.operations.stream.reduce import AllRows, GroupKeyField, KeyGrouping, Window, WindowGrouping, parse_aggregate
from src.operations.stream.reduce.grouping import Grouping
from src.services.builders import Builder, Parsing


class StreamBuilders:
    def __init__(self, parsing: Parsing) -> None:
        self._parsing = parsing

    def builders(self) -> dict[type[BaseModel], Builder]:
        return {
            FilterJson: self._filter,
            MapJson: self._map,
            ReduceJson: self._reduce,
            JoinJson: self._join,
            UnionJson: self._union,
            TopJson: self._top,
            LagJson: self._lag,
            CollectJson: self._collect,
        }

    def _filter(self, step: FilterJson, path: str) -> Operation:
        return FilterOperation(self._parsing.expression(step.where, "where"))

    def _map(self, step: MapJson, path: str) -> Operation:
        return MapOperation({field: self._parsing.expression(text, field) for field, text in step.assignments.items()})

    def _reduce(self, step: ReduceJson, path: str) -> Operation:
        aggregates = tuple(parse_aggregate(key, text) for key, text in step.aggregate.items())
        return ReduceOperation(self._grouping(step), aggregates)

    def _grouping(self, step: ReduceJson) -> Grouping:
        if step.window is not None:
            return WindowGrouping(Window.parse(step.window))
        if step.by is None:
            return AllRows()
        if isinstance(step.by, str):
            return KeyGrouping((GroupKeyField(step.by),))
        if isinstance(step.by, list):
            return KeyGrouping(tuple(GroupKeyField(name) for name in step.by))
        keys = (GroupKeyField(name, self._parsing.expression(text, name)) for name, text in step.by.items())
        return KeyGrouping(tuple(keys))

    def _join(self, step: JoinJson, path: str) -> Operation:
        on = (step.on,) if isinstance(step.on, str) else tuple(step.on)
        return JoinOperation(on, self._parsing.nested_chain(step.chain, f"{path}with."))

    def _union(self, step: UnionJson, path: str) -> Operation:
        return UnionOperation(self._parsing.nested_chain(step.chain, f"{path}with."))

    def _top(self, step: TopJson, path: str) -> Operation:
        return TopOperation(step.n, step.by, step.order)

    def _lag(self, step: LagJson, path: str) -> Operation:
        by = (step.by,) if isinstance(step.by, str) else tuple(step.by)
        return LagOperation(step.field, tuple(step.lags), by)

    def _collect(self, step: CollectJson, path: str) -> Operation:
        fields = tuple(step.fields) if step.fields is not None else None
        return CollectOperation(fields, step.decimals)
