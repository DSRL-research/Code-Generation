from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from src.exceptions import ChainError
from src.models.rows import Row, Rows
from src.operations.base import CombinesChains, RunContext

if TYPE_CHECKING:
    from src.models.chain import Chain

Key = tuple[object, ...]


@dataclass(frozen=True)
class JoinOperation(CombinesChains):
    """Inner join with the rows of the chain under "with", on one or more fields (JoinOperation.java).

    JSON: {"op": "join", "on": "slot", "with": {...}} or {"op": "join", "on": ["weekday", "hour"], "with": {...}}
    - the rows of "with" must have one row per key;
    - keys are compared as they are;
    - when both sides have a field, the value of the incoming rows is kept.
    """

    name: ClassVar[str] = "join"
    on: tuple[str, ...]
    with_chain: Chain

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        others = self._index(self.with_chain.execute(ctx).items)
        joined = []
        for row in rows.items:
            match = others.get(self._key(row, "incoming rows"))
            if match is not None:
                joined.append({**match, **row})
        return rows.with_items(joined)

    def _index(self, items: list[Row]) -> dict[Key, Row]:
        index: dict[Key, Row] = {}
        for row in items:
            key = self._key(row, 'rows of "with"')
            if key in index:
                raise ChainError(
                    f'the rows of "with" have the key {self._describe(key)} twice; a join needs one row '
                    "per key, so reduce them first"
                )
            index[key] = row
        return index

    def _key(self, row: Row, side: str) -> Key:
        for field in self.on:
            if field not in row:
                raise ChainError(f"the {side} have no field {field}")
        key = tuple(row[field] for field in self.on)
        try:
            hash(key)
        except TypeError:
            raise ChainError(f"a join key cannot be a list or an object: {self._describe(key)}") from None
        return key

    def _describe(self, key: Key) -> str:
        return ", ".join(f"{field}={value!r}" for field, value in zip(self.on, key, strict=True))
