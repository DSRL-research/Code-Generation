from dataclasses import dataclass
from typing import Any, ClassVar

from src.exceptions import ChainError
from src.ml.profiles import daily_profile
from src.models.rows import Row, Rows
from src.operations.base import Operation, RunContext


@dataclass(frozen=True)
class EmbedOperation(Operation):
    """Turns the rows of each group into one row holding an embedding: the hour-of-day profile of a field.

    JSON: {"op": "embed", "field": "kw", "by": ["mg"], "normalize": false, "as": "embedding"}
    Without "by", all the rows give one embedding. An output row has the "by" fields, the embedding (24 numbers)
    and "days", the number of days it was computed from. Compare embeddings with the similarity operator, or
    with distance(a, b) and cosine(a, b) in expressions.
    """

    name: ClassVar[str] = "embed"
    field: str
    by: tuple[str, ...] = ()
    normalize: bool = False
    output: str = "embedding"

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        groups: dict[tuple[Any, ...], list[Row]] = {}
        for row in rows.items:
            groups.setdefault(self._key(row), []).append(row)
        embedded = []
        for key, group in groups.items():
            days = len({row["ts"].date() for row in group if "ts" in row})
            profile = daily_profile(group, self.field, self.normalize)
            embedded.append({**dict(zip(self.by, key, strict=True)), self.output: profile, "days": days})
        return Rows(embedded)

    def _key(self, row: Row) -> tuple[Any, ...]:
        for field in self.by:
            if field not in row:
                raise ChainError(f"unknown field {field}")
        return tuple(row[field] for field in self.by)
