from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Literal

from src.exceptions import ChainError
from src.ml.vectors import cosine, distance
from src.models.rows import Rows
from src.operations.base import CombinesChains, RunContext

if TYPE_CHECKING:
    from src.models.chain import Chain

_METRICS = {"distance": distance, "cosine": cosine}


@dataclass(frozen=True)
class SimilarityOperation(CombinesChains):
    """Compares the vector of each row with the vector of the one row of the chain under "with".

    JSON: {"op": "similarity", "field": "embedding", "metric": "distance", "with": {...}, "as": "distance"}
    metric "distance": smaller is more similar; "cosine": closer to 1 is more similar. Use top to keep the most
    similar rows, for example {"op": "top", "n": 1, "by": "distance", "order": "asc"}.
    """

    name: ClassVar[str] = "similarity"
    field: str
    metric: Literal["distance", "cosine"]
    with_chain: Chain
    output: str

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        reference = self.with_chain.execute(ctx).items
        if len(reference) != 1 or self.field not in reference[0]:
            raise ChainError(f'the chain under "with" must give one row with {self.field}, not {len(reference)} rows')
        compare = _METRICS[self.metric]
        compared = []
        for row in rows.items:
            if self.field not in row:
                raise ChainError(f"unknown field {self.field}")
            try:
                compared.append({**row, self.output: compare(row[self.field], reference[0][self.field])})
            except (TypeError, ValueError) as error:
                raise ChainError(f"{self.metric}: {error}") from None
        return rows.with_items(compared)
