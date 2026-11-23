from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from src.models.rows import Rows, smaller_interval
from src.operations.base import CombinesChains, RunContext

if TYPE_CHECKING:
    from src.models.chain import Chain


@dataclass(frozen=True)
class UnionOperation(CombinesChains):
    """Appends the rows of the chain under "with" after the incoming rows; duplicates are kept (UnionOperation.java).

    JSON: {"op": "union", "with": {"source": "...", "steps": [...]}}
    """

    name: ClassVar[str] = "union"
    with_chain: Chain

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        other = self.with_chain.execute(ctx)
        return Rows(rows.items + other.items, smaller_interval(rows.interval, other.interval))
