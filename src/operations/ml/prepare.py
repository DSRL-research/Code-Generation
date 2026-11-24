from dataclasses import dataclass
from typing import ClassVar

from src.models.rows import Rows
from src.operations.base import Operation, RunContext
from src.operations.ml.common import replay_recipe, store_of


@dataclass(frozen=True)
class PrepareOperation(Operation):
    """Makes a model's inputs on the rows, exactly as when it was trained: runs the steps of its recipe.

    JSON: {"op": "prepare", "model": "mg2_demand"}
    The recipe is the lag and map steps that came before train; prepare runs them on the rows, so predict can
    follow. Writing the same steps by hand gives the same rows; prepare cannot get them wrong, and works for a
    model loaded from another sandbox.
    """

    name: ClassVar[str] = "prepare"
    model: str

    def apply(self, rows: Rows, ctx: RunContext) -> Rows:
        return replay_recipe(store_of(ctx).load(self.model), rows, ctx)
