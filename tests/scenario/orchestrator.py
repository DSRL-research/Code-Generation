"""UC1: one step of a workflow is a capability over a scope of microgrids.

For each microgrid, the capability is obtained:
1. from an asset of that microgrid that has it (a link, or a model);
2. else derived with the recipe of the knowledge base, after obtaining what the recipe needs;
3. else a recipe is generated, its chain is checked by the sandbox, run, and its output checked; only then it is
   registered, so the next microgrid that lacks the capability reuses it without asking the generator again.

A recipe that trains a model adds the model to the knowledge base as an asset of that microgrid, so later steps
use it instead of training again.
"""

from dataclasses import dataclass
from typing import Any

from tests.scenario.generator import CannedGenerator
from tests.scenario.knowledge import Asset, KnowledgeBase, Recipe
from tests.scenario.sandbox import Sandbox
from tests.scenario.templates import as_chain, fill


class RecipeFailed(Exception):
    pass


@dataclass(frozen=True)
class Provided:
    """A capability for one microgrid: a link, a model name, or a chain; and its rows if they were computed."""

    kind: str  # "agg", "pred", "model", or "rows" (derived)
    value: Any
    rows: list[dict[str, Any]] | None = None


class Orchestrator:
    def __init__(self, sandbox: Sandbox, knowledge: KnowledgeBase, generator: CannedGenerator) -> None:
        self.sandbox = sandbox
        self.knowledge = knowledge
        self.generator = generator

    def step(self, capability: str, mgs: list[int]) -> dict[int, list[dict[str, Any]]]:
        """The rows of a capability for each microgrid of the scope."""
        return {mg: self.rows(capability, mg) for mg in mgs}

    def rows(self, capability: str, mg: int) -> list[dict[str, Any]]:
        provided = self.provide(capability, mg)
        if provided.rows is not None:
            return provided.rows
        if provided.kind == "model":
            return [self.sandbox.model(provided.value)]
        return self.sandbox.run(as_chain(provided.value))

    def provide(self, capability: str, mg: int) -> Provided:
        asset = self.knowledge.asset(capability, mg)
        if asset is not None:
            return Provided(asset.kind, asset.ref)
        recipe = self.knowledge.recipe(capability)
        if recipe is not None:
            return self._derive(recipe, mg)
        recipe = self.generator.generate(capability, mg, self.knowledge.assets, self.sandbox.operators())
        provided = self._derive(recipe, mg)  # raises if the sandbox refuses the chain or its output is wrong
        self.knowledge.register(recipe)
        return provided

    def _derive(self, recipe: Recipe, mg: int) -> Provided:
        values = {"model_name": f"mg{mg}_{recipe.capability}"}
        for need in recipe.needs:
            values[need] = self.provide(need, mg).value
        chain = fill(recipe.chain, values)
        self.sandbox.check(chain)
        rows = self.sandbox.run(chain)
        missing = [field for field in recipe.output_fields if not rows or field not in rows[0]]
        if missing:
            raise RecipeFailed(f"{recipe.capability} for MG{mg}: the output has no {missing}")
        if recipe.produces == "model":
            name = rows[0]["name"]
            self.knowledge.add_asset(Asset(recipe.capability, mg, "model", name, derived_from=recipe.capability))
            return Provided("model", name)
        return Provided("rows", chain, rows)
