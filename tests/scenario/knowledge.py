"""The knowledge base: which assets exist, and how a capability can be derived.

An asset is something a microgrid offers for a capability: data (a link to its readings, "agg"), predictions (a
link, "pred") or a model (a model name in the sandbox, "model"). A recipe says how to derive a capability from
other capabilities with operators. A capability is obtained from an asset that has it, or derived with a recipe.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Asset:
    capability: str
    mg: int
    kind: str  # "agg", "pred" or "model"
    ref: str  # a link, or a model name
    derived_from: str | None = None  # how a derived asset was obtained, for provenance


@dataclass(frozen=True)
class Recipe:
    capability: str
    description: str
    needs: tuple[str, ...]  # the capabilities whose assets fill the chain
    produces: str  # "rows" (the chain gives the capability's rows) or "model" (the chain trains a model)
    output_fields: tuple[str, ...]  # what the capability promises; checked before the recipe is registered
    chain: dict[str, Any]  # a chain with ${need} placeholders, and ${model_name} for a model

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Recipe":
        return cls(data["capability"], data["description"], tuple(data["needs"]), data["produces"],
                   tuple(data["output_fields"]), data["chain"])  # fmt: skip


@dataclass
class KnowledgeBase:
    assets: list[Asset] = field(default_factory=list)
    recipes: dict[str, Recipe] = field(default_factory=dict)

    def asset(self, capability: str, mg: int) -> Asset | None:
        return next((a for a in self.assets if a.capability == capability and a.mg == mg), None)

    def assets_of(self, capability: str) -> list[Asset]:
        return [asset for asset in self.assets if asset.capability == capability]

    def add_asset(self, asset: Asset) -> None:
        self.assets = [a for a in self.assets if (a.capability, a.mg) != (asset.capability, asset.mg)] + [asset]

    def recipe(self, capability: str) -> Recipe | None:
        return self.recipes.get(capability)

    def register(self, recipe: Recipe) -> None:
        self.recipes[recipe.capability] = recipe
