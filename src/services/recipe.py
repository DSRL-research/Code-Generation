"""The recipe of a model: the steps of the chain that made its inputs (DESIGN.md section 17).

train saves its model with the lag and map steps just before it, as written in the chain, so that prepare and
forecast can make the same inputs again on other rows. Walking back from train:

- lag and map steps are taken: they make inputs;
- filter steps are passed over: they choose rows, they make no inputs;
- any other step (reduce, join, union, ...) ends the recipe: what comes before it is how the rows were obtained,
  not how the inputs were made from them.
"""

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from src.exceptions import ChainError

TAKEN = frozenset({"lag", "map"})
PASSED_OVER = frozenset({"filter"})

RECIPE_SOURCE = "http://recipe.invalid/"
"""The source a recipe is parsed with: a recipe has none, and is never run as a chain, only step by step."""


def check_recipe(recipe: Sequence[Any]) -> None:
    """A recipe to run holds only lag and map steps: what train saves. A recipe can come from another sandbox
    (load), so this is checked before it is parsed."""
    if not isinstance(recipe, list | tuple):
        raise ChainError("a recipe must be a list of steps")
    for index, step in enumerate(recipe):
        op = step.get("op") if isinstance(step, dict) else None
        if op not in TAKEN:
            raise ChainError(f"recipe steps[{index}]: a recipe may only hold lag and map steps, not {op!r}")


def recipe_before(steps: Sequence[BaseModel]) -> tuple[dict[str, Any], ...]:
    """The recipe of a step whose earlier steps (in its own chain) are `steps`: JSON steps, in chain order."""
    recipe = []
    for step in reversed(steps):
        op = step.op  # type: ignore[attr-defined]
        if op in TAKEN:
            recipe.append(step.model_dump(by_alias=True, exclude_defaults=True))
        elif op not in PASSED_OVER:
            break
    return tuple(reversed(recipe))
