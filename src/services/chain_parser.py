"""Turns a chain written as JSON into a Chain of operations. Reads no data (ChainParser.java).

1. pydantic checks the shape of the JSON (models/chain_json.py, models/steps/);
2. a builder per operator makes its operation (services/builders/stream.py and ml.py);
3. an operation that saves a model with its recipe (train) gets the steps before it (services/recipe.py).

To add an operator: add its model to models/steps/ and a builder to the builders of its family.
"""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from pydantic import BaseModel, ValidationError

from src.exceptions import ChainError
from src.expressions import Expression
from src.models.chain import Chain, Step
from src.models.chain_json import ChainJson
from src.models.steps.stream import CollectJson
from src.operations import RecordsRecipe
from src.services import validation_errors
from src.services.builders import Builder
from src.services.builders.ml import MlBuilders
from src.services.builders.stream import StreamBuilders
from src.services.recipe import RECIPE_SOURCE, check_recipe, recipe_before


class ChainParser:
    def __init__(self, max_expression_length: int = 500) -> None:
        self._max_expression_length = max_expression_length
        self._builders: dict[type[BaseModel], Builder] = {
            **StreamBuilders(self).builders(),
            **MlBuilders(self).builders(),
        }

    def parse(self, chain_json: Any) -> Chain:
        """Raises ChainError, naming the place, if the chain cannot be read."""
        try:
            model = ChainJson.model_validate(chain_json)
        except ValidationError as error:
            raise ChainError(validation_errors.describe(error)) from None
        return self.nested_chain(model, path="")

    def nested_chain(self, model: ChainJson, path: str) -> Chain:
        """A chain and its steps; `path` is "" for the chain that was sent, e.g. "steps[1].with." for one under it."""
        steps = []
        for index, step in enumerate(model.steps):
            where = f"{path}steps[{index}] {step.op}"
            try:
                if isinstance(step, CollectJson) and index != len(model.steps) - 1:
                    raise ChainError("collect can only be the last step")
                operation = self._builders[type(step)](step, f"{path}steps[{index}].")
                if isinstance(operation, RecordsRecipe):
                    operation = replace(operation, recipe=recipe_before(model.steps[:index]))
            except ChainError as error:
                error.at(where)
                raise
            steps.append(Step(where, operation))
        return Chain(model.source, tuple(steps), path)

    def recipe_steps(self, recipe: Sequence[Any]) -> tuple[Step, ...]:
        """The steps of a model's recipe (JSON), ready to run on rows: for prepare and forecast. Raises ChainError
        if it holds anything but lag and map steps, or cannot be read."""
        check_recipe(recipe)
        return self.parse({"source": RECIPE_SOURCE, "steps": list(recipe)}).steps

    def expression(self, text: str, about: str | None = None) -> Expression:
        try:
            return Expression.parse(text, self._max_expression_length)
        except ChainError as error:
            if about is not None:
                error.message = f"{about}: {error.message}"
            raise
