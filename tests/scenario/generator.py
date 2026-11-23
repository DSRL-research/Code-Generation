"""The LLM that writes a recipe when the knowledge base has no way to obtain a capability.

A real generator gets the capability, the microgrid, the assets it may use and the operator catalogue of the
sandbox (GET /operators), and writes a chain. This stand-in returns the chain an LLM is expected to write
(tests/resources/recipes/<capability>.json, or a reply given to it), but refuses a chain that uses an operator the
catalogue does not list: generated code may only use the operators the sandbox offers.
"""

from typing import Any

from tests.scenario.knowledge import Asset, Recipe
from tests.scenario.sandbox import operators_of
from tests.utils import load_json


class GenerationFailed(Exception):
    pass


class CannedGenerator:
    def __init__(self, replies: dict[str, dict[str, Any]] | None = None) -> None:
        self._replies = replies or {}
        self.requests: list[tuple[str, int]] = []  # (capability, microgrid) of each request

    def generate(self, capability: str, mg: int, assets: list[Asset], operators: set[str]) -> Recipe:
        self.requests.append((capability, mg))
        reply = self._replies.get(capability) or load_json(f"recipes/{capability}.json")
        unknown = sorted(set(operators_of(reply["chain"])) - operators)
        if unknown:
            raise GenerationFailed(f"{capability}: the chain uses operators the sandbox does not offer: {unknown}")
        return Recipe.from_json(reply)
