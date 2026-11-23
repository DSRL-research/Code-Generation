"""Builders: JSON of a step -> its operation, one module per family of operators (stream.py, ml.py).

A builder gets the validated JSON of a step and its place (e.g. "steps[1]."), checks what the values mean, and
makes the operation. It raises ChainError for a value that cannot be used; the parser adds the place.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Protocol

from src.operations import Operation

if TYPE_CHECKING:
    from src.expressions import Expression
    from src.models.chain import Chain
    from src.models.chain_json import ChainJson

Builder = Callable[[Any, str], Operation]


class Parsing(Protocol):
    """What the builders need from the parser."""

    def expression(self, text: str, about: str | None = None) -> "Expression":
        """Parses an expression; `about` names it in an error, e.g. the field of a map."""
        ...

    def nested_chain(self, model: "ChainJson", path: str) -> "Chain":
        """Builds the chain under "with"; `path` is its place, e.g. "steps[1].with."."""
        ...
