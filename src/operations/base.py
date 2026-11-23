"""The contract of an operation (Operation.java, CombinesChains.java)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from src.models.chain import Chain
    from src.models.rows import Rows
    from src.store import ModelStore


@dataclass(frozen=True)
class RunContext:
    """What one run gives to every operation.

    fetch(link) returns the rows of a link; ChainService makes it fetch each link once per run.
    models is the model store of the sandbox, used by the ML operators.
    """

    fetch: Callable[[str], Rows]
    models: ModelStore | None = None


class Operation(ABC):
    """One step of a chain: takes the rows of the step before, and returns new rows.

    An operation never changes the rows or the row dicts it receives. `name` is the "op" of the JSON and the key
    of its rule in policy.json.
    """

    name: ClassVar[str]

    @abstractmethod
    def apply(self, rows: Rows, ctx: RunContext) -> Rows: ...


class CombinesChains(Operation, ABC):
    """An operation that also runs another chain: the one under "with" (join, union, similarity) or under "from"
    (load). `chain_key` is that key, as errors name it.

    The policy checker and /chains/check follow `with_chain` into that chain, so its source is checked and listed
    like any other: a chain fetches nothing but its sources.
    """

    chain_key: ClassVar[str] = "with"
    with_chain: Chain
