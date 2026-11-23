"""Checking a parsed chain before it runs. Reads the chain only; fetches nothing (PolicyChecker.java).

PolicyChecker (SANDBOX_POLICY_ENABLED=true):
- every source must match a source rule; the rule says whether its rows are raw;
- rows are raw until the first reduce with a window; while raw, each step must follow its operator's rule;
- an operator without a rule in the policy is refused;
- the chain, and every chain under "with", must not end with raw rows.

NoPolicy (the default for now): allows every chain, and only lists the links it reads.
"""

from dataclasses import dataclass
from typing import Protocol

from src.exceptions import PolicyViolation
from src.models.chain import Chain
from src.operations import CombinesChains
from src.policy.policy import Policy


@dataclass(frozen=True)
class SourceUse:
    link: str
    raw: bool | None  # None when the policy is switched off: nothing says whether the link is raw


@dataclass(frozen=True)
class CheckReport:
    """What a chain reads, for POST /chains/check."""

    sources: tuple[SourceUse, ...]

    def to_json(self) -> dict[str, list[dict[str, object]]]:
        return {"sources": [{"link": use.link, "raw": use.raw} for use in self.sources]}


class ChainChecker(Protocol):
    def check(self, chain: Chain) -> CheckReport: ...


class NoPolicy:
    """The policy check switched off: every chain is allowed. The policy code stays, to switch it back on."""

    def check(self, chain: Chain) -> CheckReport:
        uses: dict[str, SourceUse] = {}
        self._collect(chain, uses)
        return CheckReport(tuple(uses.values()))

    def _collect(self, chain: Chain, uses: dict[str, SourceUse]) -> None:
        uses.setdefault(chain.source, SourceUse(chain.source, None))
        for step in chain.steps:
            if isinstance(step.operation, CombinesChains):
                self._collect(step.operation.with_chain, uses)


class PolicyChecker:
    def __init__(self, policy: Policy) -> None:
        self._policy = policy

    def check(self, chain: Chain) -> CheckReport:
        """Raises PolicyViolation, naming the step, if the chain breaks the policy."""
        uses: dict[str, SourceUse] = {}
        if self._ends_raw(chain, uses):
            raise PolicyViolation("the chain would give raw rows; reduce them with a window first")
        return CheckReport(tuple(uses.values()))

    def _ends_raw(self, chain: Chain, uses: dict[str, SourceUse]) -> bool:
        """Checks the steps in order and returns whether the chain's output is still raw rows."""
        try:
            source = self._policy.source_rule(chain.source)
        except PolicyViolation as error:
            error.at(f"{chain.path}source")
            raise
        uses.setdefault(chain.source, SourceUse(chain.source, source.raw))
        raw = source.raw
        for step in chain.steps:
            operation = step.operation
            rule = self._policy.rules.get(operation.name)
            if rule is None:
                raise PolicyViolation("this operator is not allowed by the policy", step.where)
            if isinstance(operation, CombinesChains) and self._ends_raw(operation.with_chain, uses):
                raise PolicyViolation(
                    f'the chain under "{operation.chain_key}" still gives raw rows; reduce them first', step.where
                )
            try:
                raw = rule.check(operation, raw)
            except PolicyViolation as error:
                error.at(step.where)
                raise
        return raw
