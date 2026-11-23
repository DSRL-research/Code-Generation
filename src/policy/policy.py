"""Policy: policy.json, loaded and checked at startup."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.exceptions import PolicyFileError, PolicyViolation
from src.operations import OPERATION_NAMES
from src.policy.policy_file import PolicyFile
from src.policy.rules import OperatorRule, rule_from_json
from src.policy.sources import Link, SourceRule


@dataclass(frozen=True)
class Policy:
    sources: tuple[SourceRule, ...]
    time_fields: frozenset[str]
    rules: dict[str, OperatorRule]  # operator name -> rule; an operator without a rule is refused

    @classmethod
    def load(cls, path: Path) -> "Policy":
        """Raises PolicyFileError, naming the problem, if the file cannot be read."""
        try:
            data = json.loads(Path(path).read_text())
        except (OSError, ValueError) as error:
            raise PolicyFileError(f"{path}: cannot read the policy: {error}") from None
        try:
            return cls.from_json(data)
        except PolicyFileError as error:
            raise PolicyFileError(f"{path}: {error}") from None

    @classmethod
    def from_json(cls, data: Any) -> "Policy":
        try:
            file = PolicyFile.model_validate(data)
        except ValidationError as error:
            raise PolicyFileError(str(error)) from None
        unknown = sorted(set(file.operators) - set(OPERATION_NAMES))
        if unknown:
            raise PolicyFileError(f"unknown operators {unknown}; the operators are {list(OPERATION_NAMES)}")
        time_fields = frozenset(file.time_fields)
        return cls(
            sources=tuple(SourceRule.parse(entry.match, entry.raw) for entry in file.sources),
            time_fields=time_fields,
            rules={name: rule_from_json(name, entry.on_raw, time_fields) for name, entry in file.operators.items()},
        )

    def source_rule(self, link: str) -> SourceRule:
        """The first source rule that matches the link. Raises PolicyViolation if none does."""
        parsed = Link.parse(link)
        for rule in self.sources:
            if rule.matches(parsed):
                return rule
        raise PolicyViolation(f"{link} is not an allowed source")
