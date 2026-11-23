"""Turns pydantic's errors about a chain into one short line that names the place, for example
`steps[2].fields: unknown key`. The caller is often an LLM, so the text says what to fix."""

from typing import Any

from pydantic import ValidationError

from src.operations import OPERATION_NAMES

MAX_ERRORS = 3


def describe(error: ValidationError) -> str:
    """The first MAX_ERRORS errors, each as "<place>: <problem>", joined by "; "."""
    messages = []
    for item in error.errors()[:MAX_ERRORS]:
        place = _place(item["loc"])
        problem = _problem(item)
        messages.append(f"{place}: {problem}" if place else problem)
    return "; ".join(messages)


def _place(loc: tuple[Any, ...]) -> str:
    """("steps", 1, "join", "with", "steps", 0, "reduce", "window") -> "steps[1].with.steps[0].window"."""
    parts: list[str] = []
    for index, part in enumerate(loc):
        if isinstance(part, int) and parts:
            parts[-1] += f"[{part}]"
        elif isinstance(part, int):
            parts.append(f"[{part}]")
        elif part in OPERATION_NAMES and index > 0 and isinstance(loc[index - 1], int):
            continue  # the "op" of the step, which pydantic adds to the place
        else:
            parts.append(str(part))
    return ".".join(parts)


def _problem(item: dict[str, Any]) -> str:
    kind = item["type"]
    if kind == "extra_forbidden":
        return "unknown key"
    if kind == "missing":
        return "missing"
    if kind == "union_tag_invalid":
        return f"unknown operator {item['ctx']['tag']!r}; the operators are {', '.join(OPERATION_NAMES)}"
    if kind == "union_tag_not_found":
        return 'a step needs "op"'
    if kind in ("model_type", "model_attributes_type", "dict_type") and not item["loc"]:
        return "the chain must be a JSON object"
    return str(item["msg"]).removeprefix("Value error, ")
