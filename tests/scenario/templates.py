"""Filling a chain template with the assets of one microgrid.

- "${name}" as a whole value becomes the value itself (a link, a model name, or a chain);
- "${name}" inside text is replaced by the value as text, e.g. "ts >= '${validation_from}'";
- {"extend": "${capability}", "steps": [...]} is the chain of that capability with these steps added after its
  own: how one capability is built on another (UC1: consumer flexibility on the demand forecast).
"""

import re
from typing import Any

_PLACEHOLDER = re.compile(r"\$\{(\w+)\}")


def fill(template: Any, values: dict[str, Any]) -> Any:
    if isinstance(template, dict) and "extend" in template:
        base = as_chain(fill(template["extend"], values))
        return {"source": base["source"], "steps": [*base.get("steps", []), *fill(template["steps"], values)]}
    if isinstance(template, dict):
        return {key: fill(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [fill(value, values) for value in template]
    if isinstance(template, str):
        whole = _PLACEHOLDER.fullmatch(template)
        if whole:
            return values[whole[1]]
        return _PLACEHOLDER.sub(lambda match: str(values[match[1]]), template)
    return template


def as_chain(value: Any) -> dict[str, Any]:
    """A link is the chain that reads it; a chain stays as it is."""
    return {"source": value, "steps": []} if isinstance(value, str) else value
