"""Where the list of records is in the JSON reply of a source (DESIGN.md section 4)."""

import json
from typing import Any

from src.exceptions import ChainError

READINGS = "readings"


def records_in(reply: Any) -> list[Any]:
    """The reply itself if it is a list; else its "readings"; else its only field that holds a list."""
    if isinstance(reply, list):
        return reply
    if isinstance(reply, dict):
        if READINGS in reply and isinstance(reply[READINGS], list):
            return reply[READINGS]
        lists = [value for value in reply.values() if isinstance(value, list)]
        if READINGS not in reply and len(lists) == 1:
            return lists[0]
    raise ChainError(
        f'the reply holds no list of records; a source must reply with a list, an object with "{READINGS}", '
        "or an object with exactly one list"
    )


def error_text(body: bytes, limit: int = 300) -> str:
    """The "error" of a refusal such as {"error": "..."}, or the start of the body if it has none."""
    text = body.decode("utf-8", errors="replace")
    try:
        reply = json.loads(text)
    except ValueError:
        return text[:limit]
    if isinstance(reply, dict) and isinstance(reply.get("error"), str):
        return reply["error"][:limit]
    return text[:limit]
