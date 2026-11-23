"""Rows -> the JSON of the reply (DESIGN.md section 5, "Every reply")."""

import math
from datetime import date, time, timedelta
from typing import Any

from src.exceptions import ChainError
from src.models.rows import Row
from src.models.time import to_text


def to_reply(items: list[Row]) -> list[dict[str, Any]]:
    """Times become ISO text and durations seconds. A number that is not finite cannot be written in JSON."""
    return [{name: _json_value(value, name) for name, value in row.items()} for row in items]


def _json_value(value: Any, field: str) -> Any:
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ChainError(f"{field} is {value}, which JSON cannot hold", where="reply")
        return value
    if isinstance(value, date | time):
        return to_text(value)
    if isinstance(value, timedelta):
        return value.total_seconds()
    if isinstance(value, list | tuple | range):
        return [_json_value(item, field) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item, f"{field}.{key}") for key, item in value.items()}
    raise ChainError(f"{field} holds a {type(value).__name__}, which cannot be written as JSON", where="reply")
