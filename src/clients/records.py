"""One record of a source -> one row (DESIGN.md section 4)."""

from collections.abc import Sequence
from typing import Any

from src.exceptions import ChainError
from src.models.rows import Row
from src.models.time import parse_time


def to_row(record: Any, time_fields: Sequence[str]) -> Row:
    """Every field of the record is kept as it came. The first of `time_fields` that holds ISO text is read into
    `ts`: if that field is `ts` itself, the time replaces the text; otherwise the text field stays as well."""
    if not isinstance(record, dict):
        raise ChainError(f"a record must be a JSON object, not {type(record).__name__}: {record!r:.80}")
    row = dict(record)
    for name in time_fields:
        value = record.get(name)
        if isinstance(value, str) and (time := parse_time(value)) is not None:
            row["ts"] = time
            break
    return row
