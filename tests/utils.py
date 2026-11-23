"""Helpers for the tests: test data, rows, and running a chain without HTTP."""

import json
from collections.abc import Callable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from src.clients.records import to_row
from src.models.rows import Row, Rows, measure_interval
from src.models.time import Timestamp, parse_time
from src.operations import RunContext
from src.services.chain_parser import ChainParser
from src.services.fetch_once import FetchOnce
from src.store import ModelStore

RESOURCES = Path(__file__).parent / "resources"
TIME_FIELDS = ("ts", "timestamp", "time", "date")


def load_json(relative: str) -> Any:
    return json.loads((RESOURCES / relative).read_text())


def load_chain(name: str, base_url: str) -> Any:
    """A chain from resources/chains/, with {base} replaced by the URL of the test source server."""
    return json.loads((RESOURCES / "chains" / f"{name}.json").read_text().replace("{base}", base_url))


def t(text: str) -> Timestamp:
    """A time, as a row holds it in ts."""
    time = parse_time(text)
    assert time is not None, text
    return time


def rows(*items: Row, interval_minutes: int | None = None) -> Rows:
    interval = timedelta(minutes=interval_minutes) if interval_minutes is not None else None
    return Rows(list(items), interval)


def source_rows(records: list[Any]) -> Rows:
    """Records turned into rows as the source client does it."""
    items = [to_row(record, TIME_FIELDS) for record in records]
    return Rows(items, measure_interval(items))


def run(
    chain_json: Any, sources: dict[str, Rows], fetches: list[str] | None = None, models: ModelStore | None = None
) -> list[Row]:
    """Parses and runs a chain on in-memory sources, without the policy. `fetches` records each fetched link;
    `models` is the model store for the ML operators."""

    def fetch(link: str) -> Rows:
        if fetches is not None:
            fetches.append(link)
        return sources[link]

    chain = ChainParser().parse(chain_json)
    return chain.execute(RunContext(fetch=FetchOnce(fetch), models=models)).items


def mg_rows(number: int, start: str | None = None, end: str | None = None) -> Rows:
    """The hourly readings of a microgrid (resources/microgrids/), from day `start` to day `end`, both included."""
    readings = load_json(f"microgrids/MG{number}/readings.json")["readings"]
    first, last = start or "0000", end or "9999"
    return source_rows([r for r in readings if first <= r["timestamp"][:10] <= last])


# ---------- synthetic 5-minute readings ----------

FIRST_READING_DAY = date(2026, 7, 29)


def reading_kw(time: datetime, day_index: int) -> float:
    """kw = 360 + slot + (-1, 0 or +1 within the slot) + (+10 on the first day, -10 on the second).

    So the 15-minute mean of a day is 360 + slot +-10, and the mean of a slot over both days is 360 + slot.
    """
    slot = (time.hour * 60 + time.minute) // 15
    within_slot = (time.minute % 15) // 5 - 1
    return 360.0 + slot + within_slot + (10 if day_index % 2 == 0 else -10)


def make_readings(days: int = 2, skip: Callable[[datetime], bool] = lambda time: False) -> dict[str, Any]:
    """{"readings": [{"timestamp": "2026-07-29T00:00:00", "kw": ...}, ...]}: one reading every 5 minutes."""
    readings = []
    for day_index in range(days):
        midnight = datetime.combine(FIRST_READING_DAY + timedelta(days=day_index), datetime.min.time())
        for step in range(24 * 12):
            time = midnight + timedelta(minutes=5 * step)
            if not skip(time):
                readings.append({"timestamp": time.isoformat(), "kw": reading_kw(time, day_index)})
    return {"readings": readings}
