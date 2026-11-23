from collections.abc import Iterator
from datetime import timedelta

import httpx
import pytest

from src.clients import SourceClient, SourceLimits
from src.clients.records import to_row
from src.exceptions import ChainError, SourceError
from src.models.time import Timestamp
from tests.conftest import SourceServer
from tests.utils import TIME_FIELDS, t


@pytest.fixture
def client() -> Iterator[SourceClient]:
    limits = SourceLimits(timeout_seconds=0.5, max_bytes=200_000, max_records=1000)
    with httpx.Client(follow_redirects=False) as http:
        yield SourceClient(http, limits, TIME_FIELDS)


def test_a_list_of_records(client: SourceClient, source_server: SourceServer) -> None:
    rows = client.fetch(source_server.url("/files/battery_flex"))
    first = rows.items[0]
    assert len(rows.items) == 8
    assert first["ts"] == t("2020-02-25T12:00:00")
    assert first["timestamp"] == "2020-02-25T12:00:00+00:00"  # the text stays as it came
    assert first["origin"] == "2020-02-25T12:00:00+00:00"  # every field is kept, text too
    assert first["step"] == 1
    assert rows.interval == timedelta(minutes=15)


def test_an_object_with_readings(client: SourceClient, source_server: SourceServer) -> None:
    rows = client.fetch(source_server.url("/readings"))
    assert len(rows.items) == 2 * 288
    assert rows.interval == timedelta(minutes=5)


def test_an_object_with_one_list_and_ts_as_text(client: SourceClient, source_server: SourceServer) -> None:
    rows = client.fetch(source_server.url("/files/one_list"))
    assert rows.items[0] == {"ts": t("2026-07-25T18:00"), "load": 400.5}
    assert isinstance(rows.items[0]["ts"], Timestamp)  # the time replaces the text of ts


def test_the_first_time_field_holding_iso_text_becomes_ts() -> None:
    assert to_row({"ts": 15, "date": "2026-07-25"}, TIME_FIELDS) == {"ts": t("2026-07-25"), "date": "2026-07-25"}
    assert to_row({"time": "18:15", "kw": 1}, TIME_FIELDS) == {"time": "18:15", "kw": 1}  # no date: not a time


@pytest.mark.parametrize(
    "path, error, message",
    [
        ("/status/404", ChainError, "refused by the test server"),
        ("/status/503", SourceError, "replied with status 503"),
        ("/slow", SourceError, "no reply after 0.5 s"),
        ("/redirect", ChainError, "redirects are not followed"),
        ("/not-json", ChainError, "the reply is not JSON"),
        ("/many", ChainError, "1001 records, more than the limit of 1000"),
        ("/big", ChainError, "larger than 200000 bytes"),
        ("/files/one_list_missing", ChainError, "no file one_list_missing.json"),
    ],
)
def test_errors(client: SourceClient, source_server: SourceServer, path: str, error: type, message: str) -> None:
    with pytest.raises(error, match=message):
        client.fetch(source_server.url(path))


def test_a_link_that_cannot_be_reached(client: SourceClient) -> None:
    with pytest.raises(SourceError, match="cannot be reached"):
        client.fetch("http://127.0.0.1:1/nothing")
