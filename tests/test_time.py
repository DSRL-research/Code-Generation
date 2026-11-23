from datetime import date, datetime, timedelta

import pytest

from src.models.time import Timestamp, parse_time, to_text
from tests.utils import t


@pytest.mark.parametrize(
    "text, expected",
    [
        ("2026-07-25T18:15:00", datetime(2026, 7, 25, 18, 15)),
        ("2026-07-25 18:15:00", datetime(2026, 7, 25, 18, 15)),
        ("2026-07-25T18:15", datetime(2026, 7, 25, 18, 15)),
        ("2026-07-25", datetime(2026, 7, 25)),
        ("2020-02-25T12:00:00+00:00", datetime(2020, 2, 25, 12)),  # the offset is ignored
        ("2020-02-25T12:00:00+02:00", datetime(2020, 2, 25, 12)),  # the clock time is kept as written
        ("2026-07-25T18:15:00Z", datetime(2026, 7, 25, 18, 15)),
    ],
)
def test_reads_iso_text_and_ignores_the_offset(text: str, expected: datetime) -> None:
    time = parse_time(text)
    assert isinstance(time, Timestamp)
    assert time == expected
    assert time.tzinfo is None


@pytest.mark.parametrize("text", ["18:15", "25/07/2026 18:15", "abc", ""])
def test_text_that_is_not_a_time_gives_none(text: str) -> None:
    assert parse_time(text) is None


def test_compares_with_iso_text_on_either_side() -> None:
    ts = t("2026-07-25T18:15")
    assert ts >= "2026-07-25"
    assert "2026-07-24" <= ts < "2026-07-26"
    assert ts == "2026-07-25T18:15:00+00:00"
    assert ts != "2026-07-25T18:16"
    assert ts > date(2026, 7, 25)  # a date is its midnight


def test_text_that_cannot_be_read_is_an_error() -> None:
    with pytest.raises(ValueError, match="cannot read the time"):
        _ = t("2026-07-25") < "tomorrow"


def test_a_number_is_not_a_time() -> None:
    with pytest.raises(TypeError, match="compared with a time or with text"):
        _ = t("2026-07-25") < 2026


def test_between_includes_both_ends() -> None:
    assert t("2020-02-25T13:00").between("2020-02-25T13:00", "2020-02-25T14:00")
    assert t("2020-02-25T14:00").between("2020-02-25T13:00", "2020-02-25T14:00")
    assert not t("2020-02-25T14:05").between("2020-02-25T13:00", "2020-02-25T14:00")


@pytest.mark.parametrize(
    "text, minutes, slot",
    [
        ("00:00", 15, 0),
        ("00:14", 15, 0),
        ("00:15", 15, 1),
        ("18:00", 15, 72),
        ("18:10", 15, 72),
        ("23:59", 15, 95),
        ("18:45", 60, 18),
        ("18:45", 30, 37),
    ],
)
def test_slot_of_the_day(text: str, minutes: int, slot: int) -> None:
    assert t(f"2026-07-25T{text}").slot(minutes) == slot


def test_arithmetic_and_methods_of_datetime_still_work() -> None:
    ts = t("2026-07-25T18:15")
    assert isinstance(ts + timedelta(minutes=5), Timestamp)
    assert ts.isoweekday() == 6
    assert ts - t("2026-07-25T18:00") == timedelta(minutes=15)


def test_can_be_a_key_of_a_dict() -> None:
    assert {t("2026-07-25T18:15"): 1}[t("2026-07-25T18:15:00")] == 1


def test_written_as_iso_text() -> None:
    assert to_text(t("2026-07-25T18:15")) == "2026-07-25T18:15:00"
    assert to_text(date(2026, 7, 25)) == "2026-07-25"
