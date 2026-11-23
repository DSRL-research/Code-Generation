from datetime import timedelta
from typing import Any

import pytest

from src.exceptions import ChainError
from src.models.rows import Rows
from tests.utils import load_json, make_readings, rows, run, source_rows, t

LINK = "http://test/source"
READINGS = source_rows(make_readings()["readings"])  # 2026-07-29 and 30, every 5 minutes


def reduce(step: dict[str, Any], data: Rows = READINGS, *before: dict[str, Any]) -> list[dict[str, Any]]:
    return run({"source": LINK, "steps": [*before, {"op": "reduce", **step}]}, {LINK: data})


# ---------- window ----------


def test_window_means_with_ts_at_the_start_of_each_window() -> None:
    result = reduce({"window": "15m", "aggregate": {"kw": "mean(kw)"}})
    assert len(result) == 2 * 96
    assert result[0] == {"ts": t("2026-07-29T00:00"), "kw": 370.0}  # 360 + slot 0 + 10 on the first day
    assert result[73] == {"ts": t("2026-07-29T18:15"), "kw": 443.0}
    assert result[96] == {"ts": t("2026-07-30T00:00"), "kw": 350.0}


def test_windows_start_at_midnight_and_7d_windows_on_monday() -> None:
    assert [row["ts"] for row in reduce({"window": "1d", "aggregate": {"n": "count()"}})] == [
        t("2026-07-29"),
        t("2026-07-30"),
    ]
    week = rows(*({"ts": t("2026-07-27") + timedelta(days=day), "kw": 1} for day in range(7)), interval_minutes=1440)
    assert reduce({"window": "7d", "aggregate": {"kw": "sum"}}, week) == [{"ts": t("2026-07-27"), "kw": 7}]


def test_incomplete_windows_are_dropped() -> None:
    gap = source_rows(make_readings(skip=lambda moment: moment.isoformat() == "2026-07-29T00:05:00")["readings"])
    result = reduce({"window": "15m", "aggregate": {"kw": "mean(kw)"}}, gap)
    assert result[0]["ts"] == t("2026-07-29T00:15")


def test_with_an_unknown_interval_no_window_is_complete() -> None:
    single = source_rows([{"timestamp": "2026-07-29T00:00:00", "kw": 1.0}])
    assert reduce({"window": "15m", "aggregate": {"kw": "mean(kw)"}}, single) == []


def test_a_filter_that_thins_the_rows_cannot_make_single_readings_into_means() -> None:
    thinning = {"op": "filter", "where": "ts.minute % 15 == 0"}  # keeps one 5-minute reading in three
    assert reduce({"window": "15m", "aggregate": {"kw": "mean(kw)"}}, READINGS, thinning) == []


def test_after_a_window_the_interval_is_the_window() -> None:
    result = reduce(
        {"window": "1h", "aggregate": {"kw": "mean"}},
        READINGS,
        {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean"}},
    )
    assert len(result) == 48


def test_a_window_needs_ts() -> None:
    with pytest.raises(ChainError, match="needs rows with a time"):
        reduce({"window": "15m", "aggregate": {"kw": "mean"}}, rows({"kw": 1.0}, {"kw": 2.0}))


# ---------- by ----------


def test_by_an_expression_gives_the_baseline_of_each_slot() -> None:
    result = reduce(
        {"by": {"slot": "ts.slot()"}, "aggregate": {"baseline_kw": "mean(kw)"}},
        READINGS,
        {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}},
    )
    assert len(result) == 96
    assert result[72] == {"slot": 72, "baseline_kw": 432.0}  # 360 + 72: the days' +10 and -10 cancel out


def test_by_a_field_and_a_list_of_fields_sorted_by_key() -> None:
    data = rows(
        {"g": "b", "h": 1, "kw": 1.0},
        {"g": "a", "h": 2, "kw": 2.0},
        {"g": "a", "h": 1, "kw": 4.0},
        {"g": "a", "h": 1, "kw": 6.0},
    )
    assert reduce({"by": "g", "aggregate": {"kw": "sum"}}, data) == [{"g": "a", "kw": 12.0}, {"g": "b", "kw": 1.0}]
    assert reduce({"by": ["g", "h"], "aggregate": {"kw": "max"}}, data) == [
        {"g": "a", "h": 1, "kw": 6.0},
        {"g": "a", "h": 2, "kw": 2.0},
        {"g": "b", "h": 1, "kw": 1.0},
    ]


def test_keys_are_compared_as_they_are() -> None:
    data = rows({"k": 0.5, "kw": 1.0}, {"k": 0, "kw": 2.0})
    assert reduce({"by": "k", "aggregate": {"kw": "sum"}}, data) == [{"k": 0, "kw": 2.0}, {"k": 0.5, "kw": 1.0}]


# ---------- no key ----------


def test_no_key_gives_one_row() -> None:
    # the highest reading: 360 + slot 95 + 1 within the slot + 10 on the first day
    assert reduce({"aggregate": {"kw": "max(kw)", "readings": "count()"}}) == [{"kw": 466.0, "readings": 576}]


def test_no_key_on_no_rows_gives_no_rows() -> None:
    assert reduce({"aggregate": {"n": "count()"}}, rows()) == []


def test_example_a_sums_the_battery_fields_from_13_to_14() -> None:
    battery = source_rows(load_json("sources/battery_flex.json"))
    window = {"op": "filter", "where": "ts.between('2020-02-25T13:00:00+00:00', '2020-02-25T14:00:00+00:00')"}
    [result] = reduce({"aggregate": {"charge*": "sum", "discharge*": "sum"}}, battery, window)
    assert result == pytest.approx(
        {
            "charge_p10": 1179.8444,
            "charge": 1201.2534,
            "charge_p90": 1235.0528,
            "discharge_p10": -49.9416,
            "discharge": -16.7093,
            "discharge_p90": 132.3328,
        }
    )


# ---------- aggregate ----------


def test_star_takes_every_field_that_holds_numbers() -> None:
    data = rows(
        {"g": 1, "ts": t("2026-07-25"), "kw": 1.0, "n": 2, "name": "a", "flag": True, "gone": None},
        {"g": 1, "ts": t("2026-07-26"), "kw": 3.0, "n": 4, "name": "b", "flag": False, "gone": None},
    )
    assert reduce({"by": "g", "aggregate": {"*": "sum"}}, data) == [{"g": 1, "kw": 4.0, "n": 6}]


def test_nulls_are_skipped() -> None:
    data = rows({"kw": 2.0}, {"kw": None}, {"kw": 4.0})
    assert reduce({"aggregate": {"mean": "mean(kw)", "values": "count(kw)", "rows": "count()"}}, data) == [
        {"mean": 3.0, "values": 2, "rows": 3}
    ]


def test_a_group_with_no_values_gives_none() -> None:
    assert reduce({"aggregate": {"kw": "sum"}}, rows({"kw": None})) == [{"kw": None}]


def test_several_aggregates_of_one_field() -> None:
    data = rows({"kw": 1.0}, {"kw": 3.0})
    assert reduce({"aggregate": {"kw": "mean", "low": "min(kw)", "high": "max(kw)", "total": "sum(kw)"}}, data) == [
        {"kw": 2.0, "low": 1.0, "high": 3.0, "total": 4.0}
    ]


@pytest.mark.parametrize(
    "aggregate, message",
    [
        ({"y": "sum(name)"}, r"sum\(name\) cannot be computed"),
        ({"y": "sum(nope)"}, "unknown field nope"),
        ({"z*": "sum"}, r"'z\*' matches no field"),
    ],
)
def test_aggregate_errors(aggregate: dict[str, str], message: str) -> None:
    with pytest.raises(ChainError, match=message):
        reduce({"aggregate": aggregate}, rows({"kw": 1.0, "name": "a"}))


def test_an_output_cannot_be_given_twice() -> None:
    with pytest.raises(ChainError, match="gives the field kw twice"):
        reduce({"aggregate": {"kw": "sum", "k*": "sum"}}, rows({"kw": 1.0}))
