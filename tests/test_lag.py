"""The lag operator: earlier values of a field as new fields (DESIGN.md sections 5 and 17)."""

from typing import Any

import pytest

from src.exceptions import ChainError
from src.models.rows import Rows
from src.services.chain_parser import ChainParser
from tests.utils import mg_rows, rows, run, t

LINK = "http://test/mg"


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(steps)}


def hourly(*values: float, start: int = 0, **fields: Any) -> list[dict[str, Any]]:
    """One row per hour of 2026-07-01 from `start`, with kw = each value."""
    return [{"ts": t(f"2026-07-01T{start + hour:02d}:00"), "kw": value, **fields} for hour, value in enumerate(values)]


def lag(source: Rows, **step: Any) -> list[dict[str, Any]]:
    return run(chain({"op": "lag", "field": "kw", **step}), {LINK: source})


def test_adds_the_earlier_values_and_drops_the_rows_without_them() -> None:
    result = lag(rows(*hourly(10.0, 11.0, 12.0, 13.0), interval_minutes=60), lags=[1, 2])
    assert result == [
        {"ts": t("2026-07-01T02:00"), "kw": 12.0, "kw_lag_1": 11.0, "kw_lag_2": 10.0},
        {"ts": t("2026-07-01T03:00"), "kw": 13.0, "kw_lag_1": 12.0, "kw_lag_2": 11.0},
    ]


def test_a_lag_counts_time_steps_of_the_rows() -> None:
    half_hours = rows(*[{"ts": t(f"2026-07-01T{minute // 60:02d}:{minute % 60:02d}"), "kw": float(minute)}
                        for minute in range(0, 180, 30)], interval_minutes=30)  # fmt: skip
    [first, *_] = lag(half_hours, lags=[2])
    assert first == {"ts": t("2026-07-01T01:00"), "kw": 60.0, "kw_lag_2": 0.0}  # 2 steps of 30 minutes: one hour


def test_a_gap_drops_the_rows_whose_earlier_value_is_missing() -> None:
    """The value 1 step earlier is the value at ts - 1 hour, never the row before when hours are missing."""
    with_a_gap = rows(*hourly(10.0, 11.0), *hourly(14.0, 15.0, start=4), interval_minutes=60)
    result = lag(with_a_gap, lags=[1])
    assert [(row["ts"], row["kw_lag_1"]) for row in result] == [
        (t("2026-07-01T01:00"), 10.0),
        (t("2026-07-01T05:00"), 14.0),
    ]


def test_a_missing_value_counts_as_missing() -> None:
    result = lag(rows(*hourly(10.0, None, 12.0, 13.0), interval_minutes=60), lags=[1])  # type: ignore[arg-type]
    assert [row["ts"] for row in result] == [t("2026-07-01T01:00"), t("2026-07-01T03:00")]
    assert result[0]["kw_lag_1"] == 10.0  # the row's own value may be missing; only the earlier ones must exist


def test_by_keeps_series_apart() -> None:
    """Two microgrids put together: each row gets lags from its own microgrid only."""
    both = rows(*hourly(10.0, 11.0, 12.0, mg=1), *hourly(500.0, 510.0, 520.0, mg=2), interval_minutes=60)
    result = lag(both, lags=[1], by="mg")
    assert [(row["mg"], row["kw"], row["kw_lag_1"]) for row in result] == [
        (1, 11.0, 10.0), (1, 12.0, 11.0), (2, 510.0, 500.0), (2, 520.0, 510.0)]  # fmt: skip


def test_rows_of_two_series_at_the_same_time_need_by() -> None:
    both = rows(*hourly(10.0, mg=1), *hourly(500.0, mg=2), interval_minutes=60)
    with pytest.raises(ChainError, match='two rows have the time 2026-07-01T00:00:00; .* with "by"') as error:
        lag(both, lags=[1])
    assert error.value.where == "steps[0] lag"


def test_the_interval_stays_so_a_reduce_or_model_can_follow() -> None:
    result = run(chain({"op": "lag", "field": "kw", "lags": [1]},
                       {"op": "reduce", "window": "4h", "aggregate": {"kw": "count"}}),
                 {LINK: rows(*hourly(*range(8)), interval_minutes=60)})  # fmt: skip
    assert result == [{"ts": t("2026-07-01T04:00"), "kw": 4}]  # 01:00-03:00 is not a complete 4h window


def test_on_real_readings_then_features_written_by_hand() -> None:
    """The first steps of a v3 training chain (DESIGN.md section 17), on MG1's hourly readings."""
    result = run(chain(
        {"op": "lag", "field": "kw", "lags": [1, 24]},
        {"op": "map", "set": {"hour_sin": "sin(2 * pi * ts.hour / 24)", "weekend": "1 if ts.weekday() >= 5 else 0",
                              "set": "'test' if ts >= '2026-07-18' else 'train'"}},
    ), {LINK: mg_rows(1, "2026-07-16", "2026-07-18")})  # fmt: skip
    assert len(result) == 48  # the first day only gives the lags
    first = result[0]
    assert first["ts"] == t("2026-07-17T00:00") and first["set"] == "train" and first["weekend"] == 0  # a Friday
    assert first["hour_sin"] == 0.0
    by_time = {row["ts"]: row["kw"] for row in mg_rows(1, "2026-07-16", "2026-07-18").items}
    assert first["kw_lag_24"] == by_time[t("2026-07-16T00:00")]
    assert first["kw_lag_1"] == by_time[t("2026-07-16T23:00")]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (Rows(hourly(10.0, 11.0), None), "the time step of the rows is unknown; reduce them with a window first"),
        (rows({"kw": 1.0}, interval_minutes=60), r"lag needs a time \(ts\) in every row"),
    ],
)
def test_lag_needs_a_time_step_and_times(source: Rows, message: str) -> None:
    with pytest.raises(ChainError, match=message):
        lag(source, lags=[1])


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"field": "load", "lags": [1]}, "the rows have no field load"),
        ({"field": "kw", "lags": [1], "by": "mg"}, "unknown field mg"),
    ],
)
def test_lag_names_a_missing_field(step: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError, match=message):
        run(chain({"op": "lag", **step}), {LINK: rows(*hourly(10.0, 11.0), interval_minutes=60)})


@pytest.mark.parametrize(
    ("step", "message"),
    [
        ({"field": "kw", "lags": [0, 24]}, "lags must be different whole numbers from 1 to 168"),
        ({"field": "kw", "lags": [1, 1]}, "lags must be different whole numbers"),
        ({"field": "kw", "lags": []}, r"steps\[0\]\.lags"),
        ({"field": "kw", "lags": [1], "by": 3}, "by must be a field name or a list of field names"),
        ({"lags": [1]}, r"steps\[0\]\.field: missing"),
    ],
)
def test_lag_is_checked_when_the_chain_is_parsed(step: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError, match=message):
        ChainParser().parse(chain({"op": "lag", **step}))
