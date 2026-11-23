from datetime import timedelta
from typing import Any

import pytest

from src.exceptions import ChainError
from src.models.rows import smaller_interval
from tests.utils import rows, run, t

LEFT = "http://test/left"
RIGHT = "http://test/right"
SOURCES = {
    LEFT: rows(
        {"k": 1, "a": 10, "same": "left"},
        {"k": 2, "a": 20, "same": "left"},
        {"k": 3, "a": 30, "same": "left"},
        interval_minutes=15,
    ),
    RIGHT: rows({"k": 1, "b": 100, "same": "right"}, {"k": 3, "b": 300, "same": "right"}, interval_minutes=5),
}


def chain(*steps: dict[str, Any]) -> dict[str, Any]:
    return {"source": LEFT, "steps": list(steps)}


# ---------- join ----------


def test_join_is_an_inner_join_and_keeps_the_incoming_value_on_a_clash() -> None:
    result = run(chain({"op": "join", "on": "k", "with": {"source": RIGHT}}), SOURCES)
    assert result == [{"k": 1, "b": 100, "same": "left", "a": 10}, {"k": 3, "b": 300, "same": "left", "a": 30}]


def test_join_on_a_computed_key_set_on_both_sides() -> None:
    forecast = rows({"ts": t("2020-02-25T18:00"), "load": 428.7}, {"ts": t("2020-02-25T18:15"), "load": 415.7})
    baseline = rows({"slot": 72, "baseline_kw": 432.0}, {"slot": 73, "baseline_kw": 433.0})
    result = run(
        {
            "source": "http://test/forecast",
            "steps": [
                {"op": "map", "set": {"slot": "ts.slot()"}},
                {"op": "join", "on": "slot", "with": {"source": "http://test/baseline"}},
                {"op": "map", "set": {"flex_kw": "max(0, load - baseline_kw)"}},
            ],
        },
        {"http://test/forecast": forecast, "http://test/baseline": baseline},
    )
    assert [(row["slot"], row["flex_kw"]) for row in result] == [(72, 0), (73, 0)]


def test_join_on_several_fields() -> None:
    sources = {LEFT: rows({"x": 1, "y": 1, "a": 1}, {"x": 1, "y": 2, "a": 2}), RIGHT: rows({"x": 1, "y": 2, "b": 3})}
    assert run(chain({"op": "join", "on": ["x", "y"], "with": {"source": RIGHT}}), sources) == [
        {"x": 1, "y": 2, "b": 3, "a": 2}
    ]


def test_join_needs_one_row_per_key_on_the_other_side() -> None:
    sources = {**SOURCES, RIGHT: rows({"k": 1}, {"k": 1})}
    with pytest.raises(ChainError, match=r'steps\[0\] join: the rows of "with" have the key k=1 twice'):
        run(chain({"op": "join", "on": "k", "with": {"source": RIGHT}}), sources)


def test_join_needs_the_key_on_both_sides() -> None:
    with pytest.raises(ChainError, match="the incoming rows have no field b"):
        run(chain({"op": "join", "on": "b", "with": {"source": RIGHT}}), SOURCES)
    with pytest.raises(ChainError, match='the rows of "with" have no field a'):
        run(chain({"op": "join", "on": "a", "with": {"source": RIGHT}}), SOURCES)


def test_an_error_under_with_names_its_full_place() -> None:
    inner = {"source": RIGHT, "steps": [{"op": "map", "set": {"x": "nope + 1"}}]}
    with pytest.raises(ChainError, match=r"steps\[0\]\.with\.steps\[0\] map: unknown field nope"):
        run(chain({"op": "join", "on": "k", "with": inner}), SOURCES)


# ---------- union ----------


def test_union_appends_the_other_rows_and_keeps_the_smaller_interval() -> None:
    result = run(
        chain({"op": "union", "with": {"source": RIGHT}}, {"op": "reduce", "aggregate": {"n": "count()"}}), SOURCES
    )
    assert result == [{"n": 5}]


def test_union_keeps_duplicates_and_order() -> None:
    result = run(chain({"op": "union", "with": {"source": LEFT}}), SOURCES)
    assert [row["k"] for row in result] == [1, 2, 3, 1, 2, 3]


def test_union_interval() -> None:
    assert smaller_interval(timedelta(minutes=15), timedelta(minutes=5)) == timedelta(minutes=5)
    assert smaller_interval(None, timedelta(minutes=5)) == timedelta(minutes=5)
    assert smaller_interval(None, None) is None


# ---------- fetching ----------


def test_each_link_is_fetched_once_per_run() -> None:
    fetches: list[str] = []
    run(
        chain(
            {"op": "union", "with": {"source": LEFT}},
            {"op": "join", "on": "k", "with": {"source": RIGHT}},
            {"op": "union", "with": {"source": RIGHT, "steps": [{"op": "union", "with": {"source": LEFT}}]}},
        ),
        SOURCES,
        fetches,
    )
    assert fetches == [LEFT, RIGHT]
