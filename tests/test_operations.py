"""filter, map, top, collect. reduce, join and union have their own files."""

import pytest

from src.exceptions import ChainError
from src.expressions import Expression
from src.operations import CollectOperation, FilterOperation, MapOperation, RunContext, TopOperation
from tests.utils import rows, t

CTX = RunContext(fetch=lambda link: rows())
DATA = rows(
    {"ts": t("2026-07-25T17:00"), "kw": 300.0, "name": "a"},
    {"ts": t("2026-07-25T18:00"), "kw": 500.0, "name": "b"},
    {"ts": t("2026-07-25T19:00"), "kw": 400.0, "name": "c"},
    {"ts": t("2026-07-25T21:00"), "kw": 500.0, "name": "d"},
    interval_minutes=60,
)


def names(result) -> list[str]:
    return [row["name"] for row in result.items]


# ---------- filter ----------


def test_filter_keeps_the_rows_where_the_expression_is_true() -> None:
    result = FilterOperation(Expression.parse("17 <= ts.hour < 21 and kw > 350")).apply(DATA, CTX)
    assert names(result) == ["b", "c"]
    assert result.interval == DATA.interval


def test_filter_needs_true_or_false() -> None:
    with pytest.raises(ChainError, match="where must give True or False, not 300.0"):
        FilterOperation(Expression.parse("kw")).apply(DATA, CTX)


# ---------- map ----------


def test_map_sets_fields_in_order_and_reads_earlier_ones() -> None:
    assignments = {
        "double": Expression.parse("kw * 2"),
        "high": Expression.parse("double > 900"),
        "label": Expression.parse("'high' if high else 'low'"),
    }
    result = MapOperation(assignments).apply(DATA, CTX)
    assert [(row["double"], row["high"], row["label"]) for row in result.items[:2]] == [
        (600.0, False, "low"),
        (1000.0, True, "high"),
    ]


def test_map_does_not_change_the_rows_it_receives() -> None:
    MapOperation({"kw": Expression.parse("0")}).apply(DATA, CTX)
    assert DATA.items[0]["kw"] == 300.0


# ---------- top ----------


def test_top_keeps_the_highest_and_ties_keep_their_order() -> None:
    assert names(TopOperation(3, "kw").apply(DATA, CTX)) == ["b", "d", "c"]


def test_top_asc_keeps_the_lowest() -> None:
    assert names(TopOperation(2, "kw", "asc").apply(DATA, CTX)) == ["a", "c"]


def test_top_puts_none_last() -> None:
    data = rows({"name": "x", "kw": None}, {"name": "y", "kw": 1.0})
    assert names(TopOperation(2, "kw").apply(data, CTX)) == ["y", "x"]


def test_top_needs_the_field() -> None:
    with pytest.raises(ChainError, match="unknown field load"):
        TopOperation(1, "load").apply(DATA, CTX)


# ---------- collect ----------


def test_collect_keeps_the_listed_fields_in_order_and_rounds() -> None:
    data = rows({"ts": t("2026-07-25T17:00"), "kw": 1.23456, "flag": True, "n": 3, "name": "a"})
    result = CollectOperation(("name", "kw", "flag", "n"), decimals=2).apply(data, CTX)
    assert result.items == [{"name": "a", "kw": 1.23, "flag": True, "n": 3}]
    assert list(result.items[0]) == ["name", "kw", "flag", "n"]


def test_collect_without_fields_keeps_every_field() -> None:
    data = rows({"a": 1.23456, "b": [0.111, 0.222]})
    assert CollectOperation(None, decimals=1).apply(data, CTX).items == [{"a": 1.2, "b": [0.1, 0.2]}]


def test_collect_needs_every_listed_field() -> None:
    with pytest.raises(ChainError, match="unknown field load"):
        CollectOperation(("load",)).apply(DATA, CTX)
