from typing import Any

import pytest

from src.exceptions import ChainError
from src.operations import CollectOperation, FilterOperation, JoinOperation, MapOperation, ReduceOperation
from src.services.chain_parser import ChainParser
from tests.utils import load_chain

LINK = "http://127.0.0.1:1/files/x"


def parse(chain: Any):
    return ChainParser().parse(chain)


def steps(*items: dict[str, Any]) -> dict[str, Any]:
    return {"source": LINK, "steps": list(items)}


def test_parses_example_b_with_the_places_of_its_steps() -> None:
    chain = parse(load_chain("example_b", "http://127.0.0.1:1"))
    assert [type(step.operation) for step in chain.steps] == [
        MapOperation,
        JoinOperation,
        MapOperation,
        CollectOperation,
    ]
    assert [step.where for step in chain.steps] == ["steps[0] map", "steps[1] join", "steps[2] map", "steps[3] collect"]
    join = chain.steps[1].operation
    assert isinstance(join, JoinOperation)
    assert join.on == ("slot",)
    assert [step.where for step in join.with_chain.steps] == [
        "steps[1].with.steps[0] reduce",
        "steps[1].with.steps[1] reduce",
    ]


def test_a_chain_without_steps_is_valid() -> None:
    assert parse({"source": LINK}).steps == ()


def test_builds_expressions_once() -> None:
    chain = parse(steps({"op": "filter", "where": "17 <= ts.hour < 21"}))
    operation = chain.steps[0].operation
    assert isinstance(operation, FilterOperation)
    assert operation.where.fields == {"ts"}


@pytest.mark.parametrize(
    "step, criteria",
    [
        ({"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}}, {"mean"}),
        ({"op": "reduce", "by": ["weekday", "hour"], "aggregate": {"*": "max", "n": "count()"}}, {"max", "count"}),
        ({"op": "reduce", "aggregate": {"charge*": "sum"}}, {"sum"}),
    ],
)
def test_parses_the_forms_of_reduce(step: dict[str, Any], criteria: set[str]) -> None:
    operation = parse(steps(step)).steps[0].operation
    assert isinstance(operation, ReduceOperation)
    assert operation.criteria == criteria


@pytest.mark.parametrize(
    "chain, message",
    [
        ([1], "the chain must be a JSON object"),
        ({"steps": []}, "source: missing"),
        ({"source": "MG_1"}, "source must be a link"),
        ({"source": LINK, "stpes": []}, "stpes: unknown key"),
        (steps({"op": "dump"}), "steps[0]: unknown operator 'dump'"),
        (steps({"where": "x"}), 'steps[0]: a step needs "op"'),
        (steps({"op": "collect", "feilds": ["x"]}), "steps[0].feilds: unknown key"),
        (steps({"op": "filter"}), "steps[0].where: missing"),
        (steps({"op": "top", "n": "3", "by": "kw"}), "steps[0].n: Input should be a valid integer"),
        (
            steps({"op": "reduce", "window": "1h", "by": "hour", "aggregate": {"kw": "mean"}}),
            "give window or by, not both",
        ),
        (steps({"op": "reduce", "by": 5, "aggregate": {"kw": "mean"}}), "by must be a field name"),
        (steps({"op": "join", "on": "slot", "with": {"steps": []}}), "steps[0].with.source: missing"),
        (
            steps({"op": "join", "on": "k", "with": {"source": LINK, "steps": [{"op": "map", "set": {}}]}}),
            "steps[0].with.steps[0].set",
        ),
    ],
)
def test_shape_errors_name_the_place(chain: Any, message: str) -> None:
    with pytest.raises(ChainError) as error:
        parse(chain)
    assert message in str(error.value)


@pytest.mark.parametrize(
    "step, message",
    [
        ({"op": "filter", "where": "kw >"}, "steps[0] filter: where: cannot read the expression"),
        ({"op": "map", "set": {"x": "open('f')"}}, "steps[0] map: x: \"open('f')\": open() is not allowed"),
        ({"op": "reduce", "window": "7m", "aggregate": {"kw": "mean"}}, "steps[0] reduce: a window must divide a day"),
        ({"op": "reduce", "window": "2d", "aggregate": {"kw": "mean"}}, "must be 1d or 7d"),
        ({"op": "reduce", "window": "1x", "aggregate": {"kw": "mean"}}, "cannot read the window '1x'"),
        ({"op": "reduce", "aggregate": {"kw": "median(kw)"}}, "kw: unknown criteria 'median'"),
        ({"op": "reduce", "aggregate": {"kw": "mean()"}}, "mean() needs a field"),
        ({"op": "reduce", "by": {"slot": "ts.slot("}, "aggregate": {"kw": "mean"}}, "slot: cannot read"),
    ],
)
def test_value_errors_name_the_step(step: dict[str, Any], message: str) -> None:
    with pytest.raises(ChainError) as error:
        parse(steps(step))
    assert message in str(error.value)


def test_collect_can_only_be_the_last_step() -> None:
    with pytest.raises(ChainError, match=r"steps\[0\] collect: collect can only be the last step"):
        parse(steps({"op": "collect"}, {"op": "filter", "where": "True"}))


def test_a_step_under_with_is_named_with_its_full_place() -> None:
    chain = steps({"op": "union", "with": {"source": LINK, "steps": [{"op": "filter", "where": "kw >"}]}})
    with pytest.raises(ChainError, match=r"steps\[0\]\.with\.steps\[0\] filter: where: cannot read"):
        parse(chain)
