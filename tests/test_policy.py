from typing import Any

import pytest

from src.exceptions import PolicyFileError, PolicyViolation
from src.policy import Policy, PolicyChecker
from src.services.chain_parser import ChainParser
from tests.utils import RESOURCES, load_chain, load_json

POLICY = Policy.load(RESOURCES / "policy.json")
RAW = "http://127.0.0.1:8000/readings?start=2026-07-29"
FILE = "http://127.0.0.1:8000/files/battery_flex"
WINDOW = {"op": "reduce", "window": "15m", "aggregate": {"kw": "mean(kw)"}}


def check(chain: dict[str, Any], policy: Policy = POLICY):
    return PolicyChecker(policy).check(ChainParser().parse(chain))


def refused(chain: dict[str, Any], message: str, policy: Policy = POLICY) -> None:
    with pytest.raises(PolicyViolation) as error:
        check(chain, policy)
    assert message in str(error.value), str(error.value)


# ---------- sources ----------


def test_the_first_matching_rule_says_whether_a_link_is_raw() -> None:
    assert POLICY.source_rule(RAW).raw is True
    assert POLICY.source_rule("http://127.0.0.1:9/readings-gap").raw is True
    assert POLICY.source_rule(FILE).raw is False


@pytest.mark.parametrize(
    "link, message",
    [
        ("http://localhost:8000/files/x", "is not an allowed source"),  # another host
        ("http://127.0.0.1.evil.com/files/x", "is not an allowed source"),
        ("https://127.0.0.1:8000/files/x", "is not an allowed source"),  # another scheme
        ("http://user@127.0.0.1:8000/files/x", "links with a user name"),
        ("http://127.0.0.1:8000/files/../readings", "links with .. or %"),
        ("http://127.0.0.1:8000/%72eadings", "links with .. or %"),
        ("http://127.0.0.1:x/files", "port that cannot be read"),
    ],
)
def test_links_that_are_refused(link: str, message: str) -> None:
    with pytest.raises(PolicyViolation, match=message):
        POLICY.source_rule(link)


@pytest.mark.parametrize(
    "match", ["http://*/x", "http://localhost:8000*", "localhost:8000/x", "ftp://host/x", "http://host:8*/x"]
)
def test_patterns_must_write_scheme_host_and_port_in_full(match: str) -> None:
    data = load_json("policy.json")
    data["sources"] = [{"match": match, "raw": False}]
    with pytest.raises(PolicyFileError, match="source pattern"):
        Policy.from_json(data)


def test_a_port_given_in_the_pattern_must_match() -> None:
    data = load_json("policy.json")
    data["sources"] = [{"match": "http://127.0.0.1:8000/*", "raw": False}]
    policy = Policy.from_json(data)
    assert policy.source_rule("http://127.0.0.1:8000/a").raw is False
    with pytest.raises(PolicyViolation):
        policy.source_rule("http://127.0.0.1:8001/a")


# ---------- raw rows ----------


def test_allowed_on_raw_rows_a_time_filter_then_a_window() -> None:
    report = check(
        {
            "source": RAW,
            "steps": [
                {"op": "filter", "where": "17 <= ts.hour < 21"},
                WINDOW,
                {"op": "map", "set": {"x": "kw * 2"}},
                {"op": "collect"},
            ],
        }
    )
    assert report.to_json() == {"sources": [{"link": RAW, "raw": True}]}


@pytest.mark.parametrize(
    "steps, message",
    [
        ([], "the chain would give raw rows"),
        (
            [{"op": "filter", "where": "kw > 100"}, WINDOW],
            "steps[0] filter: on raw rows a filter may only read ['ts'], not ['kw']",
        ),
        ([{"op": "map", "set": {"x": "kw"}}, WINDOW], "steps[0] map: not allowed on raw rows"),
        ([{"op": "collect"}], "steps[0] collect: not allowed on raw rows"),
        (
            [{"op": "reduce", "by": {"slot": "ts.slot()"}, "aggregate": {"kw": "mean"}}],
            "steps[0] reduce: on raw rows the first reduce must use a window",
        ),
        ([{"op": "reduce", "aggregate": {"kw": "mean"}}], "the first reduce must use a window"),
        ([{"op": "reduce", "window": "5m", "aggregate": {"kw": "mean"}}], "shorter than 15 minutes"),
        ([{"op": "reduce", "window": "15m", "aggregate": {"kw": "max"}}], "max not allowed on raw rows"),
        ([{"op": "reduce", "window": "15m", "aggregate": {"x": "mean(kw)", "y": "min(kw)"}}], "min not allowed"),
    ],
)
def test_refused_on_raw_rows(steps: list[dict[str, Any]], message: str) -> None:
    refused({"source": RAW, "steps": steps}, message)


def test_the_chain_under_with_is_checked_too() -> None:
    raw_inside = {"source": FILE, "steps": [{"op": "union", "with": {"source": RAW}}]}
    refused(raw_inside, 'steps[0] union: the chain under "with" still gives raw rows')
    wrong_inside = {
        "source": FILE,
        "steps": [{"op": "union", "with": {"source": RAW, "steps": [{"op": "filter", "where": "kw > 1"}, WINDOW]}}],
    }
    refused(wrong_inside, "steps[0].with.steps[0] filter: on raw rows a filter may only read")


def test_a_source_not_allowed_is_named_by_its_place() -> None:
    chain = {"source": FILE, "steps": [{"op": "union", "with": {"source": "http://localhost:1/x"}}]}
    refused(chain, "steps[0].with.source: http://localhost:1/x is not an allowed source")


def test_rows_that_are_not_raw_can_use_any_operator() -> None:
    check(
        {
            "source": FILE,
            "steps": [
                {"op": "map", "set": {"x": "1"}},
                {"op": "top", "n": 1, "by": "x"},
                {"op": "reduce", "aggregate": {"*": "max"}},
            ],
        }
    )


def test_example_b_is_allowed() -> None:
    report = check(load_chain("example_b", "http://127.0.0.1:8000"))
    assert [use.raw for use in report.sources] == [False, True]


# ---------- the policy file ----------


def test_an_operator_missing_from_the_policy_is_refused() -> None:
    data = load_json("policy.json")
    del data["operators"]["top"]
    refused(
        {"source": FILE, "steps": [{"op": "top", "n": 1, "by": "x"}]},
        "steps[0] top: this operator is not allowed",
        Policy.from_json(data),
    )


def test_onraw_true_allows_an_operator_on_raw_rows() -> None:
    data = load_json("policy.json")
    data["operators"]["top"] = {"onRaw": True}
    check({"source": RAW, "steps": [{"op": "top", "n": 1, "by": "kw"}, WINDOW]}, Policy.from_json(data))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda data: data["operators"].update(sort={"onRaw": False}), "unknown operators ['sort']"),
        (lambda data: data["operators"].update(map={"onRaw": "sometimes"}), "map: onRaw must be true or false"),
        (lambda data: data["operators"].update(filter={"onRaw": "yes"}), 'filter: onRaw must be false, true or "time'),
        (lambda data: data["operators"].update(reduce={"onRaw": {"criteria": []}}), "reduce: onRaw must be"),
        (lambda data: data.pop("sources"), "sources"),
    ],
)
def test_a_policy_file_that_cannot_be_read(change, message: str) -> None:
    data = load_json("policy.json")
    change(data)
    with pytest.raises(PolicyFileError, match=message.replace("[", r"\[").replace("]", r"\]")):
        Policy.from_json(data)


def test_the_sample_policy_of_the_project_loads() -> None:
    policy = Policy.load(RESOURCES.parent.parent / "policy.json")
    assert policy.source_rule("http://localhost:8000/microgrids/2/readings?start=2026-07-03").raw is True
    assert policy.source_rule("http://localhost:8000/microgrids/2/demand_forecast").raw is False
