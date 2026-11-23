from typing import Any

import pytest

from src.exceptions import ChainError
from src.expressions import Expression
from tests.utils import t

ROW = {
    "ts": t("2026-07-25T18:15"),
    "kw": 312.5,
    "load": 400.0,
    "baseline_kw": 420.0,
    "net": -71.1,
    "flag": True,
    "nothing": None,
    "meta": {"origin": "MG17"},
    "values": [1, 2, 3],
    "load (kW)": 5.0,
}


def evaluate(text: str, row: dict[str, Any] = ROW) -> Any:
    return Expression.parse(text).evaluate(row)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("17 <= ts.hour < 21", True),
        ("ts.hour in range(17, 21)", True),
        ("ts.isoweekday() in [6, 7]", True),
        ("ts.between('2026-07-25', '2026-07-25T23:59')", True),
        ("'2026-07-25' <= ts < '2026-07-26'", True),
        ("ts >= '2026-07-25T18:15:00'", True),
        ("ts.slot()", 73),
        ("ts.date().isoweekday()", 6),
        ("(ts - ts).total_seconds()", 0.0),
        ("max(0, load - baseline_kw)", 0),
        ("max(-200, min(200, kw))", 200),
        ("abs(net)", 71.1),
        ("kw * 2 if kw > 300 else 0", 625.0),
        ("-7 % 3", 2),  # Python behaviour
        ("7 // 2", 3),
        ("2 ** 10", 1024),
        ("round(kw / 3, 1)", 104.2),
        ("sqrt(16) + floor(2.7) + ceil(2.1)", 9.0),
        ("flag and kw > 100", True),
        ("not flag", False),
        ("flag + 1", 2),  # True counts as 1
        ("nothing is None", True),
        ("meta['origin'].lower() == 'mg17'", True),
        ("meta.get('missing', 0)", 0),
        ("values[0] + values[-1]", 4),
        ("field('load (kW)') * 2", 10.0),
        ("'ab' * 3", "ababab"),
    ],
)
def test_evaluates_python_expressions(text: str, expected: Any) -> None:
    assert evaluate(text) == expected


@pytest.mark.parametrize(
    "text, fields",
    [
        ("17 <= ts.hour < 21", {"ts"}),
        ("max(0, load - baseline_kw)", {"load", "baseline_kw"}),
        ("field('load (kW)') * 2", {"load (kW)"}),
        ("meta['origin']", {"meta"}),
        ("1 + 2", set()),
    ],
)
def test_knows_the_fields_it_reads(text: str, fields: set[str]) -> None:
    assert Expression.parse(text).fields == fields


def test_a_function_never_clashes_with_a_field_of_the_same_name() -> None:
    assert evaluate("abs(abs - 10)", {"abs": 4}) == 6


def test_the_hour_as_a_point_on_a_circle() -> None:
    """The cyclical encoding of v3 training inputs (DESIGN.md section 17): 18:15 is three quarters round."""
    assert evaluate("round(sin(2 * pi * ts.hour / 24), 9)") == -1.0
    assert evaluate("round(cos(2 * pi * ts.hour / 24), 9)") == 0.0
    assert Expression.parse("sin(2 * pi * ts.hour / 24)").fields == {"ts"}  # pi is a constant, not a field


def test_a_field_named_like_a_constant_is_read_with_field() -> None:
    assert evaluate("field('pi') + pi", {"pi": 1.0}) == pytest.approx(4.14159265)
    with pytest.raises(ChainError, match="pi"):
        Expression.parse("pi()")


@pytest.mark.parametrize(
    "text, reason",
    [
        ("ts.__class__.__base__.__subclasses__()", "is not allowed"),
        ("'{0.__class__}'.format(ts)", ".format is not allowed"),
        ("__import__('os').system('id')", "is not allowed"),
        ("getattr(ts, 'hour')", "getattr() is not allowed"),
        ("open('/etc/passwd')", "open() is not allowed"),
        ("ts.replace(hour=1)", ".replace is not allowed"),
        ("_row", "names that start with _"),
        ("(lambda: 1)()", "lambda is not allowed"),
        ("[x for x in values]", "comprehensions is not allowed"),
        ("(y := 1)", ":= is not allowed"),
        ("f'{kw}'", "f-strings is not allowed"),
        ("max(*values)", "* in calls is not allowed"),
        ("{'a': 1}", "objects is not allowed"),
        ("values[0]()", "only the listed functions and methods"),
        ("field(kw)", "field() takes one name in quotes"),
        ("b'bytes'", "is not allowed"),
    ],
)
def test_refuses_what_is_not_allowed_before_running(text: str, reason: str) -> None:
    with pytest.raises(ChainError, match=reason.replace("(", r"\(").replace(")", r"\)").replace("*", r"\*")):
        Expression.parse(text)


@pytest.mark.parametrize("text", ["hour >= 17 && hour < 21", "c ? a : b", "a +", "x = 1", "import os"])
def test_text_that_is_not_a_python_expression_cannot_be_read(text: str) -> None:
    with pytest.raises(ChainError, match="cannot read the expression"):
        Expression.parse(text)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("kw2 + 1", "unknown field kw2"),
        ("field('nope')", "unknown field nope"),
        ("kw / 0", "division by zero"),
        ("kw > 'text'", "not supported"),
        ("nothing + 1", "unsupported operand"),
        ("ts < 'tomorrow'", "cannot read the time"),
        ("kw.hour", "has no attribute"),
        ("meta['missing']", "no key 'missing'"),
        ("values[10]", "index out of range"),
    ],
)
def test_errors_while_running_name_the_reason_and_the_expression(text: str, reason: str) -> None:
    with pytest.raises(ChainError, match=reason) as error:
        evaluate(text)
    assert text in str(error.value)


@pytest.mark.parametrize(
    "text, reason",
    [
        ("2 ** 1000", "exponents up to 100"),
        ("(10 ** 100) ** 100", "too large"),
        ("'a' * 100000", "longer than 10000"),
        ("ts.hour in range(10 ** 9)", "range allows up to"),
    ],
)
def test_limits_the_size_of_results(text: str, reason: str) -> None:
    with pytest.raises(ChainError, match=reason):
        evaluate(text)


def test_limits_the_length_of_an_expression() -> None:
    with pytest.raises(ChainError, match="longer than 20 characters"):
        Expression.parse("kw + " * 10 + "1", max_length=20)
