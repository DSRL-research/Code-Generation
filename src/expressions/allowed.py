"""What an expression may contain (DESIGN.md section 6). To allow more, add it here and to the table in DESIGN.md.

Everything that is not listed is refused when the chain is parsed, before anything runs. Attribute access is
limited to METHODS because a plain `eval` can be escaped through attributes, for example
`().__class__.__base__.__subclasses__()` or `'{0.__class__}'.format(x)`. Only add names that are harmless on
every value an expression can reach: JSON values, times, dates, durations.
"""

import ast
import math
from collections.abc import Callable
from typing import Any

from src.expressions.guards import bounded_range
from src.ml.vectors import cosine, distance

FUNCTIONS: dict[str, Callable[..., Any]] = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
    "floor": math.floor,
    "ceil": math.ceil,
    "range": bounded_range,
    # vectors, e.g. embeddings
    "distance": distance,
    "cosine": cosine,
}

FIELD_FUNCTION = "field"
"""field('load (kW)') reads a field whose name is not a Python name."""

METHODS: frozenset[str] = frozenset({
    # times (ts, ts.date())
    "year", "month", "day", "hour", "minute", "second", "weekday", "isoweekday", "date", "between", "slot",
    # durations (ts - other)
    "days", "seconds", "total_seconds",
    # text
    "lower", "upper", "strip", "startswith", "endswith",
    # objects
    "get",
    # numbers
    "is_integer",
})  # fmt: skip

CONSTANT_TYPES = (int, float, str, bool, type(None))

NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    # values
    ast.Constant, ast.Name, ast.Load, ast.List, ast.Tuple,
    # access and calls; Attribute and Call are checked further by the checker
    ast.Attribute, ast.Subscript, ast.Slice, ast.Call, ast.keyword,
    # arithmetic
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.UnaryOp, ast.USub, ast.UAdd, ast.Not,
    # logic and comparison
    ast.BoolOp, ast.And, ast.Or,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is, ast.IsNot,
    # a if c else b
    ast.IfExp,
)  # fmt: skip

REFUSED_NAMES: dict[type[ast.AST], str] = {
    ast.Lambda: "lambda",
    ast.ListComp: "comprehensions",
    ast.SetComp: "comprehensions",
    ast.DictComp: "comprehensions",
    ast.GeneratorExp: "comprehensions",
    ast.NamedExpr: ":=",
    ast.JoinedStr: "f-strings",
    ast.Starred: "* in calls",
    ast.Dict: "{...} objects",
    ast.Set: "{...} sets",
}
"""Readable names for refused syntax, used in error messages."""
