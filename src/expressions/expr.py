"""Expression: one Python expression of a chain, parsed and checked once, evaluated for each row (Expr.java)."""

import ast
from collections.abc import Iterator, Mapping
from types import CodeType
from typing import Any

from src.exceptions import ChainError
from src.expressions.allowed import FUNCTIONS
from src.expressions.checker import Refused, check
from src.expressions.guards import bounded_mul, bounded_pow
from src.expressions.rewriter import FUNCTION_PREFIX, MUL_NAME, POW_NAME, ROW_NAME, rewrite
from src.models.rows import Row

DEFAULT_MAX_LENGTH = 500

# The only names an expression can see besides the fields of the row: no builtins, only the listed functions.
_GLOBALS: dict[str, Any] = {
    "__builtins__": {},
    POW_NAME: bounded_pow,
    MUL_NAME: bounded_mul,
    **{FUNCTION_PREFIX + name: function for name, function in FUNCTIONS.items()},
}


class Expression:
    """An expression such as "17 <= ts.hour < 21" or "max(0, load - baseline_kw)".

    `fields` are the field names it reads; the policy checker uses them. There is no `eval` of the text: the text
    is parsed, every node is checked (checker.py), the tree is rewritten (rewriter.py) and compiled, and the code
    runs with no builtins.
    """

    def __init__(self, text: str, fields: frozenset[str], code: CodeType) -> None:
        self.text = text
        self.fields = fields
        self._code = code

    @classmethod
    def parse(cls, text: str, max_length: int = DEFAULT_MAX_LENGTH) -> "Expression":
        """Raises ChainError if the text is too long, cannot be read, or holds something not allowed."""
        if len(text) > max_length:
            raise ChainError(f"the expression is longer than {max_length} characters")
        try:
            tree = ast.parse(text.strip(), mode="eval")
        except SyntaxError as error:
            raise ChainError(f"cannot read the expression {text!r}: {error.msg}") from None
        try:
            fields = check(tree)
        except Refused as error:
            raise ChainError(f"{text!r}: {error}") from None
        code = compile(rewrite(tree), "<expression>", "eval")
        return cls(text, fields, code)

    def evaluate(self, row: Row) -> Any:
        """The value of the expression for one row. Raises ChainError with the reason if it cannot be computed."""
        try:
            # Safe: the code was compiled from a tree that passed the checker, and it runs without builtins.
            return eval(self._code, _GLOBALS, _RowNames(row))
        except NameError as error:
            raise ChainError(f"unknown field {error.name} in {self.text!r}") from None
        except _UnknownField as error:
            raise ChainError(f"unknown field {error.args[0]} in {self.text!r}") from None
        except ZeroDivisionError:
            raise ChainError(f"division by zero in {self.text!r}") from None
        except KeyError as error:
            raise ChainError(f"no key {error.args[0]!r} in {self.text!r}") from None
        except (TypeError, ValueError, ArithmeticError, LookupError, AttributeError) as error:
            raise ChainError(f"{error} in {self.text!r}") from None

    def __repr__(self) -> str:
        return f"Expression({self.text!r})"


class _UnknownField(Exception):
    """field('...') named a field the row does not have."""


class _RowNames(Mapping[str, Any]):
    """The names an expression sees: the fields of the row, and `_row` for field('...').

    A missing field raises KeyError, which Python turns into NameError while evaluating.
    """

    def __init__(self, row: Row) -> None:
        self._row = row

    def __getitem__(self, name: str) -> Any:
        if name == ROW_NAME:
            return _FieldAccess(self._row)
        return self._row[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._row)

    def __len__(self) -> int:
        return len(self._row)


class _FieldAccess:
    """`_row['name']`: what field('name') becomes."""

    def __init__(self, row: Row) -> None:
        self._row = row

    def __getitem__(self, name: str) -> Any:
        if name not in self._row:
            raise _UnknownField(name)
        return self._row[name]
