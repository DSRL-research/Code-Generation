"""Checks a parsed expression against allowed.py, and collects the fields it reads.

Data reaches an expression only through a field name (or field('...')), so the collected set is exact:
`ts.hour` reads only `ts`. The policy checker relies on this to allow filters on raw rows.
"""

import ast

from src.expressions.allowed import CONSTANT_TYPES, FIELD_FUNCTION, FUNCTIONS, METHODS, NODES, REFUSED_NAMES


class Refused(ValueError):
    """The expression holds something that is not allowed."""


def check(tree: ast.Expression) -> frozenset[str]:
    """Raises Refused if the tree holds anything not allowed; returns the names of the fields it reads."""
    checker = _Checker()
    checker.visit(tree)
    return frozenset(checker.fields)


class _Checker(ast.NodeVisitor):
    def __init__(self) -> None:
        self.fields: set[str] = set()

    def generic_visit(self, node: ast.AST) -> None:
        if not isinstance(node, NODES):
            raise Refused(f"{REFUSED_NAMES.get(type(node), type(node).__name__)} is not allowed")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id.startswith("_"):
            raise Refused(f"names that start with _ are not allowed: {node.id}")
        self.fields.add(node.id)

    def visit_Constant(self, node: ast.Constant) -> None:
        if not isinstance(node.value, CONSTANT_TYPES):
            raise Refused(f"the value {node.value!r} is not allowed")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr not in METHODS:
            raise Refused(f".{node.attr} is not allowed; the allowed methods are {', '.join(sorted(METHODS))}")
        self.visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if any(keyword.arg is None for keyword in node.keywords):
            raise Refused("** in calls is not allowed")
        function = node.func
        if isinstance(function, ast.Name) and function.id == FIELD_FUNCTION:
            self._field_call(node)
            return
        if isinstance(function, ast.Name):
            if function.id not in FUNCTIONS:
                raise Refused(f"{function.id}() is not allowed; the functions are {', '.join(sorted(FUNCTIONS))}")
        elif isinstance(function, ast.Attribute):
            self.visit(function)  # checks the method name and the value it is called on
        else:
            self.visit(function)  # reports refused syntax such as a lambda first
            raise Refused("only the listed functions and methods can be called")
        for argument in node.args:
            self.visit(argument)
        for keyword in node.keywords:
            self.visit(keyword.value)

    def _field_call(self, node: ast.Call) -> None:
        arguments = node.args
        if node.keywords or len(arguments) != 1 or not isinstance(arguments[0], ast.Constant):
            raise Refused(f"{FIELD_FUNCTION}() takes one name in quotes, for example field('load (kW)')")
        name = arguments[0].value
        if not isinstance(name, str):
            raise Refused(f"{FIELD_FUNCTION}() takes one name in quotes, for example field('load (kW)')")
        self.fields.add(name)
