"""Turns a checked expression tree into the tree that is compiled.

- abs(x)            -> _fn_abs(x)            a function never clashes with a field of the same name
- field('load kW')  -> _row['load kW']       a field whose name is not a Python name
- a ** b            -> _pow(a, b)            bounded size (guards.py)
- a * b             -> _mul(a, b)            bounded size (guards.py)

The names that start with `_` cannot be written in a chain (the checker refuses them), so only this module
can introduce them.
"""

import ast

from src.expressions.allowed import FIELD_FUNCTION

FUNCTION_PREFIX = "_fn_"
ROW_NAME = "_row"
POW_NAME = "_pow"
MUL_NAME = "_mul"

_GUARDED_OPERATORS: dict[type[ast.operator], str] = {ast.Pow: POW_NAME, ast.Mult: MUL_NAME}


def rewrite(tree: ast.Expression) -> ast.Expression:
    """The tree to compile. `tree` must have passed checker.check."""
    rewritten = _Rewriter().visit(tree)
    return ast.fix_missing_locations(rewritten)


class _Rewriter(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        function = node.func
        if not isinstance(function, ast.Name):
            return node  # a method call: left as it is
        if function.id == FIELD_FUNCTION:
            return ast.Subscript(value=ast.Name(ROW_NAME, ast.Load()), slice=node.args[0], ctx=ast.Load())
        node.func = ast.Name(FUNCTION_PREFIX + function.id, ast.Load())
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        guard = _GUARDED_OPERATORS.get(type(node.op))
        if guard is None:
            return node
        return ast.Call(func=ast.Name(guard, ast.Load()), args=[node.left, node.right], keywords=[])
