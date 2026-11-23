"""The expressions of a chain: single Python expressions, checked before they run (DESIGN.md section 6).

- allowed.py  what an expression may contain: syntax, functions, methods
- guards.py   functions that bound the size of results (**, *, range)
- checker.py  refuses anything not allowed; collects the fields an expression reads
- rewriter.py turns a checked tree into one that is safe to compile
- expr.py     Expression: parse once, evaluate per row
"""

from src.expressions.expr import Expression

__all__ = ["Expression"]
