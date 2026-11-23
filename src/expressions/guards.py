"""Functions that run in place of `**`, `*` and `range`, so that one expression cannot use all the memory or CPU.

The rewriter (rewriter.py) turns `a ** b` into `_pow(a, b)` and `a * b` into `_mul(a, b)`.
"""

from typing import Any

MAX_EXPONENT = 100
"""The largest exponent of `**`."""

MAX_INT_BITS = 10_000
"""The largest whole number `**` may produce, in bits (about 3,000 digits)."""

MAX_ITEMS = 10_000
"""The longest text or list that `*` may produce, and the longest `range`."""


class LimitExceeded(ValueError):
    """An expression would produce a value larger than the limits above."""


def bounded_pow(base: Any, exponent: Any) -> Any:
    """`base ** exponent`, with the exponent and the size of a whole-number result bounded."""
    if isinstance(exponent, int | float) and abs(exponent) > MAX_EXPONENT:
        raise LimitExceeded(f"** allows exponents up to {MAX_EXPONENT}, not {exponent}")
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0:
        if base.bit_length() * exponent > MAX_INT_BITS:
            raise LimitExceeded("** would give a number that is too large")
    return base**exponent


def bounded_mul(left: Any, right: Any) -> Any:
    """`left * right`, with text and lists repeated up to MAX_ITEMS items."""
    for sequence, count in ((left, right), (right, left)):
        if isinstance(sequence, str | list | tuple) and isinstance(count, int) and len(sequence) * count > MAX_ITEMS:
            raise LimitExceeded(f"* would make text or a list longer than {MAX_ITEMS}")
    return left * right


def bounded_range(*args: int) -> range:
    """`range(...)` with at most MAX_ITEMS numbers; `x in range(a, b)` stays as fast as before."""
    numbers = range(*args)
    if len(numbers) > MAX_ITEMS:
        raise LimitExceeded(f"range allows up to {MAX_ITEMS} numbers")
    return numbers
