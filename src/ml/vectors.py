"""Comparing two vectors, such as two embeddings. Also available in expressions: distance(a, b), cosine(a, b)."""

import math
from typing import Any


def distance(first: Any, second: Any) -> float:
    """Euclidean distance: 0 for equal vectors, larger for vectors further apart."""
    pairs = _pairs(first, second)
    return math.sqrt(sum((a - b) ** 2 for a, b in pairs))


def cosine(first: Any, second: Any) -> float:
    """Cosine similarity: 1 for vectors that point the same way, 0 for unrelated, -1 for opposite."""
    pairs = _pairs(first, second)
    norm = math.sqrt(sum(a * a for a, _ in pairs)) * math.sqrt(sum(b * b for _, b in pairs))
    if norm == 0:
        raise ValueError("cosine of a zero vector")
    return sum(a * b for a, b in pairs) / norm


def _pairs(first: Any, second: Any) -> list[tuple[float, float]]:
    if not isinstance(first, list | tuple) or not isinstance(second, list | tuple):
        raise TypeError("vectors must be lists of numbers")
    if len(first) != len(second):
        raise ValueError(f"vectors of different lengths: {len(first)} and {len(second)}")
    return list(zip(first, second, strict=True))
