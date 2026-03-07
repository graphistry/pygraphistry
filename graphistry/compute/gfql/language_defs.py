"""Shared GFQL language definitions for operators, builtins, and aggregates."""

from __future__ import annotations

import operator
from typing import Any, Callable, Final


GFQL_COMPARISON_BINARY_OPS: Final[dict[str, Callable[[Any, Any], Any]]] = {
    "=": operator.eq,
    "!=": operator.ne,
    "<>": operator.ne,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}

GFQL_BOOLEAN_BINARY_OPS: Final[frozenset[str]] = frozenset({"or", "and"})
GFQL_COMPARISON_BINARY_OP_NAMES: Final[frozenset[str]] = frozenset(GFQL_COMPARISON_BINARY_OPS)
GFQL_STRING_PREDICATE_OPS: Final[frozenset[str]] = frozenset({"contains", "starts_with", "ends_with"})
GFQL_ARITHMETIC_BINARY_OPS: Final[frozenset[str]] = frozenset({"+", "-", "*", "/", "%"})
GFQL_ALLOWED_BINARY_OPS: Final[frozenset[str]] = (
    GFQL_BOOLEAN_BINARY_OPS
    | GFQL_COMPARISON_BINARY_OP_NAMES
    | GFQL_STRING_PREDICATE_OPS
    | GFQL_ARITHMETIC_BINARY_OPS
    | frozenset({"in"})
)

GFQL_ALLOWED_UNARY_OPS: Final[frozenset[str]] = frozenset({"+", "-", "not"})
GFQL_ALLOWED_QUANTIFIERS: Final[frozenset[str]] = frozenset({"any", "all", "none", "single"})

GFQL_SCALAR_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"size", "abs", "toboolean", "tostring", "coalesce", "sign"}
)
GFQL_SEQUENCE_FUNCTIONS: Final[frozenset[str]] = frozenset({"head", "tail", "reverse"})
GFQL_PATH_VALUE_FUNCTIONS: Final[frozenset[str]] = frozenset({"nodes", "relationships"})
GFQL_ALLOWED_FUNCTIONS: Final[frozenset[str]] = (
    GFQL_SCALAR_FUNCTIONS
    | GFQL_SEQUENCE_FUNCTIONS
    | GFQL_PATH_VALUE_FUNCTIONS
    | GFQL_ALLOWED_QUANTIFIERS
)

GFQL_GROUPBY_AGG_METHODS: Final[dict[str, str]] = {
    "count": "count",
    "count_distinct": "nunique",
    "sum": "sum",
    "min": "min",
    "max": "max",
    "avg": "mean",
    "mean": "mean",
}

GFQL_AGGREGATION_FUNCTIONS: Final[frozenset[str]] = frozenset(
    set(GFQL_GROUPBY_AGG_METHODS) | {"collect"}
)
GFQL_ORDER_AGG_ALIAS_FUNCTIONS: Final[frozenset[str]] = frozenset(
    {"count", "sum", "min", "max", "avg", "mean", "collect"}
)

GFQL_COMPARISON_GRAMMAR_ALTS: Final[str] = " | ".join(
    f'"{op}"' for op in sorted(GFQL_COMPARISON_BINARY_OP_NAMES, key=lambda value: (-len(value), value))
)
