"""Compatibility shim for row expression dispatch helpers."""

from graphistry.compute.gfql.row.dispatch import (
    apply_string_predicate_scalar,
    apply_string_predicate_series,
    eval_sequence_fn_scalar,
    eval_sequence_fn_series,
)

__all__ = [
    "apply_string_predicate_scalar",
    "apply_string_predicate_series",
    "eval_sequence_fn_scalar",
    "eval_sequence_fn_series",
]
