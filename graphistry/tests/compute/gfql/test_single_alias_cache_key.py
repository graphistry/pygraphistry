"""Completeness of the single-alias lowering cache key.

``lower_single_alias_predicate`` is memoized, so anything the lowering depends on
must be IN the key or a stale entry is served. Two of those requirements were
argued only in a docstring; a wrong key is a silent wrong-answer bug (a cached
expression built for another dtype), which no existing test would catch.
"""
import pytest

pl = pytest.importorskip("polars")  # gfql-core lane has no polars; collection must not fail

from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import (  # noqa: E402
    _single_alias_cache_key,
)


def _key(schema, expr="(a.x = 1)", alias="a", nan_free=False):
    return _single_alias_cache_key(expr, alias, schema, nan_free)


def test_dtype_is_in_the_key_not_just_column_names() -> None:
    """Same predicate, same column NAMES, different dtype lowers differently -- a
    float operand gains a NaN mask, a string-vs-numeric pair declines outright. So
    names alone would serve a stale expression."""
    assert _key({"x": pl.Int64}) != _key({"x": pl.Float64})
    assert _key({"x": pl.Int64}) != _key({"x": pl.Utf8})


def test_parameterized_dtypes_do_not_collide() -> None:
    """The key uses ``str(dtype)`` rather than the dtype OBJECT because polars
    equates a dtype class with its parameterized instances, so as dict keys they
    would HIT across dtypes that lower differently."""
    assert pl.Datetime == pl.Datetime("ns")  # the hazard, asserted so it is visible
    assert _key({"t": pl.Datetime}) != _key({"t": pl.Datetime("ns")})
    assert _key({"t": pl.Datetime("ns")}) != _key({"t": pl.Datetime("us")})


def test_every_other_lowering_input_is_keyed() -> None:
    base = {"x": pl.Int64}
    assert _key(base, expr="(a.x = 1)") != _key(base, expr="(a.x = 2)")
    assert _key(base, alias="a") != _key(base, alias="b")
    assert _key(base, nan_free=False) != _key(base, nan_free=True)


def test_key_is_hashable_and_stable() -> None:
    k = _key({"x": pl.Int64})
    assert hash(k) == hash(_key({"x": pl.Int64}))
    assert {k: 1}[_key({"x": pl.Int64})] == 1


def test_schema_order_is_part_of_the_key() -> None:
    """``lower_expr`` resolves names against ``list(schema)``, so the ORDER is an
    input, not incidental."""
    assert _key({"x": pl.Int64, "y": pl.Int64}) != _key({"y": pl.Int64, "x": pl.Int64})
