"""Lazy Polars GFQL backend.

Builds one ``pl.LazyFrame`` plan per query and collects ONCE on the active
execution target (CPU or GPU) — so polars (CPU) and polars-gpu are two TARGETS of
this single engine, not two engines. DRY: reuses the cypher-AST -> ``pl.Expr``
lowering from this package (``lower_expr`` / ``predicate_to_expr`` / agg /
select / order_by lowering) verbatim — only the materialization strategy differs
(eager ``.collect()`` per op  ->  lazy plan + collect-once).
"""

from .hop_eager import hop_polars
from .chain import chain_polars

__all__ = ["hop_polars", "chain_polars"]
