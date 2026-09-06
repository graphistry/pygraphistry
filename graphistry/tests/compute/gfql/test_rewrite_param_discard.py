"""A rewrite or fast path must not silently DISCARD a param the caller set on ``rows()``.

Amplification of the named-middle ``rows(table=...)`` fix: the same *class* of defect —
an interception that answers a different question than the one asked — appears twice more
around the same two functions, and neither is caught by the table= cases.

1. THE INDEXED BYPASS IGNORES ``table``. ``_plan_indexed_middle`` (generic chain) and
   ``_try_indexed_middle_polars`` (native polars chain) decline for ``source`` /
   ``alias_endpoints`` / ``alias_prefilters``, but not for a non-default ``table``. When
   the bypass serves, the canonical traversal is SKIPPED entirely — the whole point — so
   the graph handed to the suffix is the PRE-traversal one. A ``rows(table="edges")`` then
   reads the FULL edge table instead of the traversal-narrowed edges: on the fixture below,
   12 edges instead of 3. Silent, wrong values, and it survives a following ``select``.

   This is only observable once the named-middle rewrite stops firing for a non-default
   ``table`` (the parent commit). Before it, the rewrite converted the call to
   ``binding_ops`` on BOTH the indexed and the scan path, so the two agreed — on the wrong
   table. The parent commit makes the scan path right and leaves the indexed path reading
   the whole graph, which is the shape pinned here.

2. THE REWRITE DROPS ``attach_prop_aliases`` (and ``alias_prefilters``). Both rewrites
   build a FRESH ``rows(binding_ops=...)`` rather than adding ``binding_ops`` to the call
   the caller wrote, so every other param the caller set is thrown away. For
   ``attach_prop_aliases`` — the #1711 projection pushdown — that is observable: naming the
   middle instead of spelling ``binding_ops`` by hand silently attaches EVERY alias's
   properties. Same query, same bindings, different output schema.

ENGINE LIST IS FIXED, NOT PROBED-INTO-EXISTENCE. ``polars-gpu`` joins the three engines this
file already carried: it is the GPU collect target of the SAME native polars chain, so it
runs the same rewrite and the same indexed bypass, and it must therefore land on the same
side of every gate polars does. An engine that
cannot run in the current environment reports SKIPPED with a reason; it never vanishes from
the report (the trap in ``polars_test_utils.available_nonpandas_engines()``, which silently
shrinks). The probe RUNS a query rather than importing a module, because a box where
``cudf``/``cudf_polars`` import against an incomplete CUDA runtime passes every import check
and then dies inside the first real kernel — which reads as a product failure rather than a
missing environment.

COVERAGE BOUNDARY, stated rather than hidden behind a skip: no CI lane runs cuDF or
polars-gpu (``ci-gpu.yml`` is hard-disabled and does not install the ``polars`` extra), so
those two parameters report SKIPPED on CI. They are exercised out of band on the dgx GPU box
against ``graphistry/test-rapids-official:26.02-gfql-polars`` (``docker run --gpus all`` —
omitting ``--gpus all`` FABRICATES failures rather than skipping). Treat a green CI run as
evidence for pandas + polars only.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason
from graphistry.compute.ast import (
    ASTObject, e_forward, e_undirected, n, rows, select, serialize_binding_ops,
)

NODES = pd.DataFrame({
    "id": np.arange(1, 13, dtype=np.int64),
    "kind": ["seed", "mid", "mid", "end", "end", "tail",
             "tail", "reverse", "seed", "noise", "noise", "noise"],
})
EDGES = pd.DataFrame({
    "src": [1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 4, 5],
    "dst": [2, 2, 3, 4, 5, 5, 6, 6, 7, 1, 2, 3],
    "type": ["A", "A", "A", "B", "B", "B", "C", "C", "D", "REV", "B", "B"],
    "weight": np.arange(10, 22, dtype=np.int64),
})
# Every edge in the fixture. If the bypass leaks the pre-traversal graph, this is what
# comes back — so the tests below pin the SMALL number, not merely "not empty".
ALL_EDGES = len(EDGES)

PANDAS_API_ENGINES: Tuple[str, ...] = ("pandas", "cudf")
POLARS_API_ENGINES: Tuple[str, ...] = ("polars", "polars-gpu")
ENGINES: Tuple[str, ...] = PANDAS_API_ENGINES + POLARS_API_ENGINES


def _frames(engine: str) -> Any:
    if engine in POLARS_API_ENGINES:
        pl = pytest.importorskip("polars")
        return pl.from_pandas(NODES), pl.from_pandas(EDGES)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(NODES), cudf.from_pandas(EDGES)
    return NODES, EDGES


def _graph(engine: str, *, indexed: bool) -> Any:
    nodes, edges = _frames(engine)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    return g.gfql_index_all(engine=engine) if indexed else g


@lru_cache(maxsize=None)
def _engine_skip_reason(engine: str) -> Optional[str]:
    """``None`` => this engine MUST run here; a string => a stated, checkable skip reason.

    Runs BOTH halves this file needs, because the INDEXED path is the one that reaches cupy
    kernels: a scan-only smoke test lets an incomplete CUDA install through and reports its
    kernel error as a test failure (measured on a box with cudf 25.10 importable and
    `libnvrtc.so.12` missing — 4 fabricated failures).

    Never the shape under test: a probe that runs it disarms the file — reverting the
    indexed-bypass `table` guard turned every parameter into a SKIP instead of a failure. And
    never a blanket `except Exception`: a missing module skips, a recognisable GPU-stack error
    skips with its text quoted, and ANY other failure propagates, because a skipped GPU
    parameter reads as evidence of passing.
    """
    def smoke() -> None:
        ops = [n({"id": 1}), e_forward(), n()]
        _graph(engine, indexed=False).gfql(list(ops), engine=engine)
        _graph(engine, indexed=True).gfql(list(ops), engine=engine, index_policy="force")

    return engine_skip_reason(engine, smoke)


def _require(engine: str) -> None:
    reason = _engine_skip_reason(engine)
    if reason is not None:
        pytest.skip(
            f"engine {engine!r} unavailable here ({reason}) — NOT evidence that it passes; "
            "see the COVERAGE BOUNDARY note in this module's docstring"
        )


def _run(engine: str, ops: List[ASTObject], *, indexed: bool) -> pd.DataFrame:
    kwargs: dict = {"engine": engine}
    if indexed:
        # `force` is how the index suite engages the bypass deterministically; without it
        # the cost gate may decline and the test would pass for the wrong reason.
        kwargs["index_policy"] = "force"
    out = _graph(engine, indexed=indexed).gfql(list(ops), **kwargs)._nodes
    assert out is not None
    return out.to_pandas() if hasattr(out, "to_pandas") else out


def _edge_pairs(frame: pd.DataFrame) -> List[tuple]:
    assert {"src", "dst"} <= set(map(str, frame.columns)), \
        f"expected the EDGES table, got columns {sorted(map(str, frame.columns))}"
    return sorted(zip(frame["src"].tolist(), frame["dst"].tolist()))


# ---------------------------------------------------------------------------
# 1. the indexed bypass must honour rows(table=...)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_indexed_bypass_honours_rows_table_edges(engine: str) -> None:
    """`(a {id:1})-[r]->(b)` + `rows(table='edges')` -> node 1's THREE outgoing edges.

    Indexed and unindexed must agree, and both must be the narrowed set. The absolute
    count is pinned because the failure mode is exactly `ALL_EDGES`.
    """
    _require(engine)
    ops = [n({"id": 1}, name="a"), e_forward(name="r"), n(name="b"), rows(table="edges")]
    scan = _run(engine, ops, indexed=False)
    indexed = _run(engine, ops, indexed=True)
    assert _edge_pairs(scan) == [(1, 2), (1, 2), (1, 3)]
    assert len(indexed) != ALL_EDGES, (
        f"[{engine}] the indexed bypass returned the WHOLE edge table "
        f"({ALL_EDGES} rows) — rows(table='edges') was discarded"
    )
    assert _edge_pairs(indexed) == _edge_pairs(scan)


@pytest.mark.parametrize("engine", ENGINES)
def test_indexed_bypass_honours_rows_table_edges_undirected(engine: str) -> None:
    """Undirected middle: node 1 has four incident edges, not twelve."""
    _require(engine)
    ops = [n({"id": 1}, name="a"), e_undirected(name="r"), n(name="b"), rows(table="edges")]
    scan = _run(engine, ops, indexed=False)
    indexed = _run(engine, ops, indexed=True)
    assert len(scan) == 4
    assert len(indexed) != ALL_EDGES, (
        f"[{engine}] the indexed bypass returned the WHOLE edge table "
        f"({ALL_EDGES} rows) — rows(table='edges') was discarded"
    )
    assert _edge_pairs(indexed) == _edge_pairs(scan)


@pytest.mark.parametrize("engine", ENGINES)
def test_indexed_bypass_table_edges_survives_a_projection(engine: str) -> None:
    """The leak is not schema noise: it flows through a following `select` as wrong VALUES.

    After `select(['type'])` the two paths share a one-column schema, so nothing but the
    values can distinguish them — and the indexed path returned every edge type in the
    graph rather than the three on node 1's outgoing edges.
    """
    _require(engine)
    ops = [n({"id": 1}, name="a"), e_forward(name="r"), n(name="b"),
           rows(table="edges"), select([("type", "type")])]
    scan = _run(engine, ops, indexed=False)
    indexed = _run(engine, ops, indexed=True)
    assert sorted(scan["type"].tolist()) == ["A", "A", "A"]
    assert sorted(indexed["type"].tolist()) == sorted(scan["type"].tolist()), (
        f"[{engine}] indexed projection saw {len(indexed)} rows, scan saw {len(scan)}"
    )


@pytest.mark.parametrize("engine", [
    "pandas",
    "cudf",
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.xfail(
        strict=True,
        reason="#1803: the indexed bindings bypass excludes Engine.POLARS_GPU "
               "(index/bindings.py gate), so polars-gpu always takes the scan path",
    )),
])
@pytest.mark.route_engaged("indexed-kernel")
def test_indexed_bypass_still_serves_a_bare_rows(engine: str) -> None:
    """THE NEGATIVE SIDE: declining on a non-default `table` must not decline everything.

    Without this, "never take the bypass" would pass every test above while silently
    retiring the indexed bindings path. Asserted on the trace, not on the answer — the
    answer is identical either way, which is precisely why the regression would be quiet.

    ``polars-gpu`` STRICT-xfails, and that is the point of adding the parameter (#1802). The
    native polars chain hands `Engine.POLARS_GPU` to `try_indexed_connected_bindings_state`
    whenever the GPU target is active, but that function's first gate admits only
    `(PANDAS, CUDF, POLARS)` — so the bypass reports `served: False,
    reason: 'unsupported_engine'` and polars-gpu silently runs the canonical scan. Values are
    unaffected (every other case in this file passes on polars-gpu); what is lost is the
    optimization, with no signal. A strict xfail rather than a skip so that whoever widens the
    gate is told to delete the marker instead of leaving a stale "known gap" behind.
    """
    _require(engine)
    from graphistry.compute.gfql.index import index_trace

    ops = [n({"id": 1}, name="a"), e_forward(name="r"), n(name="b"), rows()]
    with index_trace() as captured:
        _graph(engine, indexed=True).gfql(list(ops), engine=engine, index_policy="force")
    served = [step for step in captured
              if step.get("operation") == "indexed_traversal" and step.get("served")]
    assert served, (
        f"[{engine}] a bare rows() over a named middle no longer engages the indexed "
        f"bindings bypass; trace={[dict(s) for s in captured]}"
    )


# ---------------------------------------------------------------------------
# 2. the named-middle rewrite must carry the caller's other rows() params
# ---------------------------------------------------------------------------

MIDDLE: List[ASTObject] = [n({"id": 1}, name="a"), e_forward(name="r"), n(name="b")]


@pytest.mark.parametrize("engine", ENGINES)
def test_named_middle_rewrite_keeps_attach_prop_aliases(engine: str) -> None:
    """Naming the middle must not throw away the caller's projection pushdown.

    `[...named ops..., rows(attach_prop_aliases=['a'])]` and the hand-spelled
    `rows(binding_ops=<those ops>, attach_prop_aliases=['a'])` are the same query; the
    rewrite exists only to turn the first into the second. Building a FRESH rows() drops
    the hint, so the first attaches `b`'s properties and the second does not.

    Compared within one engine (the two surfaces emit different scaffolding columns), and
    against the explicit spelling rather than a hardcoded column list, so the assertion
    tracks whatever the bindings builder emits.
    """
    _require(engine)
    binding_ops = serialize_binding_ops(MIDDLE)
    rewritten = _run(engine, list(MIDDLE) + [rows(attach_prop_aliases=["a"])], indexed=False)
    explicit = _run(engine, [rows(binding_ops=binding_ops, attach_prop_aliases=["a"])],
                    indexed=False)
    unrestricted = _run(engine, [rows(binding_ops=binding_ops)], indexed=False)

    # Guard the test itself: if pushing down 'a' stopped changing the schema there would
    # be nothing to observe and the assertion below would hold vacuously.
    assert set(map(str, explicit.columns)) < set(map(str, unrestricted.columns)), \
        f"[{engine}] attach_prop_aliases no longer narrows the bindings schema"

    dropped = {c for c in map(str, unrestricted.columns)} - {c for c in map(str, explicit.columns)}
    leaked = dropped & set(map(str, rewritten.columns))
    assert not leaked, (
        f"[{engine}] rows(attach_prop_aliases=['a']) after a NAMED middle attached "
        f"properties it excluded: {sorted(leaked)}"
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_named_middle_rewrite_keeps_attach_prop_aliases_values(engine: str) -> None:
    """The pushdown must narrow the SCHEMA without changing the bindings themselves."""
    _require(engine)
    binding_ops = serialize_binding_ops(MIDDLE)
    rewritten = _run(engine, list(MIDDLE) + [rows(attach_prop_aliases=["a"])], indexed=False)
    explicit = _run(engine, [rows(binding_ops=binding_ops, attach_prop_aliases=["a"])],
                    indexed=False)
    assert len(rewritten) == len(explicit) == 3
    shared = [c for c in map(str, explicit.columns) if c in set(map(str, rewritten.columns))]
    assert "a.id" in shared and "b" in shared
    for col in shared:
        assert sorted(rewritten[col].tolist()) == sorted(explicit[col].tolist()), \
            f"[{engine}] column {col!r} differs between the rewritten and explicit spellings"


# ---------------------------------------------------------------------------
# 3. the third instance of the class: rows(alias_prefilters=...) — fixed (#1804)
# ---------------------------------------------------------------------------

# Unseeded so the prefilter is the ONLY thing that can narrow the result.
OPEN_MIDDLE: List[ASTObject] = [n(name="a"), e_forward(name="r"), n(name="b")]
ALIAS_PREFILTER = {"a": [{"kind": "expr", "text": "a.id < 3"}]}


@pytest.mark.parametrize("engine", ENGINES)
def test_alias_prefilters_are_honoured_by_the_bindings_builder(engine: str) -> None:
    """`alias_prefilters` narrows the bindings on EVERY engine (#1804, fixed).

    The native polars row op used to call its bindings builder with `binding_ops` and
    `attach_prop_aliases` only, so the param never reached it and the filtered-out rows
    came back: same query, 12 rows instead of 5. The builder now applies the specs
    natively per alias (`_apply_alias_prefilters_polars`) at the same points the pandas
    builder does, and a spec the polars lowering cannot serve raises a typed
    NotImplementedError naming `alias_prefilters` instead of dropping the filter
    (see test_engine_polars_alias_prefilters.py).
    """
    _require(engine)
    binding_ops = serialize_binding_ops(OPEN_MIDDLE)
    unfiltered = _run(engine, [rows(binding_ops=binding_ops)], indexed=False)
    filtered = _run(engine, [rows(binding_ops=binding_ops, alias_prefilters=ALIAS_PREFILTER)],
                    indexed=False)
    assert len(unfiltered) == ALL_EDGES
    assert len(filtered) == 5, (
        f"[{engine}] alias_prefilters was discarded: {len(filtered)} rows, "
        f"unfiltered is {len(unfiltered)}"
    )
    assert max(filtered["a.id"].tolist()) < 3
