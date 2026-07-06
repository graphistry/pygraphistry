"""Parity + guard tests for L4 single-alias predicate pushdown.

The pushdown pass peels single-alias WHERE conjuncts (and ``searchAny`` ops) off
the post-join filter into ``rows(alias_prefilters=...)``, which pre-filters each
alias frame BEFORE the binding join. This must be a pure optimization: identical
results to the un-pushed chain, on every engine and graph shape. It must also
refuse to push in the unsafe cases (shortestPath scalar bindings, OPTIONAL /
reentry-carried values, internal marker columns) where the peeled filter would be
lost or mis-evaluated.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql.cypher.lowering as _lowering
from graphistry.compute.ast import ASTCall
from graphistry.compute.gfql.cypher.api import cypher_to_gfql
from graphistry.compute.gfql.cypher.row_pushdown import (
    _flatten_and_conjuncts,
    _strip_redundant_parens,
    apply_row_prefilter_pushdown,
)


def _engines() -> List[str]:
    out = ["pandas"]
    try:
        import cudf  # noqa: F401
        out.append("cudf")
    except Exception:
        pass
    return out


ENGINES = _engines()


# --------------------------------------------------------------------------- #
# fixtures: directed / self-loop workload with nullable + textual columns
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def g():
    rng = np.random.default_rng(7)
    n = 400
    ndf = pd.DataFrame({
        "id": np.arange(n),
        "kind": rng.choice(["internal", "user", "svc", "admin"], n),
        "score": np.where(rng.random(n) < 0.15, np.nan, rng.random(n)),
        "flag": rng.random(n) < 0.3,
        "name": [f"node_{['Ember','Wire','Ash','Flux'][i % 4]}_{i}" for i in range(n)],
    })
    m = 1500
    src = rng.integers(0, n, m)
    dst = rng.integers(0, n, m)
    # inject self-loops
    src[:60] = dst[:60]
    edf = pd.DataFrame({
        "src": src,
        "dst": dst,
        "w": rng.integers(0, 8, m),
        "etype": rng.choice(["ref", "WIRE_link", "call", "owns"], m),
    })
    return graphistry.nodes(ndf, "id").edges(edf, "src", "dst")


# queries that MUST trigger a pushdown and must be result-identical pre/post
PUSHDOWN_QUERIES: List[Tuple[str, str]] = [
    ("edge_search", "MATCH (a)-[e]->(b) WHERE searchAny(e, 'WIRE') RETURN e.w AS w ORDER BY w LIMIT 5000"),
    ("node_search", "MATCH (n) WHERE searchAny(n, 'Ember') RETURN n.id AS id ORDER BY id LIMIT 5000"),
    (
        "panel",
        "MATCH (a)-[e]->(b) WHERE a.kind <> 'internal' AND NOT (a.flag = true AND a.score < 0.1) "
        "AND (a.score > 0.25 OR a.score IS NULL) AND e.w > 2 "
        "RETURN a.id AS id, a.score AS score ORDER BY id LIMIT 5000",
    ),
    (
        "undirected_edge_filter",
        "MATCH (a)-[e]-(b) WHERE e.w > 3 AND (a.score > 0.5 OR a.score IS NULL) "
        "RETURN a.id AS id ORDER BY id LIMIT 5000",
    ),
    (
        "combined",
        "MATCH (n) WHERE n.kind <> 'internal' AND (n.score > 0.25 OR n.score IS NULL) "
        "AND EXISTS { (n)--() } AND searchAny(n, 'Ash') RETURN n.id AS id ORDER BY id LIMIT 5000",
    ),
]


def _base_chain(query: str):
    """Compile WITHOUT the pushdown pass (identity-patched), to get the baseline chain."""
    real = _lowering._maybe_pushdown_row_prefilters
    _lowering._maybe_pushdown_row_prefilters = lambda r, **k: r  # type: ignore[assignment]
    try:
        return cypher_to_gfql(query)
    finally:
        _lowering._maybe_pushdown_row_prefilters = real  # type: ignore[assignment]


def _norm(df) -> List[tuple]:
    mod = type(df).__module__
    pdf = df.to_pandas() if ("cudf" in mod or "polars" in mod) else df
    pdf = pdf.reset_index(drop=True)
    rows = []
    for rec in pdf.itertuples(index=False):
        rows.append(tuple(
            None if (isinstance(v, float) and pd.isna(v)) or v is None
            else (round(float(v), 9) if isinstance(v, float) else v)
            for v in rec
        ))
    return sorted(rows, key=repr)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name,query", PUSHDOWN_QUERIES, ids=[q[0] for q in PUSHDOWN_QUERIES])
def test_pushdown_parity(g, engine, name, query):
    pushed = cypher_to_gfql(query)
    base = _base_chain(query)
    # sanity: the pass actually changed the plan for these queries
    assert any(
        isinstance(op, ASTCall) and op.function == "rows" and (op.params.get("alias_prefilters"))
        for op in pushed.chain
    ), f"{name}: expected a pushdown but none was produced"
    r_push = g.gfql(pushed, engine=engine)
    r_base = g.gfql(base, engine=engine)
    assert _norm(r_push._nodes) == _norm(r_base._nodes), f"{name}: pushdown changed results on {engine}"


# --------------------------------------------------------------------------- #
# guard cases: the pass MUST NOT push (would lose / mis-evaluate the filter)
# --------------------------------------------------------------------------- #
def _has_prefilter(chain) -> bool:
    return any(
        isinstance(op, ASTCall) and op.function == "rows" and op.params.get("alias_prefilters")
        for op in chain.chain
    )


@pytest.mark.parametrize("query", [
    # EXISTS marker column conjunct — produced downstream, not on the pre-join frame
    "MATCH (n) WHERE EXISTS { (n)--() } RETURN n.id AS id",
    # multi-alias conjunct — depends on two aliases, cannot pre-filter one frame
    "MATCH (a)-[e]->(b) WHERE a.score > b.score OR a.score IS NULL RETURN a.id AS id",
])
def test_guard_no_unsafe_pushdown(query):
    assert not _has_prefilter(cypher_to_gfql(query))


def test_guard_shortest_path_not_pushed(g):
    # shortestPath scalar bindings route to a builder that ignores alias_prefilters
    q = (
        "MATCH (p1), (p2), p = shortestPath((p1)-[*]-(p2)) "
        "WHERE p1.kind <> 'internal' RETURN p1.id AS a, p2.id AS b, p IS NULL AS noPath LIMIT 100"
    )
    try:
        chain = cypher_to_gfql(q)
    except Exception:
        pytest.skip("shortestPath shape not single-chain in this build")
    assert not _has_prefilter(chain)


# --------------------------------------------------------------------------- #
# pass internals
# --------------------------------------------------------------------------- #
def test_strip_redundant_parens():
    assert _strip_redundant_parens("((x))") == "x"
    assert _strip_redundant_parens("(a) AND (b)") == "(a) AND (b)"  # not fully enclosing
    assert _strip_redundant_parens("(a AND b)") == "a AND b"
    assert _strip_redundant_parens("('a )b')") == "'a )b'"  # paren inside string literal


def test_flatten_and_conjuncts_nested():
    expr = "((((a.kind <> 'internal') AND (NOT ((a.flag = true) AND (a.score < 0.1)))) "
    expr += "AND ((a.score > 0.25) OR (a.score IS NULL))) AND (e.w > 2))"
    parts = _flatten_and_conjuncts(expr)
    assert parts == [
        "a.kind <> 'internal'",
        "NOT ((a.flag = true) AND (a.score < 0.1))",
        "(a.score > 0.25) OR (a.score IS NULL)",
        "e.w > 2",
    ]


def test_flatten_single_conjunct_kept_whole():
    assert _flatten_and_conjuncts("(a.score > 0.25) OR (a.score IS NULL)") == [
        "(a.score > 0.25) OR (a.score IS NULL)"
    ]


def test_pass_is_noop_on_plain_chain():
    from graphistry.compute.chain import Chain
    ch = Chain([ASTCall("rows", {"table": "nodes"}), ASTCall("limit", {"value": 10})])
    assert apply_row_prefilter_pushdown(ch) is ch
