"""Differential parity: native polars ``rows(binding_ops)`` == pandas (#1709).

Fixed-length connected multi-alias binding tables — the ``rows(binding_ops=...)``
op emitted by Cypher multi-alias lowering (graph-benchmark q1/q2 1-hop top-k,
q8/q9 2-hop counts, seeded expansions). Pandas is the oracle: every supported
query must return an identical result table, polars-typed (no pandas bridge).
Out-of-subset shapes must raise NotImplementedError, never silently diverge.
See plans/gfql-1709-polars-binding-ops.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, rows, serialize_binding_ops

pl = pytest.importorskip("polars")


NODES = pd.DataFrame({
    "id": [0, 1, 2, 3, 4],
    "age": [10, 20, 30, 40, 50],
    "kind": ["a", "b", "a", "b", "a"],
})
EDGES = pd.DataFrame({
    "s": [0, 1, 2, 0, 3, 1],
    "d": [1, 2, 3, 2, 4, 4],
    "type": ["F", "F", "F", "G", "F", "F"],
    "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
})
BASE = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _to_pandas(df):
    if df is not None and "polars" in type(df).__module__:
        return df.to_pandas()
    return df


def _assert_parity(query, *, order_sensitive=False):
    rpd = BASE.gfql(query, engine="pandas")._nodes
    rpl = BASE.gfql(query, engine="polars")._nodes
    assert "polars" in type(rpl).__module__, f"expected polars frame for {query!r}"
    a = _to_pandas(rpd).reset_index(drop=True)
    b = _to_pandas(rpl).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), f"columns differ for {query!r}: {list(a.columns)} vs {list(b.columns)}"
    if not order_sensitive and len(a):
        a = a.sort_values(list(a.columns)).reset_index(drop=True)
        b = b.sort_values(list(b.columns)).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


# End-to-end Cypher over binding tables: the graph-benchmark shapes + variations.
SUPPORTED = [
    # 1-hop multi-alias property projections
    "MATCH (a)-[e]->(b) RETURN a.id, b.id, b.age",
    "MATCH (a)-[e]->(b) RETURN a.id, e.w, b.id",           # edge-alias payload
    "MATCH (a)-[e]->(b) RETURN DISTINCT a.id, b.id",
    # graph-bench q1 shape: top-k in-degree via count(<non-active alias>) (#1708)
    "MATCH (a)-[e]->(b) RETURN b.id AS id, count(a) AS c ORDER BY c DESC, id LIMIT 3",
    # grouped aggregates over the bindings table
    "MATCH (a)-[e]->(b) RETURN b.kind, count(*) AS c ORDER BY c DESC, b.kind",
    # graph-bench q8/q9 shapes: fixed-length multi-hop counts (+ WHERE)
    "MATCH (a)-[]->(b)-[]->(c) RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[]->(c) WHERE b.age > 15 AND c.age > 25 RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[]->(c)-[]->(d2) RETURN count(*) AS c",
    # directions
    "MATCH (a)<-[e]-(b) RETURN a.id, b.id",
    "MATCH (a)-[e]-(b) RETURN a.id, b.id",                  # undirected, both orients
    # filters: node inline-map, edge type
    "MATCH (a {kind: 'a'})-[e]->(b) RETURN a.id, b.id",
    "MATCH (a)-[e:F]->(b) RETURN a.id, b.id",
    "MATCH (a)-[e:F]->(b {kind: 'b'}) RETURN a.id, b.id, e.w",
    # single-alias WHERE over the bindings row table
    "MATCH (a)-[e]->(b) WHERE b.age >= 30 RETURN a.id, b.id",
    # empty result keeps schema
    "MATCH (a)-[e]->(b) WHERE a.age > 999 RETURN a.id, b.id",
]

# Outside the MVP subset: must raise NotImplementedError (honest NIE, no bridge,
# no silent wrong answer).
DEFERRED = [
    "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c",           # var-length multihop
    "MATCH (a)-[e]->(b) WHERE a.age < b.age RETURN a.id",    # cross-alias same-path WHERE
]


@pytest.mark.parametrize("query", SUPPORTED)
def test_polars_binding_rows_parity(query):
    _assert_parity(query, order_sensitive="ORDER BY" in query)


@pytest.mark.parametrize("query", DEFERRED)
def test_polars_binding_rows_deferred_raises(query):
    with pytest.raises(NotImplementedError):
        BASE.gfql(query, engine="polars")


def test_polars_binding_rows_raw_table_meaningful_cols():
    """Raw rows(binding_ops): polars carries the meaningful schema (bare alias id
    cols, alias.{col} node props, edge_alias.{col} payload) with values equal to
    pandas. The pandas frame's extra join-residue columns (raw node-id, marker
    and suffix cols) are internal and intentionally not replicated."""
    bo = serialize_binding_ops([n(name="a"), e_forward(name="e"), n(name="b")])
    rpd = BASE.gfql([rows(binding_ops=bo)], engine="pandas")._nodes
    rpl = BASE.gfql([rows(binding_ops=bo)], engine="polars")._nodes
    assert "polars" in type(rpl).__module__
    expected_cols = {"a", "b", "e.w", "e.type", "a.id", "a.age", "a.kind", "b.id", "b.age", "b.kind"}
    assert expected_cols <= set(rpl.columns)
    assert set(rpl.columns) <= set(rpd.columns)  # no columns pandas lacks
    key = ["a", "b", "e.w"]
    a = rpd[list(rpl.columns)].sort_values(key).reset_index(drop=True)
    b = rpl.to_pandas().sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_polars_binding_rows_multiplicity():
    """One row per matched path: parallel edges yield distinct binding rows."""
    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0, 0], "d": [1, 1], "w": [1.0, 2.0]})  # parallel
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    rpl = g.gfql("MATCH (a)-[e]->(b) RETURN a.id, e.w, b.id", engine="polars")._nodes
    assert rpl.height == 2
    rpd = g.gfql("MATCH (a)-[e]->(b) RETURN a.id, e.w, b.id", engine="pandas")._nodes
    pd.testing.assert_frame_equal(
        rpd.sort_values("e.w").reset_index(drop=True),
        rpl.to_pandas().sort_values("e.w").reset_index(drop=True),
        check_dtype=False,
    )


def test_polars_binding_rows_undirected_self_loop():
    """Undirected orientation concats both directions without dedup (== pandas)."""
    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 1]})  # includes self-loop 1->1
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a)-[e]-(b) RETURN a.id, b.id"
    rpd = g.gfql(q, engine="pandas")._nodes
    rpl = g.gfql(q, engine="polars")._nodes.to_pandas()
    key = ["a.id", "b.id"]
    pd.testing.assert_frame_equal(
        rpd.sort_values(key).reset_index(drop=True),
        rpl.sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )
