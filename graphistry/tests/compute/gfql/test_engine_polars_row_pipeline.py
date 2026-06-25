"""Differential parity: native polars cypher row pipeline == pandas.

Phase 2 of the GFQL polars engine. Covers the boundary-call dispatch
(``chain_polars`` splitting traversal from trailing ``call()`` ops) plus the
native polars frame ops (rows / limit / skip / distinct / drop_cols) and the
host-bridged result projection. Pandas is the oracle: for every supported
cypher query the polars engine must return an identical result table (and a
polars-typed frame). Not-yet-ported ops must raise NotImplementedError, never
silently diverge. See plans/gfql-polars-engine.
"""
import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")


NODES = pd.DataFrame({
    "id": [0, 1, 2, 3, 4, 5],
    "val": [10, 20, 30, 40, 50, 60],
    "kind": ["a", "b", "a", "b", "a", "c"],
    "name": ["alice", "bob", "carol", "dave", "erin", "frank"],
})
EDGES = pd.DataFrame({
    "s": [0, 1, 2, 3, 4, 0, 2],
    "d": [1, 2, 3, 4, 5, 2, 4],
    "w": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
})
BASE = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


def _to_pandas(df):
    if df is not None and "polars" in type(df).__module__:
        return df.to_pandas()
    return df


def _assert_parity(query, *, order_sensitive=True):
    """Polars result table equals the pandas oracle (and is polars-typed)."""
    rpd = BASE.gfql(query, engine="pandas")._nodes
    rpl = BASE.gfql(query, engine="polars")._nodes
    assert "polars" in type(rpl).__module__, f"expected polars frame for {query!r}"
    a = _to_pandas(rpd).reset_index(drop=True)
    b = _to_pandas(rpl).reset_index(drop=True)
    assert list(a.columns) == list(b.columns), f"columns differ for {query!r}: {list(a.columns)} vs {list(b.columns)}"
    assert len(a) == len(b), f"row count differs for {query!r}: {len(a)} vs {len(b)}"
    if order_sensitive:
        pd.testing.assert_frame_equal(a, b, check_dtype=False)
    else:
        a_sorted = a.sort_values(list(a.columns)).reset_index(drop=True)
        b_sorted = b.sort_values(list(b.columns)).reset_index(drop=True)
        pd.testing.assert_frame_equal(a_sorted, b_sorted, check_dtype=False)


SUPPORTED = [
    # whole-entity RETURN (pure projection, no row-pipeline op)
    "MATCH (n) RETURN n",
    # limit / skip / skip+limit (frame ops)
    "MATCH (n) RETURN n LIMIT 3",
    "MATCH (n) RETURN n LIMIT 0",
    "MATCH (n) RETURN n LIMIT 100",
    "MATCH (n) RETURN n SKIP 2",
    "MATCH (n) RETURN n SKIP 4",
    "MATCH (n) RETURN n SKIP 100",
    "MATCH (n) RETURN n SKIP 1 LIMIT 2",
    "MATCH (n) RETURN n SKIP 2 LIMIT 3",
    # whole-row distinct
    "MATCH (n) RETURN DISTINCT n",
    # single-entity WHERE (folds into the node matcher, handled by PR1 traversal)
    "MATCH (n) WHERE n.val > 25 RETURN n",
    "MATCH (n) WHERE n.val >= 30 RETURN n",
    'MATCH (n) WHERE n.kind = "a" RETURN n',
    "MATCH (n) WHERE n.val < 30 RETURN n LIMIT 1",
    # relationship patterns into a row return
    "MATCH (n)-[e]->(m) RETURN m",
    "MATCH (a)-[e]->(b) WHERE a.val < 30 RETURN b",
    "MATCH (a)-[e]->(b) RETURN b LIMIT 2",
    "MATCH (a)-[e]->(b) RETURN DISTINCT b",
    # multi-column projection handled by the (host-bridged) result projection
    "MATCH (n) RETURN n, n.val",
    "MATCH (n) RETURN n, n.val, n.kind",
]


@pytest.mark.parametrize("query", SUPPORTED)
def test_polars_row_pipeline_parity(query):
    _assert_parity(query)


# Ops that route into the pandas cypher expression engine; must defer, not
# silently diverge. Each compiles to a row-pipeline call (select / order_by /
# where_rows / group_by / unwind) or a multi-entity binding_ops rows().
DEFERRED = [
    "MATCH (n) RETURN n.val",                              # select
    "MATCH (n) RETURN n.val, n.kind",                      # select
    "MATCH (n) RETURN DISTINCT n.kind",                    # select + distinct
    "MATCH (n)-[e]->(m) WHERE n.val < m.val RETURN n, m",  # cross-entity where_rows + binding_ops
    "MATCH (n) RETURN count(n) AS c",                      # group_by / aggregation
]


@pytest.mark.parametrize("query", DEFERRED)
def test_polars_row_pipeline_deferred_raises(query):
    with pytest.raises(NotImplementedError):
        BASE.gfql(query, engine="polars")


def test_polars_frame_op_limit_matches_slice():
    """limit/skip operate on a polars active table without index artifacts."""
    g = BASE.gfql("MATCH (n) RETURN n LIMIT 4", engine="polars")
    assert g._nodes.height == 4
    assert "polars" in type(g._nodes).__module__


def test_polars_distinct_preserves_first_order():
    """Whole-row distinct keeps first occurrence in order (== pandas)."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["a", "a", "b", "b"]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    rpd = _to_pandas(g.gfql("MATCH (n) RETURN DISTINCT n", engine="pandas")._nodes)
    rpl = _to_pandas(g.gfql("MATCH (n) RETURN DISTINCT n", engine="polars")._nodes)
    pd.testing.assert_frame_equal(
        rpd.reset_index(drop=True), rpl.reset_index(drop=True), check_dtype=False
    )


def test_polars_empty_result_shape():
    """A LIMIT 0 / over-skip empties to 0 rows but keeps the projected schema."""
    g = BASE.gfql("MATCH (n) RETURN n SKIP 1000", engine="polars")
    assert g._nodes.height == 0
    assert list(g._nodes.columns) == ["n"]


# Direct frame-op coverage: exercises each native polars branch on a real
# polars-framed graph, independent of which cypher shapes happen to compile to
# which ops. Keeps the engine-polymorphic frame_ops layer pinned.
def _polars_graph():
    from graphistry.Engine import Engine, df_to_engine
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "k": ["a", "a", "b", "b"], "v": [1, 2, 3, 4]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 2]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    return g.nodes(df_to_engine(g._nodes, Engine.POLARS), g._node).edges(
        df_to_engine(g._edges, Engine.POLARS), g._source, g._destination
    )


def _adapter(g):
    from graphistry.compute.gfql.row.pipeline import _RowPipelineAdapter
    return _RowPipelineAdapter(g)


def test_frame_ops_polars_limit_skip():
    from graphistry.compute.gfql.row import frame_ops as fo
    g = _polars_graph()
    assert fo.limit(_adapter(g), 2)._nodes.height == 2
    assert fo.skip(_adapter(g), 1)._nodes.height == 3
    assert "polars" in type(fo.limit(_adapter(g), 2)._nodes).__module__


def test_frame_ops_polars_distinct_drop_cols():
    from graphistry.compute.gfql.row import frame_ops as fo
    g = _polars_graph()
    assert fo.distinct(_adapter(g))._nodes.height == 4
    cols = list(fo.drop_cols(_adapter(g), ["k"])._nodes.columns)
    assert "k" not in cols and "id" in cols and "v" in cols


def test_frame_ops_polars_rows_and_empty_frame():
    from graphistry.compute.gfql.row import frame_ops as fo
    g = _polars_graph()
    # rows() with no source returns the full active table (polars-typed)
    rows_out = fo.rows(_adapter(g), table="nodes")._nodes
    assert "polars" in type(rows_out).__module__ and rows_out.height == 4
    # empty_frame with explicit columns yields a 0-row polars frame with those cols
    ef = fo.empty_frame(_adapter(g), template_df=g._nodes, columns=["x", "y"])
    assert "polars" in type(ef).__module__
    assert list(ef.columns) == ["x", "y"] and ef.height == 0


def test_polars_chain_interior_call_mix_raises():
    """call() between traversals is rejected (boundary-only), like the pandas path."""
    from graphistry.compute.ast import call, n, e_forward
    from graphistry.compute.exceptions import GFQLValidationError
    with pytest.raises(GFQLValidationError):
        BASE.chain([n(), call("limit", {"value": 2}), e_forward(), n()], engine="polars")


def test_polars_chain_prefix_call_before_traversal_defers():
    """Leading call() before a traversal is deferred on polars (not a cypher shape)."""
    from graphistry.compute.ast import call, n
    with pytest.raises(NotImplementedError):
        BASE.chain([call("limit", {"value": 3}), n()], engine="polars")


def test_polars_chain_pure_call_no_traversal():
    """A chain of only call() ops (no traversal) runs the calls on polars."""
    from graphistry.compute.ast import call
    g = BASE.chain([call("limit", {"value": 2})], engine="polars")
    assert "polars" in type(g._nodes).__module__
    assert g._nodes.height == 2


def test_frame_ops_polars_rows_empty_table():
    """rows() materializes an empty active table without index artifacts."""
    from graphistry.Engine import Engine, df_to_engine
    from graphistry.compute.gfql.row import frame_ops as fo
    nodes = pd.DataFrame({"id": [0, 1], "v": [1, 2]})
    edges = pd.DataFrame({"s": [0], "d": [1]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    g = g.nodes(df_to_engine(g._nodes, Engine.POLARS), g._node).edges(
        df_to_engine(g._edges, Engine.POLARS), g._source, g._destination
    )
    empty = g.nodes(g._nodes.clear(), g._node)
    out = fo.rows(_adapter(empty), table="nodes")._nodes
    assert "polars" in type(out).__module__ and out.height == 0
