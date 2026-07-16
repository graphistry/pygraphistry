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
    # bounded directed var-length (graph-bench q3 shape) — aggregate forms route
    # through rows(binding_ops) and expand via iterative pair joins
    "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..2]->(b) WHERE a.id = 0 RETURN avg(b.age) AS m",
    "MATCH (a)-[*2..2]->(b) RETURN count(*) AS c",           # exactly-k
    "MATCH (a)-[:F*1..2]->(b) RETURN count(*) AS c",         # typed var-length
    "MATCH (a)-[*1..2]->(b)-[]->(c) RETURN count(*) AS c",   # var-length + fixed hop
]

# Outside the MVP subset: must raise NotImplementedError (honest NIE, no bridge,
# no silent wrong answer).
DEFERRED = [
    "MATCH (a)-[*]->(b) RETURN count(*) AS c",               # unbounded var-length
    "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c",            # undirected var-length
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


def test_binding_rows_projection_pushdown_skips_unused_props():
    """#1711: a query referencing no node properties (count(*)) attaches ZERO
    property columns to the binding table; one referencing only b's property
    attaches only b's — on both pandas and polars. Values stay correct."""
    import graphistry as _g
    from graphistry.compute.ast import n, e_forward, serialize_binding_ops, rows
    bo = serialize_binding_ops([n(name="a"), e_forward(), n(name="b"), e_forward(), n(name="c")])
    for engine in ("pandas", "polars"):
        # attach nothing -> no alias.<prop> columns, only bare a/b/c id columns
        raw = _g.nodes(NODES, "id").edges(EDGES, "s", "d").gfql(
            [rows(binding_ops=bo, attach_prop_aliases=[])], engine=engine
        )._nodes
        cols = list(raw.columns)
        assert not [c for c in cols if str(c).startswith(("a.", "b.", "c."))], f"{engine}: {cols}"
        assert {"a", "b", "c"} <= set(cols)
        # attach only b -> only b.* property columns
        raw_b = _g.nodes(NODES, "id").edges(EDGES, "s", "d").gfql(
            [rows(binding_ops=bo, attach_prop_aliases=["b"])], engine=engine
        )._nodes
        prop_cols = {str(c).split(".", 1)[0] for c in raw_b.columns if "." in str(c)}
        assert prop_cols == {"b"}, f"{engine}: {prop_cols}"


def test_binding_rows_pushdown_end_to_end_parity():
    """#1711: count(*)/top-k queries (pushdown active) stay pandas==polars correct."""
    for q in [
        "MATCH (a)-[]->(b)-[]->(c) RETURN count(*) AS c",
        "MATCH (a)-[e]->(b) RETURN b.id AS id, count(a) AS c ORDER BY c DESC, id LIMIT 3",
    ]:
        _assert_parity(q, order_sensitive="ORDER BY" in q)


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



def test_polars_binding_rows_focused_native_coverage():
    """Focused coverage for narrow native-polars helpers used by binding rows."""
    from graphistry.Engine import Engine
    from graphistry.compute.dataframe.join import (
        connected_inner_join_rows,
        joined_alias_columns,
        joined_hidden_scalar_columns,
    )
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import (
        binding_rows_polars,
        can_order_by_native,
        can_select_native,
        select_extend_polars,
    )

    # Polars dataframe join helper paths used by connected comma joins.
    hidden = pl.DataFrame({
        "a.__gfql_hidden_score": [None, 2],
        "b.__gfql_hidden_score": [1, None],
    })
    hidden_out = joined_hidden_scalar_columns(hidden)
    assert hidden_out.select("__gfql_hidden_score").to_series().to_list() == [1, 2]

    alias_out = joined_alias_columns(pl.DataFrame({"a.id": ["a1"], "b.b": ["b1"]}))
    assert alias_out.select(["a", "b"]).to_dicts() == [{"a": "a1", "b": "b1"}]

    joined = connected_inner_join_rows(
        pl.DataFrame({"a.id": ["a1", "a2"], "a.num": [1, 2]}),
        pl.DataFrame({"a.id": ["a1", "a1", "a3"], "b.id": ["b1", "b2", "b3"]}),
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.POLARS,
    )
    assert joined.select(["a.id", "b.id"]).to_dicts() == [
        {"a.id": "a1", "b.id": "b1"},
        {"a.id": "a1", "b.id": "b2"},
    ]

    # select_extend_polars is the binding-aggregate extension helper, distinct from
    # public with_(extend=True) dispatch.
    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    extended = select_extend_polars(g, [("age2", "age + 1")])
    assert extended is not None
    assert extended._nodes.select(["id", "age2"]).to_dicts()[0] == {"id": 0, "age2": 11}
    assert select_extend_polars(g, [("bad", "unsupported(age)")]) is None

    assert can_select_native([("age2", "age + 1")], ["age"])
    assert can_order_by_native([("age", "desc")], ["age"])


def test_polars_binding_rows_decline_branches_direct():
    """Directly exercise honest-decline branches that are hard to reach via Cypher."""
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    bo = serialize_binding_ops([n(name="a"), e_forward(), n(name="b")])
    no_edges = graphistry.nodes(pl.from_pandas(NODES), "id")
    assert binding_rows_polars(no_edges, bo) is None

    seeded = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    setattr(seeded, "_gfql_start_nodes", pl.DataFrame({"id": [0]}))
    assert binding_rows_polars(seeded, bo) is None

    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), n(name="b")])) is None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a", query="id > 0"), e_forward(), n(name="b")])) is None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), e_forward(source_node_match={"kind": "a"}), n(name="b")])) is None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), e_forward(label_seeds=True), n(name="b")])) is None

    # Polars cannot implicitly unify mismatched join-key dtypes; decline on collect SchemaError.
    bad_edges = pl.DataFrame({"s": ["0"], "d": ["1"]})
    bad = graphistry.nodes(pl.from_pandas(NODES), "id").edges(bad_edges, "s", "d")
    assert binding_rows_polars(bad, bo) is None
