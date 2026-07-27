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
    # zero-hop window: the seed binds to itself, INCLUDING seeds with no outgoing
    # edge (which the traversal prunes away — hence the base-graph rebuild).
    "MATCH (a)-[*0..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*0..2]->(b) RETURN a.id AS ai, b.id AS bi ORDER BY ai, bi",
    # UNBOUNDED directed fixed point (#1709, LDBC IS6 `REPLY_OF*0..` ancestor walk)
    "MATCH (a)-[*]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*0..]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*]->(b) RETURN a.id AS ai, b.id AS bi ORDER BY ai, bi",
    "MATCH (a)<-[*]-(b) RETURN count(*) AS c",               # reverse direction
    "MATCH (a)-[:F*]->(b) RETURN count(*) AS c",             # typed unbounded
    "MATCH (a)-[*]->(b) WHERE b.age > 25 RETURN a.id AS ai, b.id AS bi ORDER BY ai, bi",
    "MATCH (a)-[*]->(b)-[]->(c) RETURN count(*) AS c",       # unbounded + fixed hop
    "MATCH (a {kind: 'a'})-[*0..]->(b) RETURN count(*) AS c",  # filtered seed alias
    # bounded UNDIRECTED var-length, min_hops == 1 (LDBC IC11/IC6 `-[*1..k]-` shape):
    # doubled-pair join + immediate-backtrack avoidance, exact pandas multiplicity.
    "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..3]-(b) RETURN count(*) AS c",
    "MATCH (a)-[:F*1..2]-(b) RETURN count(*) AS c",          # typed undirected var-length
    # the residual IC11 clause: cross-alias node inequality over an undirected
    # var-length bindings table (previously NIE'd on the undirected `rows` op).
    "MATCH (a)-[*1..2]-(b) WHERE NOT a = b RETURN b.id AS id ORDER BY id",
    "MATCH (a)-[*1..2]-(b) WHERE a <> b RETURN a.id AS ai, b.id AS bi ORDER BY ai, bi",
    # node-only cartesian: disconnected multi-source MATCH (#1273). <=3 named aliases.
    "MATCH (a), (b) RETURN a.id AS ai, b.id AS bi ORDER BY ai, bi",
    "MATCH (a {kind: 'a'}), (b {kind: 'b'}) RETURN a.id AS ai, b.id AS bi",
    "MATCH (a {kind: 'a'}), (b) RETURN a.id AS ai, a.age AS aa, b.id AS bi",
    "MATCH (a {kind: 'a'}), (b {kind: 'b'}) WHERE a.age > 20 RETURN a.id AS ai, b.id AS bi",
    "MATCH (a), (b), (c {kind: 'a'}) RETURN a.id AS ai, b.id AS bi, c.id AS ci ORDER BY ai, bi, ci",
    "MATCH (a {kind: 'z'}), (b) RETURN a.id AS ai, b.id AS bi",   # empty (no kind=z) keeps schema
]

# Outside the MVP subset: must raise NotImplementedError (honest NIE, no bridge,
# no silent wrong answer).
DEFERRED = [
    # UNDIRECTED unbounded: would need both the min_hops == 1 multiplicity
    # reconstruction and backtrack-aware termination (pandas rejects it outright).
    "MATCH (a)-[*]-(b) RETURN count(*) AS c",
    # undirected var-length is native ONLY for min_hops == 1 (see SUPPORTED). Other
    # windows still DECLINE: pandas' step_pairs come from the var-length hop whose
    # backward-pruning / zero-hop handling changes edge multiplicity in a way the
    # raw-edge reconstruction only reproduces for min_hops == 1.
    "MATCH (a)-[*0..2]-(b) RETURN count(*) AS c",            # undirected, min_hops 0
    "MATCH (a)-[*2..3]-(b) RETURN count(*) AS c",            # undirected, min_hops 2
    "MATCH (a)-[*2..2]-(b) RETURN count(*) AS c",            # undirected, exactly-2
    "MATCH (a)-[e]->(b) WHERE a.age < b.age RETURN a.id",    # cross-alias same-path WHERE
    # cartesian outside the pandas-reliable subset (#1273): pandas itself errors or
    # is fragile here, so polars declines rather than diverge.
    "MATCH (a), (b), (c), (d) RETURN a.id, b.id, c.id, d.id",  # >3 named aliases
    "MATCH (a), (b), () RETURN a.id, b.id",                    # anonymous companion
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


def test_polars_cartesian_binding_rows_raw_meaningful_cols():
    """Raw node-only cartesian rows(binding_ops): polars carries the same meaningful
    per-alias schema as pandas (bare ``alias`` id, ``alias.id``, ``alias.<prop>``,
    plus the leaked named-op flag ``alias.alias = True``), values equal to pandas."""
    bo = serialize_binding_ops([n(name="a"), n(name="b")])
    rpd = BASE.gfql([rows(binding_ops=bo)], engine="pandas")._nodes
    rpl = BASE.gfql([rows(binding_ops=bo)], engine="polars")._nodes
    assert "polars" in type(rpl).__module__
    expected = {"a", "a.id", "a.age", "a.kind", "a.a", "b", "b.id", "b.age", "b.kind", "b.b"}
    assert expected <= set(rpl.columns)
    assert set(rpl.columns) <= set(rpd.columns)  # no columns pandas lacks
    key = ["a", "b"]
    a = rpd[list(rpl.columns)].sort_values(key).reset_index(drop=True)
    b = rpl.to_pandas().sort_values(key).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b, check_dtype=False)


def test_polars_cartesian_alias_name_collides_with_property():
    """A node property named the same as a MATCH alias is shadowed by the leaked
    named-op flag (``alias.alias = True``) on BOTH engines — polars mirrors the
    pandas quirk exactly rather than surfacing the real property value."""
    nodes = pd.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "a"], "a": [10, 20, 30], "b": [1, 2, 3]})
    g = graphistry.nodes(nodes, "id").edges(pd.DataFrame({"s": [0], "d": [1]}), "s", "d")
    q = "MATCH (a {kind: 'a'}), (b {kind: 'b'}) RETURN a.id AS ai, a.a AS aa, b.id AS bi, b.b AS bb"
    rpd = g.gfql(q, engine="pandas")._nodes.reset_index(drop=True)
    rpl = g.gfql(q, engine="polars")._nodes.to_pandas().reset_index(drop=True)
    assert list(rpd["aa"]) == [True, True] and list(rpd["bb"]) == [True, True]  # flag, not 10/30
    pd.testing.assert_frame_equal(
        rpd.sort_values(["ai", "bi"]).reset_index(drop=True),
        rpl[rpd.columns.tolist()].sort_values(["ai", "bi"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_polars_cartesian_multiplicity_three_aliases():
    """Three-alias cross product has |a|*|b|*|c| rows in left-major order on polars,
    identical to pandas."""
    q = "MATCH (a {kind: 'a'}), (b {kind: 'b'}), (c {kind: 'a'}) RETURN a.id AS ai, b.id AS bi, c.id AS ci"
    rpd = BASE.gfql(q, engine="pandas")._nodes.reset_index(drop=True)
    rpl = BASE.gfql(q, engine="polars")._nodes.to_pandas().reset_index(drop=True)
    assert len(rpl) == 3 * 2 * 3  # kind a = {0,2,4}, kind b = {1,3}
    pd.testing.assert_frame_equal(rpd, rpl[rpd.columns.tolist()], check_dtype=False)  # order-exact


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



def _undirected_chain_graph():
    # chain 0-1-2-3 (directed edges, read undirected by the query)
    nodes = pd.DataFrame({"id": [0, 1, 2, 3]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def test_polars_undirected_varlen_min1_backtrack_and_multiplicity_pins():
    """Undirected `-[*1..k]-` binding table (min_hops == 1) — pin the EXACT pandas
    oracle semantics, not just parity, so a future drift on either engine is caught:

    * immediate-backtrack avoidance (0->1->0 excluded, so `WHERE NOT a = b` on a chain
      never re-reaches the start via a single edge),
    * pandas' edge multiplicity (each non-loop edge contributes each directed
      orientation TWICE), so a length-1 pair appears x2 and a length-2 pair appears x4.
    """
    g = _undirected_chain_graph()
    # length-1 pairs appear x2, length-2 pairs appear x4 (pandas step_pairs doubling).
    q = "MATCH (a)-[*1..2]-(b) WHERE NOT a = b RETURN a.id AS ai, b.id AS bi"
    rpl = g.gfql(q, engine="polars")._nodes.to_pandas()
    counts = rpl.groupby(["ai", "bi"]).size().to_dict()
    # 1-hop neighbours (both orientations), each doubled
    assert counts[(0, 1)] == 2 and counts[(1, 0)] == 2
    assert counts[(1, 2)] == 2 and counts[(2, 1)] == 2
    assert counts[(2, 3)] == 2 and counts[(3, 2)] == 2
    # 2-hop reaches (backtrack-free), each x4
    assert counts[(0, 2)] == 4 and counts[(2, 0)] == 4
    assert counts[(1, 3)] == 4 and counts[(3, 1)] == 4
    # backtrack pairs (a==b via 0->1->0) are excluded -> no (0,0)/(1,1)/... rows here
    assert (0, 0) not in counts and (1, 1) not in counts
    # and it exactly matches the pandas oracle
    _assert_parity(q)


def test_polars_undirected_varlen_min1_self_loop_multiplicity():
    """Self-loops contribute (u,u) x2 only (NOT x4 like non-loops) — mirrors pandas,
    which sources a single self-loop row from the var-length hop before orienting."""
    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 1]})  # edge 0->1 + self-loop 1->1
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a)-[*1..2]-(b) RETURN a.id AS ai, b.id AS bi"
    rpd = g.gfql(q, engine="pandas")._nodes
    rpl = g.gfql(q, engine="polars")._nodes
    assert "polars" in type(rpl).__module__
    key = ["ai", "bi"]
    pd.testing.assert_frame_equal(
        rpd.sort_values(key).reset_index(drop=True),
        rpl.to_pandas().sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_polars_undirected_varlen_min1_parity_string_ids_and_parallel():
    """String node ids + parallel/antiparallel edges: the `__prev__` backtrack marker
    is dtype-matched to the id column and multiplicity stays pandas-exact."""
    nodes = pd.DataFrame({"id": ["x", "y", "z"]})
    edges = pd.DataFrame({"s": ["x", "y", "y"], "d": ["y", "x", "z"]})  # antiparallel + extra
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    for q in [
        "MATCH (a)-[*1..2]-(b) RETURN a.id AS ai, b.id AS bi",
        "MATCH (a)-[*1..3]-(b) WHERE a <> b RETURN a.id AS ai, b.id AS bi",
    ]:
        rpd = g.gfql(q, engine="pandas")._nodes
        rpl = g.gfql(q, engine="polars")._nodes
        assert "polars" in type(rpl).__module__
        key = ["ai", "bi"]
        pd.testing.assert_frame_equal(
            rpd.sort_values(key).reset_index(drop=True),
            rpl.to_pandas().sort_values(key).reset_index(drop=True),
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

    # WITH->MATCH re-entry seed (#1273): a UNIQUE carried-id seed now lowers natively — the
    # first alias is semi-joined to the seed ids (native twin of the pandas wavefront).
    seeded = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    setattr(seeded, "_gfql_start_nodes", pl.DataFrame({"id": [0]}))
    seeded_out = binding_rows_polars(seeded, bo)
    assert seeded_out is not None
    # seed {0} -[F]-> b: EDGES forward from 0 with type F -> b in {1}; the bindings 'a' column
    # is pinned to the seed and every row's 'a' id is 0.
    seeded_rows = seeded_out._nodes
    assert set(seeded_rows["a"].to_list()) == {0}
    # A DUPLICATE-id seed still declines (semi-join would drop path multiplicity vs pandas).
    dup = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    setattr(dup, "_gfql_start_nodes", pl.DataFrame({"id": [0, 0]}))
    assert binding_rows_polars(dup, bo) is None

    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    # node-only cartesian (#1273) is now natively supported for <=3 named aliases;
    # it declines only outside the pandas-reliable subset:
    #  - anonymous node op (pandas raises a spurious schema error on empty results)
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), n()])) is None
    #  - >3 named aliases (pandas' bare-id merge residue collides on the 4th frame)
    assert binding_rows_polars(
        g, serialize_binding_ops([n(name="a"), n(name="b"), n(name="c"), n(name="d")])
    ) is None
    #  - an alias named exactly like the bound node-id column ("id"): pandas' leaked
    #    flag column would overwrite the id column — no sane shared semantics, so decline
    assert binding_rows_polars(g, serialize_binding_ops([n(name="id"), n(name="b")])) is None
    # ...but 2-3 named aliases now lower natively (returns a row table, not None)
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), n(name="b")])) is not None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a", query="id > 0"), e_forward(), n(name="b")])) is None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), e_forward(source_node_match={"kind": "a"}), n(name="b")])) is None
    assert binding_rows_polars(g, serialize_binding_ops([n(name="a"), e_forward(label_seeds=True), n(name="b")])) is None

    # Polars cannot implicitly unify mismatched join-key dtypes; decline on collect SchemaError.
    bad_edges = pl.DataFrame({"s": ["0"], "d": ["1"]})
    bad = graphistry.nodes(pl.from_pandas(NODES), "id").edges(bad_edges, "s", "d")
    assert binding_rows_polars(bad, bo) is None


# ---------------------------------------------------------------------------
# Unbounded DIRECTED variable-length binding rows (#1709): `-[*]->` / `-[*0..]->`.
# Motivating query: LDBC SNB interactive-short-6 walks `REPLY_OF*0..` from a
# message to its root post, then out to the forum and moderator.
# ---------------------------------------------------------------------------


def _reply_chain_graph():
    """IS6 shape: message 1 -REPLY_OF-> 2 -REPLY_OF-> post 3, contained by forum 10,
    moderated by person 30. Message 4 is its own root post (zero-hop case)."""
    nodes = pd.DataFrame({
        "id": [1, 2, 3, 4, 10, 30],
        "label__Message": [True, True, True, True, False, False],
        "label__Post": [False, False, True, True, False, False],
        "label__Forum": [False, False, False, False, True, False],
        "label__Person": [False, False, False, False, False, True],
        "title": ["m1", "m2", "p3", "p4", "f10", "mod"],
    })
    edges = pd.DataFrame({
        "s": [1, 2, 10, 10, 10],
        "d": [2, 3, 3, 4, 30],
        "type": ["REPLY_OF", "REPLY_OF", "CONTAINER_OF", "CONTAINER_OF", "HAS_MODERATOR"],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("seed", [1, 4])
def test_polars_unbounded_varlen_is6_ancestor_walk_parity(seed):
    """The LDBC IS6 pattern end-to-end. seed=1 exercises the 2-hop ancestor walk,
    seed=4 the ZERO-hop case (the message already is the post) — the case that
    forced the bindings rebuild onto the pre-chain base graph."""
    from graphistry.compute.ast import e_reverse, n as node, rows as rows_call, select as select_call

    g = _reply_chain_graph()
    query = [
        node({"id": seed, "label__Message": True}, name="message"),
        e_forward({"type": "REPLY_OF"}, min_hops=0, to_fixed_point=True),
        node({"label__Post": True}, name="post"),
        e_reverse({"type": "CONTAINER_OF"}),
        node({"label__Forum": True}, name="forum"),
        e_forward({"type": "HAS_MODERATOR"}),
        node({"label__Person": True}, name="moderator"),
        rows_call(),
        select_call([("forumId", "forum.id"), ("moderatorId", "moderator.id")]),
    ]
    rpd = g.gfql(query, engine="pandas")._nodes
    rpl = g.gfql(query, engine="polars")._nodes
    assert "polars" in type(rpl).__module__
    assert len(rpd) == 1  # the pattern really does match (guards a vacuous pass)
    pd.testing.assert_frame_equal(
        rpd.reset_index(drop=True), rpl.to_pandas().reset_index(drop=True), check_dtype=False
    )


def test_polars_unbounded_varlen_zero_hop_seed_without_outgoing_edge():
    """`-[*0..]->` binds a seed that has NO matching outgoing edge. The chain
    traversal prunes such a node away, so a bindings rebuild from the chain OUTPUT
    would silently return nothing; pandas (and now polars) rebuild from the
    pre-chain graph and keep the zero-hop row."""
    from graphistry.compute.ast import n as node, rows as rows_call, select as select_call

    nodes = pd.DataFrame({"id": [0, 1, 2], "kind": ["B", "B", "A"]})
    edges = pd.DataFrame({"s": [0, 0], "d": [2, 1], "type": ["X", "R"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = [
        node({"kind": "A"}, name="a"),
        e_forward({"type": "R"}, min_hops=0, to_fixed_point=True),
        node(name="b"),
        rows_call(),
        select_call([("ai", "a.id"), ("bi", "b.id")]),
    ]
    rpd = g.gfql(query, engine="pandas")._nodes
    rpl = g.gfql(query, engine="polars")._nodes.to_pandas()
    assert rpd.to_dict("records") == [{"ai": 2, "bi": 2}]
    pd.testing.assert_frame_equal(
        rpd.reset_index(drop=True), rpl.reset_index(drop=True), check_dtype=False
    )


def test_polars_unbounded_varlen_path_multiplicity_matches_pandas():
    """One row per distinct edge SEQUENCE: parallel edges multiply per hop, and a
    diamond contributes one row per path — not per reachable endpoint."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3]})
    edges = pd.DataFrame({
        "s": [0, 0, 1, 2],
        "d": [1, 2, 3, 3],   # diamond 0->{1,2}->3
        "type": ["R"] * 4,
    })
    edges = pd.concat([edges, pd.DataFrame({"s": [0], "d": [1], "type": ["R"]})])  # parallel 0->1
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a)-[*]->(b) RETURN a.id AS ai, b.id AS bi"
    rpd = g.gfql(q, engine="pandas")._nodes
    rpl = g.gfql(q, engine="polars")._nodes.to_pandas()
    # 0->3 has three distinct edge sequences (two via the parallel 0->1, one via 2)
    assert sorted(rpl[(rpl.ai == 0) & (rpl.bi == 3)].bi.tolist()) == [3, 3, 3]
    key = ["ai", "bi"]
    pd.testing.assert_frame_equal(
        rpd.sort_values(key).reset_index(drop=True),
        rpl.sort_values(key).reset_index(drop=True),
        check_dtype=False,
    )


def test_polars_unbounded_varlen_cycle_raises_same_error_as_pandas():
    """A cycle reachable from the seed means infinitely many paths. Pandas raises
    E108 ('require terminating variable-length segments'); polars must raise the
    SAME diagnosis, never truncate to a subtly different answer.

    Asserts the CLASS and `.code` are engine-INDEPENDENT, not just that polars carries
    some code of its own. An earlier version pinned `errs['polars'].code == E108` and so
    passed while the two engines actually disagreed — pandas re-wraps the kernel error as
    GFQLTypeError(E303) via `execute_call`, polars ran the native kernel before that
    wrapper and leaked the raw GFQLValidationError(E108). Callers switch on `.code`, so
    a per-engine code is a real divergence; the E108 text survives inside the message."""
    from graphistry.compute.exceptions import GFQLValidationError

    nodes = pd.DataFrame({"id": [0, 1, 2]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0], "type": ["R"] * 3})  # 3-cycle
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a)-[*]->(b) RETURN count(*) AS c"
    errs = {}
    for engine in ("pandas", "polars"):
        with pytest.raises(GFQLValidationError) as exc:
            g.gfql(q, engine=engine)
        errs[engine] = exc.value
    assert type(errs["polars"]) is type(errs["pandas"]), (
        f"exception class differs by engine: {type(errs['polars'])} vs {type(errs['pandas'])}")
    assert errs["polars"].code == errs["pandas"].code, (
        f"error code differs by engine: {errs['polars'].code} vs {errs['pandas'].code}")
    for engine in ("pandas", "polars"):
        assert (
            "Cypher multi-alias row bindings currently require terminating "
            "variable-length segments"
        ) in str(errs[engine])


def test_polars_unbounded_varlen_cycle_bound_is_the_reachable_set_not_the_graph():
    """The exhaustion probe bounds itself by the REACHABLE node count, not the graph's.

    Both directions of that bound are failure modes, so pin both:
      * too TIGHT and a long acyclic walk is mistaken for a cycle (false E108),
      * too LOOSE and a reachable cycle is never detected (wrong answer, or a hang).
    The chain is deliberately longer than the cyclic component so a bound derived from
    the cycle's own size would cut the acyclic walk short.
    """
    from graphistry.compute.exceptions import GFQLValidationError

    # 0 -> 1 -> ... -> 60, acyclic, walk length 60.
    chain_edges = pd.DataFrame({"s": list(range(60)), "d": list(range(1, 61))})
    chain_nodes = pd.DataFrame({"id": list(range(61))})
    g_pd = graphistry.nodes(chain_nodes, "id").edges(chain_edges, "s", "d")
    g_pl = graphistry.nodes(pl.from_pandas(chain_nodes), "id").edges(
        pl.from_pandas(chain_edges), "s", "d")
    q = "MATCH (a)-[*]->(b) RETURN count(*) AS c"
    assert _to_pandas(g_pl.gfql(q, engine="polars")._nodes)["c"].tolist() == \
        _to_pandas(g_pd.gfql(q, engine="pandas")._nodes)["c"].tolist(), \
        "a long acyclic walk must exhaust, not be reported as a cycle"

    # A 2-cycle {0,1} reachable from the seed, plus 300 nodes of UNREACHABLE chain. A
    # probe bounded by the global node count would spend ~300 eager collects here; the
    # answer (E108) must be the same either way.
    cyc = pd.DataFrame({"s": [0, 1] + list(range(100, 400)),
                        "d": [1, 0] + list(range(101, 401))})
    cyc_nodes = pd.DataFrame({"id": list(range(401))})
    g_cyc = graphistry.nodes(pl.from_pandas(cyc_nodes), "id").edges(
        pl.from_pandas(cyc), "s", "d")
    with pytest.raises(GFQLValidationError):
        g_cyc.gfql(q, engine="polars")


def test_polars_unbounded_varlen_self_loop_raises():
    """A self-loop is the smallest cycle — one hop that never exhausts."""
    from graphistry.compute.exceptions import GFQLValidationError

    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame({"s": [0, 1], "d": [1, 1], "type": ["R", "R"]})  # 1->1 self-loop
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises(GFQLValidationError):
        g.gfql("MATCH (a)-[*]->(b) RETURN count(*) AS c", engine="polars")


def test_polars_unbounded_varlen_declines_undirected_and_aliased():
    """Honest declines around the supported subset (NIE, never a silent answer)."""
    from graphistry.compute.ast import e_undirected
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    # undirected unbounded: multiplicity reconstruction + backtrack-aware termination
    assert binding_rows_polars(
        g, serialize_binding_ops([n(name="a"), e_undirected(to_fixed_point=True), n(name="b")])
    ) is None
    # aliased variable-length relationship: pandas rejects it outright
    assert binding_rows_polars(
        g, serialize_binding_ops([n(name="a"), e_forward(to_fixed_point=True, name="r"), n(name="b")])
    ) is None
    # unbounded WITHOUT to_fixed_point: pandas silently truncates at a bound this
    # lowering cannot reconstruct, so decline rather than guess the depth.
    assert binding_rows_polars(
        g, serialize_binding_ops([n(name="a"), e_forward(hops=None, min_hops=2), n(name="b")])
    ) is None


@pytest.mark.parametrize("min_hops", [0, 1, 2, 3])
def test_polars_unbounded_varlen_gate_admits_min_hops_le_1_only(min_hops):
    """Either side of the min_hops boundary on the UNBOUNDED DIRECTED arm.

    This is the Cypher-reachable shape: `-[*k..]->` lowers to
    ``{hops: None, min_hops: k, to_fixed_point: True, direction: forward}``. The sibling
    decline test above uses ``to_fixed_point=False``, which Cypher never emits — that is
    exactly why it missed this. For k >= 2 pandas' step_pairs come from the var-length
    ``edge_op.execute`` hop, which empties/prunes by min_hops against a dedup-by-node
    eccentricity; the raw-edge reconstruction here does not, so serving k >= 2 diverges
    SILENTLY (see the value test below).
    """
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    out = binding_rows_polars(g, serialize_binding_ops([
        n(name="a"), e_forward(hops=None, min_hops=min_hops, to_fixed_point=True), n(name="b"),
    ]))
    if min_hops <= 1:
        assert out is not None, f"min_hops={min_hops} is a supported shape and must be served"
    else:
        assert out is None, f"min_hops={min_hops} must DECLINE, not answer"


# 7 nodes, strictly increasing edges (acyclic, so the fixed point terminates and neither
# engine errors), with repeated edges so multiplicity — not just reachability — is under
# test. Serving `-[*k..]->` for k >= 2 returns counts of 30/41/6 here against pandas'
# 0/24/0; the whole point is that both engines answer, so only a value check catches it.
_MINHOP_NODES = pd.DataFrame({"id": list(range(7))})
_MINHOP_EDGES = pd.DataFrame(
    [(0, 3), (1, 2), (4, 6), (3, 5), (5, 6), (1, 2), (3, 5), (0, 6),
     (5, 6), (3, 4), (0, 4), (0, 4), (4, 5), (0, 3), (0, 3)],
    columns=["s", "d"],
)


@pytest.mark.parametrize("query", [
    "MATCH (a)-[*]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*0..]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*2..]->(b) RETURN count(*) AS c",
    "MATCH (a {id: 0})-[*2..]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*3..]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*4..]->(b) RETURN count(*) AS c",
])
def test_cypher_unbounded_varlen_min_hops_matches_pandas_or_declines(query):
    """End-to-end through Cypher, where the divergence is actually reachable.

    ``RETURN count(*)`` lowers to a pure-CALL chain (no ASTNode/ASTEdge), so
    ``_is_native_multihop`` in ``_chain_traversal_polars`` never runs and the ``rows`` gate
    is the ONLY gate — a decline there is the only thing standing between a user and a
    wrong count. Contract: polars either matches the pandas oracle or raises; never a
    different number.
    """
    g_pd = graphistry.nodes(_MINHOP_NODES, "id").edges(_MINHOP_EDGES, "s", "d")
    g_pl = graphistry.nodes(pl.from_pandas(_MINHOP_NODES), "id").edges(
        pl.from_pandas(_MINHOP_EDGES), "s", "d")
    expected = _to_pandas(g_pd.gfql(query, engine="pandas")._nodes)["c"].tolist()
    try:
        got = _to_pandas(g_pl.gfql(query, engine="polars")._nodes)["c"].tolist()
    except NotImplementedError:
        return  # an honest decline is allowed; a different answer is not
    assert got == expected, f"polars diverged from the pandas oracle on {query!r}"


@pytest.mark.parametrize("min_hops,max_hops", [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3),
                                               (3, 4), (3, 5), (4, 5)])
def test_polars_bounded_varlen_with_to_fixed_point_declines(min_hops, max_hops):
    """`to_fixed_point=True` COMBINED WITH an explicit bound must DECLINE, as on master.

    Master declined on `bool(op.to_fixed_point)` alone. #1781 restructured the gate to key
    on the RESOLVED MAX, which incidentally let bounded+fixed-point through to the bounded
    arm — and that arm is silently wrong there for min_hops >= 3, the same reconstruction
    gap as the unbounded case: `-[*3..5]->` with the flag set returns 30 rows against
    pandas' 0. Declining restores master's behaviour exactly, so it cannot regress anything.

    An earlier version of this test asserted the OPPOSITE (that the shape is served) and
    checked values by comparing flagged-vs-unflagged WITHIN each engine, with the
    parametrization stopping at min_hops=2. It was structurally incapable of catching the
    divergence: within-engine comparison never consults the pandas oracle, and min_hops<=2
    happens to agree on this fixture. Hence the range now runs past 3 and the assertion is
    about the gate, not about values.
    """
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    ops = [n(name="a"),
           e_forward(min_hops=min_hops, max_hops=max_hops, to_fixed_point=True),
           n(name="b")]
    g_pl = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    assert binding_rows_polars(g_pl, serialize_binding_ops(ops)) is None, \
        "bounded var-length carrying to_fixed_point must DECLINE, not answer"


def test_bounded_varlen_with_to_fixed_point_never_answers_differently_than_pandas():
    """The value side of the gate above, on the graph where it actually diverged.

    Contract: polars either matches the pandas oracle or raises. `min_hops >= 3` is where
    the bounded reconstruction and pandas' hop-window pruning part company, so these are
    the shapes an over-permissive gate returns wrong counts for (30 vs 0, 6 vs 0, 24 vs 0).

    Compares ROW COUNTS, not frames: a bare `rows()` call with no trailing projection emits
    engine-specific scaffolding columns (pandas keeps `id`, `a__a_join__`, `a.a`, ...;
    polars emits only the alias columns) on every shape, including a plain fixed 1-hop.
    That is pre-existing and unrelated; the row count is the binding multiplicity, which is
    exactly what diverges here and is engine-neutral.
    """
    g_pd = graphistry.nodes(_MINHOP_NODES, "id").edges(_MINHOP_EDGES, "s", "d")
    g_pl = graphistry.nodes(pl.from_pandas(_MINHOP_NODES), "id").edges(
        pl.from_pandas(_MINHOP_EDGES), "s", "d")
    for min_hops, max_hops in [(3, 4), (3, 5), (4, 5), (2, 3), (1, 3), (0, 2)]:
        ops = [n(name="a"),
               e_forward(min_hops=min_hops, max_hops=max_hops, to_fixed_point=True),
               n(name="b")]
        call = [rows(binding_ops=serialize_binding_ops(ops))]
        expected = len(_to_pandas(g_pd.gfql(call, engine="pandas")._nodes))
        try:
            got = len(_to_pandas(g_pl.gfql(call, engine="polars")._nodes))
        except NotImplementedError:
            continue  # an honest decline is allowed; a different answer is not
        assert got == expected, \
            f"polars diverged from the pandas oracle on min={min_hops} max={max_hops} " \
            f"to_fixed_point=True: {got} rows vs {expected}"


def test_polars_unbounded_varlen_no_matching_edges_keeps_zero_hop_rows():
    """No edge matches the type at all: the fixed point has depth 0, so only the
    zero-hop rows survive (min_hops == 0) or the table is empty (min_hops >= 1)."""
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import binding_rows_polars

    g = graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    zero = binding_rows_polars(
        g,
        serialize_binding_ops([
            n(name="a"), e_forward({"type": "NOPE"}, min_hops=0, to_fixed_point=True), n(name="b"),
        ]),
    )
    assert zero is not None and zero._nodes.height == len(NODES)
    one = binding_rows_polars(
        g,
        serialize_binding_ops([
            n(name="a"), e_forward({"type": "NOPE"}, min_hops=1, to_fixed_point=True), n(name="b"),
        ]),
    )
    assert one is not None and one._nodes.height == 0
