"""Parity + gating tests for the seeded node-lookup fast path and the two-alias
projection of the seeded typed hop.

  * ``MATCH (p {id}) RETURN p`` / ``RETURN p.a AS x`` resolve the seed through the
    resident node-id index, a resident node-property index, or one scalar scan, and
    project the matched rows directly (``_execute_seeded_node_lookup_fast_path``).
  * ``MATCH (m {id})-[:T]->(p) RETURN m.a, p.b`` projects from both aliases, one row per
    matched edge (``_execute_seeded_typed_hop_fast_path``).

Both are value-identical to the full path (same rows/columns/dtypes; row order and
index may differ, comparisons canonicalize); the tests pin fast==full differentially,
against an independent oracle, that the paths ENGAGE for the covered shapes, and that
they DECLINE for full-path side-channels.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
import graphistry.compute.gfql_unified as gfql_unified
from graphistry.tests.compute.gfql.engagement import fast_path_decisions

ENGINES = ["pandas", "polars", "cudf"]


def _frames(n_persons=300, n_messages=900, seed=0):
    rng = np.random.default_rng(seed)
    persons = pd.DataFrame({
        "id": np.arange(n_persons), "type": "Person",
        "firstName": [f"f{i}" for i in range(n_persons)],
        "age": rng.integers(20, 60, n_persons),
        "flag": rng.integers(0, 2, n_persons).astype(bool),
        "score": rng.integers(0, 100, n_persons),
    })
    messages = pd.DataFrame({
        "id": np.arange(n_persons, n_persons + n_messages), "type": "Message",
        "firstName": None, "age": np.nan, "flag": rng.integers(0, 2, n_messages).astype(bool),
        "score": rng.integers(0, 100, n_messages),
    })
    nodes = pd.concat([persons, messages], ignore_index=True)
    edges = pd.DataFrame({
        "src": np.arange(n_persons, n_persons + n_messages),
        "dst": rng.integers(0, n_persons, n_messages), "type": "HAS_CREATOR",
        "w": rng.integers(0, 100, n_messages), "eflag": rng.integers(0, 2, n_messages).astype(bool),
    })
    # parallel edges keep openCypher bag multiplicity honest
    edges = pd.concat([edges, edges.iloc[:5]], ignore_index=True)
    return nodes, edges


def _graph(engine, indexed=False):
    nodes, edges = _frames()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    return g.gfql_index_all(engine=engine) if indexed else g


def _canon(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    df.columns = [str(c) for c in df.columns]
    cols = sorted(df.columns)
    return df.sort_values(cols).reset_index(drop=True)[cols] if cols else df


def _run(g, engine, query, fast):
    real_hop = gfql_unified._execute_seeded_typed_hop_fast_path
    real_lookup = gfql_unified._execute_seeded_node_lookup_fast_path
    try:
        if not fast:
            gfql_unified._execute_seeded_typed_hop_fast_path = lambda *a, **k: None
            gfql_unified._execute_seeded_node_lookup_fast_path = lambda *a, **k: None
        return g.gfql(query, engine=engine)
    finally:
        gfql_unified._execute_seeded_typed_hop_fast_path = real_hop
        gfql_unified._execute_seeded_node_lookup_fast_path = real_lookup


def _assert_parity(g, engine, query, path, served=True):
    fast, full = _run(g, engine, query, True), _run(g, engine, query, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    fe, fu = fast._edges, full._edges
    assert (fe is None) == (fu is None)
    if fe is not None:
        assert len(fe) == len(fu)
    seen = fast_path_decisions(g, query, engine=engine)
    assert seen.get(path) is served, f"{path}: {seen}"
    return _canon(fast)


LOOKUP_SHAPES = [
    ("MATCH (p:Person {id: 7}) RETURN p", "whole row"),
    ("MATCH (p:Person {id: 7}) RETURN p.firstName AS firstName, p.age AS age", "props"),
    ("MATCH (p {id: 7}) RETURN p.flag AS flag, p.id AS pid", "bool + id, no label"),
    ("MATCH (p:Person {id: 7}) RETURN p.id AS a, p.id AS b, p.score AS s", "same column twice + int"),
    ("MATCH (p:Person {id: 999999}) RETURN p.score AS s, p.flag AS f", "no match, int + bool"),
    ("MATCH (p:Person {id: 7}) RETURN p.age AS age ORDER BY age LIMIT 1", "order/limit"),
    ("MATCH (p:Person {id: 7}) RETURN DISTINCT p.age AS age", "distinct"),
    ("MATCH (p:Person {id: 999999}) RETURN p.age AS age", "no match"),
    ("MATCH (p:Message {id: 7}) RETURN p", "label mismatch -> empty"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("q,label", LOOKUP_SHAPES)
def test_node_lookup_engages_with_parity(engine, indexed, q, label):
    _assert_parity(_graph(engine, indexed), engine, q, "seeded_node_lookup")


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_matches_independent_oracle(engine):
    g = _graph(engine)
    got = _assert_parity(g, engine, "MATCH (p:Person {id: 7}) RETURN p.firstName AS f, p.age AS a",
                         "seeded_node_lookup")
    nodes, _ = _frames()
    row = nodes[nodes["id"] == 7].iloc[0]
    assert got["f"].tolist() == [row["firstName"]]
    assert float(got["a"].iloc[0]) == float(row["age"])


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_uses_the_property_index_when_the_seed_is_not_the_binding(engine):
    """The seed predicate is on a business key that is not the node binding: the
    resident property index answers it and the traversal trace says so."""
    nodes, edges = _frames()
    nodes = nodes.assign(key=np.arange(len(nodes)) + 1000)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "src", "dst")
    g = g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine)
    q = "MATCH (p:Person {id: 7}) RETURN p.firstName AS f, p.key AS k"
    got = _assert_parity(g, engine, q, "seeded_node_lookup")
    assert got["k"].tolist() == [1007]
    steps = g.gfql_explain(q, engine=engine).get("steps", [])
    lookups = [s for s in steps if s.get("seam") == "node_lookup"]
    assert lookups and lookups[0]["served"] is True


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("q,label", [
    ("MATCH (p:Person {id: 7}) WHERE p.age > 0 RETURN p.age AS age", "same-path WHERE"),
    ("MATCH (p:Person {id: 7}) RETURN p.age AS age, p.nosuch AS x", "absent property"),
    ("MATCH (p:Person) RETURN p.age AS age ORDER BY age LIMIT 1", "no selective seed"),
    ("MATCH (p:Person {id: 7}), (q:Person {id: 8}) RETURN p.age AS a, q.age AS b", "two node patterns"),
    ("MATCH (p:Person {id: 7}) RETURN count(*) AS n", "aggregate"),
    ("OPTIONAL MATCH (p:Person {id: 999999}) RETURN p.age AS age", "optional null row"),
])
def test_node_lookup_declines_out_of_shape_with_parity(engine, q, label):
    g = _graph(engine)
    try:
        fast = _run(g, engine, q, True)
    except Exception as exc:  # noqa: BLE001
        with pytest.raises(type(exc)):
            _run(g, engine, q, False)
        return
    full = _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    assert fast_path_decisions(g, q, engine=engine).get("seeded_node_lookup") is not True, label


def test_node_lookup_declines_under_policy():
    g = _graph("pandas")
    fired = {"n": 0}

    def hook(ctx):
        fired["n"] += 1

    q = "MATCH (p:Person {id: 7}) RETURN p.age AS age"
    out = g.gfql(q, engine="pandas", policy={"preload": hook, "postload": hook})
    assert fired["n"] > 0, "policy hooks must fire (fast path must decline under policy)"
    pd.testing.assert_frame_equal(_canon(out), _canon(_run(g, "pandas", q, False)))


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_returns_each_duplicate_id_row_once(engine):
    """Two node rows with the same id both match a lookup on that id: one row each."""
    nodes, edges = _frames()
    nodes = pd.concat([nodes, nodes[nodes["id"] == 7]], ignore_index=True)
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    out = g.gfql("MATCH (p:Person {id: 7}) RETURN p.score AS s", engine=engine)
    assert len(out._nodes) == 2


TWO_ALIAS_SHAPES = [
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.firstName AS f", "id + str"),
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN p.age AS age, m.type AS mt, p.id AS pid", "mixed"),
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS a, m.id AS b, p.flag AS c", "repeated + bool"),
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.score AS ms, p.score AS ps, m.flag AS mf, p.flag AS pf", "int + bool from both aliases"),
    ("MATCH (m:Message {id: 999999})-[:HAS_CREATOR]->(p:Person) RETURN m.score AS ms, p.score AS ps, p.flag AS pf", "no match, int + bool"),
    ("MATCH (m:Message {score: 42})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.score AS ps, p.flag AS pf", "non-binding seed, int + bool"),
    ("MATCH (m:Message {id: 301})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.id AS pid", "parallel edges"),
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.age AS age ORDER BY age LIMIT 1", "order/limit"),
    ("MATCH (m:Message {id: 305})-[r:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, r.w AS w, r.eflag AS ef, r.type AS rt, p.age AS age", "all three aliases"),
    ("MATCH (m:Message {id: 305})-[r:HAS_CREATOR]->(p:Person) RETURN r.w AS w, r.src AS s, r.dst AS d", "edge alias only, incl. endpoints"),
    ("MATCH (m:Message {id: 305})-[r:HAS_CREATOR]->(p:Person) RETURN m.id, m.score, m.flag, m.firstName, r.w, r.eflag, r.type, p.id, p.firstName, p.age, p.score, p.flag", "twelve properties across three aliases"),
    ("MATCH (m:Message {id: 301})-[r:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, r.w AS w, p.id AS pid", "parallel edges, edge prop distinguishes rows"),
    ("MATCH (m:Message {id: 999999})-[r:HAS_CREATOR]->(p:Person) RETURN r.w AS w, p.age AS age", "no match with an edge prop"),
    ("MATCH (m:Message {id: 999999})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.age AS age", "no match"),
    ("MATCH (p:Person)<-[:HAS_CREATOR]-(m:Message {id: 305}) RETURN m.id AS mid, p.age AS age", "reverse pattern, seed on return side"),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("q,label", TWO_ALIAS_SHAPES)
def test_two_alias_projection_parity(engine, indexed, q, label):
    g = _graph(engine, indexed)
    fast, full = _run(g, engine, q, True), _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))


@pytest.mark.parametrize("engine", ENGINES)
def test_two_alias_projection_engages_and_keeps_bag_multiplicity(engine):
    g = _graph(engine)
    q = "MATCH (m:Message {id: 301})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.id AS pid"
    got = _assert_parity(g, engine, q, "seeded_typed_hop")
    _, edges = _frames()
    expected = edges[edges["src"] == 301]["dst"].tolist()
    assert sorted(got["pid"].tolist()) == sorted(expected)
    assert len(got) == 2, "the duplicated edge yields two rows"
    assert got["mid"].tolist() == [301, 301]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
def test_hub_seed_over_the_frontier_gate_keeps_parity(engine, indexed):
    """A seed set whose frontier trips the indexed kernel's cost gate: the canonical
    path falls back to the rows pivot, and the fast path must match its dtypes."""
    nodes, edges = _frames()
    nodes = nodes.assign(bucket=(nodes["id"] % 2).astype("int64"))
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    if indexed:
        g = g.gfql_index_all(engine=engine)
    q = "MATCH (m:Message {bucket: 1})-[:HAS_CREATOR]->(p:Person) RETURN m.score AS ms, p.score AS ps, p.flag AS pf"
    fast, full = _run(g, engine, q, True), _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))


@pytest.mark.parametrize("engine", ENGINES)
def test_seed_matching_several_nodes_projects_each_seed(engine):
    """A non-unique seed predicate: every seed row pairs with its own destinations."""
    nodes, edges = _frames()
    nodes = nodes.assign(bucket=(nodes["id"] % 7).astype("int64"))
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "src", "dst")
    q = "MATCH (m:Message {bucket: 3})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.id AS pid"
    got = _assert_parity(g, engine, q, "seeded_typed_hop")
    raw_nodes, raw_edges = _frames()
    seeds = raw_nodes[(raw_nodes["type"] == "Message") & (raw_nodes["id"] % 7 == 3)]["id"]
    oracle = raw_edges[raw_edges["src"].isin(seeds)][["src", "dst"]]
    assert sorted(zip(got["mid"].tolist(), got["pid"].tolist())) == sorted(zip(oracle["src"], oracle["dst"]))


# ---- boundary pins on the other side of each admission check (owner review of #2035) ----

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("q,label", [
    ("MATCH (p:Person {id: 7}) RETURN p ORDER BY p.age LIMIT 1", "whole row + suffix"),
    ("MATCH (p:Person {id: 7}) RETURN p.age + 1 AS x", "expression item"),
    ("MATCH (p:Person {id: 7}) RETURN p.age AS a, p AS whole", "property and whole row"),
])
def test_node_lookup_declines_shapes_outside_its_projection_contract(engine, q, label):
    g = _graph(engine)
    fast, full = _run(g, engine, q, True), _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    assert fast_path_decisions(g, q, engine=engine).get("seeded_node_lookup") is not True, label


def test_node_lookup_declines_lazyframe_nodes_with_parity():
    pl = pytest.importorskip("polars")
    g = _graph("polars")
    lazy = g.nodes(g._nodes.lazy())
    q = "MATCH (p:Person {id: 7}) RETURN p.age AS age"
    fast, full = _run(lazy, "polars", q, True), _run(lazy, "polars", q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    assert fast_path_decisions(lazy, q, engine="polars").get("seeded_node_lookup") is not True
    assert isinstance(g._nodes, pl.DataFrame)


def test_node_lookup_declines_when_the_requested_engine_is_not_the_frames_engine():
    pytest.importorskip("polars")
    g = _graph("polars")
    q = "MATCH (p:Person {id: 7}) RETURN p.age AS age"
    fast, full = _run(g, "pandas", q, True), _run(g, "pandas", q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    assert fast_path_decisions(g, q, engine="pandas").get("seeded_node_lookup") is not True


def _lookup_step(report):
    steps = [s for s in report["steps"] if s.get("seam") == "node_lookup"]
    assert len(steps) == 1, report["steps"]
    return steps[0]


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_explains_why_it_scanned(engine):
    q = "MATCH (p:Person {id: 7}) RETURN p.age AS age"
    g = _graph(engine)
    missing = _lookup_step(g.gfql_explain(q, engine=engine, index_policy="use"))
    assert (missing["served"], missing["reason"]) == (False, "index_missing")
    gi = _graph(engine, indexed=True)
    served = _lookup_step(gi.gfql_explain(q, engine=engine, index_policy="use"))
    assert (served["served"], served["reason"]) == (True, "served")
    off = gi.gfql_explain(q, engine=engine, index_policy="off")
    assert off["used_index"] is False and off["decision_code"] == "policy_off"
    nodes, _ = _frames()
    rebound = gi.nodes(gi._nodes.head(len(nodes) - 1))
    stale = _lookup_step(rebound.gfql_explain(q, engine=engine, index_policy="use"))
    assert (stale["served"], stale["reason"]) == (False, "index_stale")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
def test_two_alias_projection_keeps_parity_on_datetime_columns(engine, indexed):
    g = _graph(engine, indexed)
    nodes = g._nodes
    if engine == "polars":
        import polars as pl
        nodes = nodes.with_columns(pl.lit("2026-01-01").str.to_datetime().alias("ts"))
    else:
        nodes = nodes.assign(ts=pd.Timestamp("2026-01-01"))
    g = g.nodes(nodes)
    q = "MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.ts AS t, p.age AS age"
    fast, full = _run(g, engine, q, True), _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))


@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("q,served", [
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.nullable AS n, p.age AS age", None),
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, p.nullable AS pn", False),
], ids=["seed-side extension dtype", "dst-side extension dtype declines"])
def test_two_alias_projection_extension_dtypes_keep_parity(indexed, q, served):
    g = _graph("pandas", indexed)
    g = g.nodes(g._nodes.assign(nullable=pd.array([1] * len(g._nodes), dtype="Int64")))
    fast, full = _run(g, "pandas", q, True), _run(g, "pandas", q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    if served is False:
        assert fast_path_decisions(g, q, engine="pandas").get("seeded_typed_hop") is not True


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_served_under_policy_off_is_not_reported_as_an_index(engine):
    q = "MATCH (p:Person {id: 7}) RETURN p.age AS age"
    gi = _graph(engine, indexed=True)
    from graphistry.compute.gfql.index import index_trace
    with index_trace() as steps:
        gi.gfql(q, engine=engine, index_policy="off")
    assert any(s.get("op") == "fast_path" and s.get("seam") == "seeded_node_lookup" and s.get("served")
               for s in steps), steps
    off = gi.gfql_explain(q, engine=engine, index_policy="off")
    assert off["used_index"] is False and off["decision_code"] == "policy_off", off


@pytest.mark.parametrize("engine", ENGINES)
def test_edge_alias_properties_engage_with_one_row_per_matched_edge(engine):
    g = _graph(engine)
    q = "MATCH (m:Message {id: 301})-[r:HAS_CREATOR]->(p:Person) RETURN m.id AS mid, r.w AS w, p.id AS pid"
    got = _assert_parity(g, engine, q, "seeded_typed_hop")
    _, edges = _frames()
    assert sorted(got["w"].tolist()) == sorted(edges[edges["src"] == 301]["w"].tolist())
    assert len(got) == 2, "the duplicated edge yields two rows with the same edge props"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("q,label", [
    ("MATCH (m:Message {id: 305})-[:HAS_CREATOR]->(p:Person)-[:HAS_CREATOR]->(q) RETURN m.id AS a, p.id AS b, q.id AS c", "two hops"),
    ("MATCH (m:Message {id: 305})-[r:HAS_CREATOR]->(p:Person) RETURN r.nosuch AS x", "absent edge property"),
])
def test_typed_hop_declines_beyond_one_hop_or_unknown_edge_property_with_parity(engine, q, label):
    g = _graph(engine)
    try:
        fast = _run(g, engine, q, True)
    except Exception as exc:  # noqa: BLE001
        with pytest.raises(type(exc)):
            _run(g, engine, q, False)
        return
    full = _run(g, engine, q, False)
    pd.testing.assert_frame_equal(_canon(fast), _canon(full))
    assert fast_path_decisions(g, q, engine=engine).get("seeded_typed_hop") is not True, label
