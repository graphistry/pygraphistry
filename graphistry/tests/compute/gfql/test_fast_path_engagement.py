"""Fast paths are contracted "same answer, faster", so a DEAD one is invisible:
the query still returns the right result via the fallback and every value test
passes. These assert ENGAGEMENT -- that the path actually fired -- which is the
assertion a value test structurally cannot make.
"""
import os

import pandas as pd
import pytest

from graphistry.tests.compute.gfql.engagement import (
    assert_fast_path, fast_path_decisions,
)

import graphistry

Q_TWO_HOP = ("MATCH (a {kind:'P'})-[{rel:'F'}]->(b {kind:'P'})"
             "-[{rel:'F'}]->(c {kind:'P'}) RETURN count(*) AS n")
Q_GROUPED = ("MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) "
             "RETURN c.city AS city, count(*) AS n ORDER BY city ASC")
Q_PLAIN = "MATCH (a)-[]->(b) RETURN a.id AS x ORDER BY x ASC"


def _graph(engine: str = "pandas"):
    nodes = pd.DataFrame({"id": list(range(6)), "kind": ["P"] * 3 + ["C"] * 3,
                          "city": ["LA", "NY", "SF"] * 2})
    edges = pd.DataFrame({"s": [0, 1, 2, 0, 0], "d": [1, 2, 0, 3, 4],
                          "rel": ["F", "F", "F", "L", "L"]})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


ENGINES = ["pandas", "polars", "cudf"]


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_two_hop_count_fast_path_engages(engine: str) -> None:
    """Engagement is per-ENGINE: a path that serves on pandas can silently decline
    on polars or cuDF and no value test would notice, because both answer."""
    assert_fast_path(_graph(engine), Q_TWO_HOP, "two_hop_count",
                     served=True, engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.route_engaged("cypher-fast")
def test_grouped_aggregate_fast_path_engages(engine: str) -> None:
    assert_fast_path(_graph(engine), Q_GROUPED, "single_hop_grouped_aggregate",
                     served=True, engine=engine)


_requires_test_cudf = pytest.mark.skipif(
    os.environ.get("TEST_CUDF", "0") != "1", reason="cudf tests need TEST_CUDF=1 and a GPU")
GPU_GATED_ENGINES = ["pandas", "polars", pytest.param("cudf", marks=_requires_test_cudf)]

# 1 A x 10, 2 B y 20, 3 B x 30, 4 null y 40, 5 C x 50; edges 1->2, 1->3, 1->4, 2->3, 3->4, 5->1
GROUPED_BY_PROPERTY = [
    ("seeded", "MATCH (a {id:1})-[:R]->(b) RETURN b.COL AS key, count(a) AS c ORDER BY key",
     [{"key": "B", "c": 2}, {"key": None, "c": 1}]),
    ("key_named_after_column", "MATCH (a)-[:R]->(b) RETURN b.COL AS COL, count(*) AS c ORDER BY COL",
     [{"COL": "A", "c": 1}, {"COL": "B", "c": 3}, {"COL": None, "c": 2}]),
    ("null_group_and_sum", "MATCH (a)-[:R]->(b) RETURN b.COL AS key, count(b) AS c, sum(b.w) AS s ORDER BY key",
     [{"key": "A", "c": 1, "s": 10}, {"key": "B", "c": 3, "s": 80}, {"key": None, "c": 2, "s": 80}]),
]


def _rows(df) -> list:
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    rows = [{k: (None if pd.isna(v) else v) for k, v in r.items()} for r in df.to_dict("records")]
    return sorted(rows, key=lambda r: str(r))


def _graph_with_property(engine: str, col: str):
    nodes = pd.DataFrame({"id": [1, 2, 3, 4, 5], col: ["A", "B", "B", None, "C"],
                          "kind": ["x", "y", "x", "y", "x"], "w": [10, 20, 30, 40, 50]})
    edges = pd.DataFrame({"s": [1, 1, 1, 2, 3, 5], "d": [2, 3, 4, 3, 4, 1], "type": ["R"] * 6})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", GPU_GATED_ENGINES)
@pytest.mark.parametrize("col", ["title", "name"])
@pytest.mark.parametrize("_case,query,expected", GROUPED_BY_PROPERTY, ids=[c[0] for c in GROUPED_BY_PROPERTY])
def test_grouped_aggregate_fast_path_serves_a_property_named_name(engine, col, _case, query, expected) -> None:
    """A node property named ``name`` crashed this path on cuDF (#1946); ``title`` is the control."""
    g = _graph_with_property(engine, col)
    q = query.replace("COL", col)
    assert_fast_path(g, q, "single_hop_grouped_aggregate", served=True, engine=engine)
    want = [{(col if k == "COL" else k): v for k, v in row.items()} for row in expected]
    assert _rows(g.gfql(q, engine=engine)._nodes) == sorted(want, key=lambda r: str(r))


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", GPU_GATED_ENGINES)
def test_grouped_aggregate_fast_path_serves_an_edge_source_column_named_name(engine) -> None:
    nodes = pd.DataFrame({"id": [1, 2, 3, 4, 5], "kind": ["x", "y", "x", "y", "x"]})
    edges = pd.DataFrame({"name": [1, 1, 1, 2, 3, 5], "d": [2, 3, 4, 3, 4, 1]})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "name", "d")
    q = "MATCH (a)-[]->(b) RETURN a.kind AS k, count(*) AS c ORDER BY k"
    assert_fast_path(g, q, "single_hop_grouped_aggregate", served=True, engine=engine)
    assert _rows(g.gfql(q, engine=engine)._nodes) == [{"k": "x", "c": 5}, {"k": "y", "c": 1}]


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_aggregate_fast_path_declines_output_name_colliding_with_edge_endpoint(engine: str) -> None:
    """An output name equal to an edge endpoint column ('s') is a collision BOTH lanes get
    wrong: the eager pandas merges suffix it and the groupby raises a bare KeyError, and the
    fused polars lane silently returned the edge src id instead of c.city. The decline is
    asserted on every engine because a per-lane guard is what let polars answer wrong."""
    g = _graph(engine)
    q = "MATCH (p {kind:'P'})-[{rel:'L'}]->(c {kind:'C'}) RETURN c.city AS s, count(*) AS n"
    assert_fast_path(g, q, "single_hop_grouped_aggregate", served=False, engine=engine)
    out = g.gfql(q, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert sorted(out["s"].tolist()) == ["LA", "NY"]


@pytest.mark.parametrize("engine", ENGINES)
def test_a_shape_neither_path_serves_declines_both(engine: str) -> None:
    """The negative side: EVERY path consulted, all decline, answer still correct.

    Asserted as an exact dict rather than a subset so that instrumenting a new
    fast path forces this control to be revisited -- a negative control that
    silently ignores a new path stops being a control."""
    g = _graph(engine)
    seen = fast_path_decisions(g, Q_PLAIN, engine=engine)
    assert seen == {"single_hop_grouped_aggregate": False, "two_hop_count": False,
                    "seeded_typed_hop": False, "seeded_node_lookup": False}
    # openCypher bag semantics (#1899): one row per pattern match. Edges are
    # (0->1),(1->2),(2->0),(0->3),(0->4), so a.id is [0,0,0,1,2] -- 5 rows.
    # The old `== 3` asserted the deduplicated node set, i.e. the #1899
    # multiplicity-collapse bug this suite's fallback now no longer has.
    out = g.gfql(Q_PLAIN, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert [int(v) for v in out["x"]] == [0, 0, 0, 1, 2]


@pytest.mark.route_engaged("cypher-fast")
def test_short_circuit_is_distinguishable_from_decline() -> None:
    """When grouped-agg serves, two_hop_count is never CONSULTED -- absent from the
    map rather than present-and-False. Conflating those would let a genuinely dead
    path read as 'declined for a good reason'."""
    seen = fast_path_decisions(_graph(), Q_GROUPED)
    assert seen.get("single_hop_grouped_aggregate") is True
    assert "two_hop_count" not in seen


@pytest.mark.route_engaged("cypher-fast")
def test_assert_fast_path_fails_when_the_path_did_not_fire() -> None:
    """An engagement pin that cannot fail is worse than none."""
    with pytest.raises(AssertionError, match="expected served=True"):
        assert_fast_path(_graph(), Q_PLAIN, "two_hop_count", served=True)
    with pytest.raises(AssertionError, match="never consulted"):
        assert_fast_path(_graph(), Q_GROUPED, "two_hop_count", served=True)


def test_unknown_fast_path_name_is_reported_not_silently_missing() -> None:
    """``FastPathName`` is a Literal so a typo is a TYPE error at author time; at
    RUNTIME an unknown name must still fail loudly rather than read as 'not
    consulted', which would look exactly like a correctly-declining path."""
    with pytest.raises(AssertionError, match="never consulted"):
        assert_fast_path(_graph(), Q_TWO_HOP, "two_hop_cont", served=True)  # type: ignore[arg-type]


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_typed_hop_fast_path_engages(engine: str) -> None:
    """The third path. It is consulted LAST, so its pin doubles as evidence the
    two ahead of it declined rather than short-circuited -- all three appear."""
    nodes = pd.DataFrame({"id": list(range(8)), "kind": ["P"] * 4 + ["C"] * 4})
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4], "rel": ["F"] * 4})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a {id:0})-[{rel:'F'}]->(b) RETURN b.id AS x ORDER BY x ASC"

    seen = fast_path_decisions(g, q, engine=engine)
    assert seen.get("seeded_typed_hop") is True
    assert seen.get("single_hop_grouped_aggregate") is False
    assert seen.get("two_hop_count") is False


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_node_lookup_fast_path_engages(engine: str) -> None:
    """The fourth path, consulted last: a seeded single-node pattern with a property
    RETURN. The three ahead of it decline on op shape, so all four appear."""
    g = _graph(engine)
    q = "MATCH (a {id: 1}) RETURN a.city AS c"
    seen = fast_path_decisions(g, q, engine=engine)
    assert seen == {"single_hop_grouped_aggregate": False, "two_hop_count": False,
                    "seeded_typed_hop": False, "seeded_node_lookup": True}
    out = g.gfql(q, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    assert out["c"].tolist() == ["NY"]


def test_fast_paths_have_no_bare_collect():
    """#1824: a bare LazyFrame.collect() in a fast path runs on CPU regardless of
    the requested engine, silently mislabeling CPU work as polars-gpu. Every
    collect must go through the target-honoring lazy.collect/collect_all."""
    import re
    from pathlib import Path
    import graphistry.compute.gfql_fast_paths as fp

    src = Path(fp.__file__).read_text()
    code = [l for l in src.splitlines() if not l.lstrip().startswith("#")]
    offenders = [l.strip() for l in code
                 if re.search(r"\.collect\(\)", l) or re.search(r"\bpl\.collect_all\(", l)]
    assert not offenders, f"bare collects bypass the execution target: {offenders}"


@pytest.mark.route_engaged("cypher-fast")
@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fast_paths_serve_identically_under_cpu_target_context(engine: str) -> None:
    """The #1824 target threading must be a no-op on CPU engines: same answers,
    same engagement, with the dispatch now setting an explicit CPU target."""
    assert_fast_path(_graph(engine), Q_TWO_HOP, "two_hop_count",
                     served=True, engine=engine)
    assert_fast_path(_graph(engine), Q_GROUPED, "single_hop_grouped_aggregate",
                     served=True, engine=engine)
