"""Fast paths are contracted "same answer, faster", so a DEAD one is invisible:
the query still returns the right result via the fallback and every value test
passes. These assert ENGAGEMENT -- that the path actually fired -- which is the
assertion a value test structurally cannot make.
"""
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


@pytest.mark.parametrize("engine", ENGINES)
def test_two_hop_count_fast_path_engages(engine: str) -> None:
    """Engagement is per-ENGINE: a path that serves on pandas can silently decline
    on polars or cuDF and no value test would notice, because both answer."""
    assert_fast_path(_graph(engine), Q_TWO_HOP, "two_hop_count",
                     served=True, engine=engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_grouped_aggregate_fast_path_engages(engine: str) -> None:
    assert_fast_path(_graph(engine), Q_GROUPED, "single_hop_grouped_aggregate",
                     served=True, engine=engine)


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


def test_short_circuit_is_distinguishable_from_decline() -> None:
    """When grouped-agg serves, two_hop_count is never CONSULTED -- absent from the
    map rather than present-and-False. Conflating those would let a genuinely dead
    path read as 'declined for a good reason'."""
    seen = fast_path_decisions(_graph(), Q_GROUPED)
    assert seen.get("single_hop_grouped_aggregate") is True
    assert "two_hop_count" not in seen


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


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fast_paths_serve_identically_under_cpu_target_context(engine: str) -> None:
    """The #1824 target threading must be a no-op on CPU engines: same answers,
    same engagement, with the dispatch now setting an explicit CPU target."""
    assert_fast_path(_graph(engine), Q_TWO_HOP, "two_hop_count",
                     served=True, engine=engine)
    assert_fast_path(_graph(engine), Q_GROUPED, "single_hop_grouped_aggregate",
                     served=True, engine=engine)
