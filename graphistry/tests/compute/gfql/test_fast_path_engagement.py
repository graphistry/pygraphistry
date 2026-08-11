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
def test_a_shape_neither_path_serves_declines_both(engine: str) -> None:
    """The negative side: both consulted, both decline, answer still correct."""
    g = _graph(engine)
    seen = fast_path_decisions(g, Q_PLAIN, engine=engine)
    assert seen == {"single_hop_grouped_aggregate": False, "two_hop_count": False}
    assert len(g.gfql(Q_PLAIN, engine=engine)._nodes) == 3  # distinct a.id, not edge count


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
