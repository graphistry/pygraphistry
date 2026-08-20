"""`CALL graphistry.std.*` — the kernels reachable as GFQL queries."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.algorithms import kernels as K
from graphistry.compute.algorithms._dfops import dense_renumber


@pytest.fixture(scope="module")
def g():
    return graphistry.edges(pd.DataFrame({"s": [10, 11, 10, 55], "d": [11, 12, 12, 66]}), "s", "d")


@pytest.mark.parametrize(
    "query,col",
    [
        ("CALL graphistry.std.wcc.write()", "component"),
        ("CALL graphistry.std.cdlp.write({params: {iterations: 3}})", "cdlp"),
        ("CALL graphistry.std.mis.write()", "mis"),
        ("CALL graphistry.std.sssp.write({params: {source: 10}})", "distance"),
        ("CALL graphistry.std.pagerank.write()", "pagerank"),
    ],
)
def test_std_call_writes_expected_column(g, query, col):
    assert col in g.gfql(query)._nodes.columns


def test_out_col_override(g):
    assert "pr" in g.gfql("CALL graphistry.std.pagerank.write({out_col: 'pr'})")._nodes.columns


def test_std_row_call_yields_node_values(g):
    out = g.gfql("CALL graphistry.std.pagerank() YIELD nodeId, pagerank RETURN nodeId, pagerank")
    assert list(out._nodes.columns) == ["nodeId", "pagerank"]
    assert set(out._nodes["nodeId"]) == {10, 11, 12, 55, 66}


def test_unknown_std_procedure_is_rejected(g):
    from graphistry.compute.exceptions import GFQLValidationError

    with pytest.raises(GFQLValidationError):
        g.gfql("CALL graphistry.std.louvain.write()")


def test_weighted_sssp_uses_original_string_source_id():
    edges = pd.DataFrame({"s": ["a", "b", "a"], "d": ["b", "c", "c"], "cost": [2.0, 3.0, 99.0]})
    out = graphistry.edges(edges, "s", "d").gfql("CALL graphistry.std.sssp.write({params: {source: 'a', weight: 'cost'}})")
    got = out._nodes.set_index("id")["distance"].to_dict()
    assert got == {"a": 0.0, "b": 2.0, "c": 5.0}


@pytest.mark.parametrize("source", [0, "missing"])
def test_sssp_rejects_unknown_original_source_id(g, source):
    with pytest.raises(ValueError, match="is not a graph node"):
        g.gfql(f"CALL graphistry.std.sssp.write({{params: {{source: {source!r}}}}})")


def test_mis_drops_self_loops_at_public_boundary():
    graph = graphistry.edges(pd.DataFrame({"s": [1], "d": [1]}), "s", "d").nodes(pd.DataFrame({"id": [1]}), "id")
    out = graph.gfql("CALL graphistry.std.mis.write()")
    assert bool(out._nodes.set_index("id").loc[1, "mis"]) is True


@pytest.mark.parametrize(
    "query,column,expected",
    [
        ("CALL graphistry.std.wcc.write()", "component", 99),
        ("CALL graphistry.std.cdlp.write()", "cdlp", 99),
        ("CALL graphistry.std.mis.write()", "mis", True),
        ("CALL graphistry.std.sssp.write({params: {source: 10}})", "distance", np.inf),
    ],
)
def test_std_call_retains_explicit_isolate(query, column, expected):
    graph = graphistry.edges(pd.DataFrame({"s": [10], "d": [20]}), "s", "d").nodes(pd.DataFrame({"id": [10, 20, 99]}), "id")
    out = graph.gfql(query)
    isolate = out._nodes.set_index("id").loc[99, column]
    assert isolate == expected


def test_pagerank_retains_explicit_isolate():
    graph = graphistry.edges(pd.DataFrame({"s": [10], "d": [20]}), "s", "d").nodes(pd.DataFrame({"id": [10, 20, 99]}), "id")
    isolate = graph.gfql("CALL graphistry.std.pagerank.write()")._nodes.set_index("id").loc[99]
    assert isolate["pagerank"] > 0.0


def test_std_call_supports_negative_node_ids():
    graph = graphistry.edges(pd.DataFrame({"s": [-2], "d": [-1]}), "s", "d")
    got = graph.gfql("CALL graphistry.std.wcc.write()")._nodes.set_index("id")["component"]
    assert got.to_dict() == {-2: -2, -1: -2}


def test_call_result_matches_the_direct_kernel():
    """The query surface must not change the answer, and WCC labels must come
    back in the caller's id space -- the label IS a vertex id."""
    rng = np.random.default_rng(5)
    e = pd.DataFrame({"s": rng.integers(0, 500, 3000), "d": rng.integers(0, 500, 3000)})
    e = e[e["s"] != e["d"]]

    out = graphistry.edges(e, "s", "d").gfql("CALL graphistry.std.wcc.write()")
    got = out._nodes.sort_values("id").reset_index(drop=True)

    dense, ids, v_count = dense_renumber(e, "s", "d")
    ref = [int(ids.iloc[int(x)]) for x in K.wcc(dense, "s", "d", v_count)]

    assert list(got["component"]) == ref
    assert got["component"].min() == got["id"].min()
