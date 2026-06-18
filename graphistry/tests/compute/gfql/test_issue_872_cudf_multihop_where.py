import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import e_forward, n
from graphistry.compute.gfql.df_executor import execute_same_path_chain
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.tests.test_compute import CGFull


def _to_pandas(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _graph(node_rows, edge_rows):
    return CGFull().nodes(pd.DataFrame(node_rows), "id").edges(pd.DataFrame(edge_rows), "src", "dst")


def _graph_cudf(cudf, graph):
    return CGFull().nodes(cudf.from_pandas(graph._nodes), graph._node).edges(
        cudf.from_pandas(graph._edges),
        graph._source,
        graph._destination,
    )


def _node_edge_sets(result):
    nodes = _to_pandas(result._nodes)
    edges = _to_pandas(result._edges)
    return set(nodes["id"]), set(map(tuple, edges[["src", "dst"]].itertuples(index=False, name=None)))


ISSUE_872_CASES = [
    (
        [
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ],
        [
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ],
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=3), n(name="end")],
        [compare(col("start", "v"), "<", col("end", "v"))],
        {"a", "b", "c", "d"},
        {("a", "b"), ("b", "c"), ("c", "d")},
    ),
    (
        [
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ],
        [
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "c", "dst": "d"},
        ],
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
        [compare(col("start", "v"), "<", col("end", "v"))],
        {"a", "b", "c", "d"},
        {("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")},
    ),
    (
        [
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ],
        [
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},
        ],
        [n({"id": "a"}, name="start"), e_forward(min_hops=3, max_hops=3), n(name="end")],
        [compare(col("start", "v"), "<=", col("end", "v"))],
        {"a", "b", "c"},
        {("a", "b"), ("b", "c"), ("c", "a")},
    ),
]

ISSUE_872_IDS = ["linear_min_hops_where", "diamond_multihop_where", "cycle_exact_min_hops_where"]


@pytest.mark.parametrize(
    "node_rows,edge_rows,chain,where,expected_nodes,expected_edges",
    ISSUE_872_CASES,
    ids=ISSUE_872_IDS,
)
def test_issue_872_pandas_multihop_where_baseline(
    node_rows,
    edge_rows,
    chain,
    where,
    expected_nodes,
    expected_edges,
):
    graph = _graph(node_rows, edge_rows)
    pandas_result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
    assert _node_edge_sets(pandas_result) == (expected_nodes, expected_edges)


@pytest.mark.parametrize(
    "node_rows,edge_rows,chain,where,expected_nodes,expected_edges",
    ISSUE_872_CASES,
    ids=ISSUE_872_IDS,
)
def test_issue_872_cudf_multihop_where_matches_pandas(
    node_rows,
    edge_rows,
    chain,
    where,
    expected_nodes,
    expected_edges,
):
    graph = _graph(node_rows, edge_rows)
    cudf = pytest.importorskip("cudf")
    try:
        cudf_graph = _graph_cudf(cudf, graph)
    except Exception as exc:
        pytest.skip(f"cuDF installed but no usable CUDA device: {exc}")

    cudf_result = execute_same_path_chain(cudf_graph, chain, where, Engine.CUDF)
    assert _node_edge_sets(cudf_result) == (expected_nodes, expected_edges)
    assert type(cudf_result._nodes).__module__.startswith("cudf")
    assert type(cudf_result._edges).__module__.startswith("cudf")
