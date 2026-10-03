"""Row-evaluator results stay row-aligned on a frame whose labels are not 0..n-1.

#2020 made the positive-IN pushdown apply its mask by position. ``NOT (x IN [...])`` goes
through the tri-valued NOT builder instead, which broadcasts on the table's labels and then
``.where``-aligns against positionally built masks; on a frame the index path hands back
(labels with gaps), the two disagreed and whole destination ids vanished. Every row-values
series now carries the table's own index, so both conventions agree on any frame.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

Q_NOT = "MATCH (a)-[e]->(t) WHERE NOT (t.type IN ['person', 'company']) RETURN t.id AS id"
Q_EQ = "MATCH (a)-[e]->(t) WHERE t.type <> 'person' AND t.type <> 'company' RETURN t.id AS id"


def _graph(engine):
    nodes = pd.DataFrame({"id": ["a", "b", "c", "tx1", "tx2"],
                          "type": ["person", "person", "company", "transaction", "transaction"]})
    edges = pd.DataFrame({"src": ["a", "b", "a", "tx1", "tx2"], "dst": ["b", "c", "tx1", "tx2", "c"]})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _ids(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    return sorted(df["id"].tolist())


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_not_in_on_a_non_range_indexed_node_frame_matches_the_scalar_form(engine):
    g = _graph("pandas")
    nodes = g._nodes.iloc[[4, 2, 0, 3, 1]]  # labels out of order, no RangeIndex
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes = cudf.from_pandas(nodes)
        g = _graph("cudf")
    g = g.nodes(nodes)
    assert _ids(g.gfql(Q_NOT, engine=engine)) == _ids(g.gfql(Q_EQ, engine=engine)) == ["tx1", "tx2"]


def test_not_in_agrees_across_index_policies_on_the_index_path():
    # 30 nodes / 120 edges, seed 3001: the index path's hop returns a nodes frame whose
    # labels skip 5, and the forced lane dropped destinations 9, 28 and 29 before the fix.
    rng = np.random.default_rng(3001)
    n_nodes, n_edges = 30, 120
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    g = graphistry.edges(edges, "src", "dst").nodes(pd.DataFrame({"id": np.arange(n_nodes)}), "id").gfql_index_all()
    q = "MATCH (a)-[e]->(b) WHERE NOT (b.id IN [3, 5, 10]) RETURN b.id AS id"
    truth = int((~edges["dst"].isin([3, 5, 10])).sum())
    assert truth == 111
    for policy in ("off", "use", "force"):
        assert len(g.gfql(q, engine="pandas", index_policy=policy)._nodes) == truth, policy


def test_row_values_series_carries_the_table_index():
    table = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 2, 7])  # gaps, like an index-path hop result
    out = RowPipelineMixin._gfql_series_from_row_values(RowPipelineMixin(), table, [True, False, None], "__t__")
    assert list(out.index) == [0, 2, 7]
    assert list(out) == [True, False, None]


def _gappy_graph():
    rng = np.random.default_rng(3001)
    n_nodes, n_edges = 30, 120
    edges = pd.DataFrame({"src": rng.integers(0, n_nodes, n_edges), "dst": rng.integers(0, n_nodes, n_edges)})
    nodes = pd.DataFrame({"id": np.arange(n_nodes),
                          "ts": pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n_nodes), unit="D")})
    return edges, graphistry.edges(edges, "src", "dst").nodes(nodes, "id").gfql_index_all()


@pytest.mark.parametrize("label,query,truth_mask", [
    ("temporal IN", "MATCH (a)-[e]->(b) WHERE b.ts IN [datetime('2024-01-10T00:00:00')] RETURN b.id AS id", lambda d: d == 9),
    ("NOT temporal IN", "MATCH (a)-[e]->(b) WHERE NOT (b.ts IN [datetime('2024-01-10T00:00:00')]) RETURN b.id AS id", lambda d: d != 9),
    ("list =", "MATCH (a)-[e]->(b) WHERE [b.id] = [9] RETURN b.id AS id", lambda d: d == 9),
    ("list <>", "MATCH (a)-[e]->(b) WHERE [b.id] <> [9] RETURN b.id AS id", lambda d: d != 9),
    ("list <", "MATCH (a)-[e]->(b) WHERE [b.id] < [10] RETURN b.id AS id", lambda d: d < 10),
])
def test_evaluator_families_agree_across_index_policies_on_the_index_path(label, query, truth_mask):
    # Temporal IN and list comparison build a reset work frame and insert evaluator series into
    # it; before the fix the temporal path kept node 10 instead of node 9 under 'force'.
    edges, g = _gappy_graph()
    truth = int(truth_mask(edges["dst"]).sum())
    for policy in ("off", "use", "force"):
        assert len(g.gfql(query, engine="pandas", index_policy=policy)._nodes) == truth, (label, policy)


def test_positional_assign_keeps_rows_in_place_on_a_gappy_series():
    frame = pd.DataFrame({"x": range(5)})                       # RangeIndex 0..4 (a reset work frame)
    gappy = pd.Series(["v0", "v1", "v2", "v3", "v4"], index=[0, 1, 2, 4, 7])  # labels with gaps
    out = RowPipelineMixin._gfql_assign_positional(frame, c=gappy, k=range(5))
    assert list(out["c"]) == ["v0", "v1", "v2", "v3", "v4"]     # no NaN at the gap, no tail lost
    assert list(out["k"]) == [0, 1, 2, 3, 4]
    # a length mismatch is left to pandas' own alignment rather than silently truncated
    short = pd.Series([1, 2], index=[0, 1])
    assert out.shape[0] == RowPipelineMixin._gfql_assign_positional(frame, s=short).shape[0]


def test_on_table_index_leaves_unalignable_values_alone():
    table = pd.DataFrame({"x": [1, 2, 3]}, index=[0, 2, 7])
    assert RowPipelineMixin._gfql_on_table_index(table, 5) == 5                       # scalar: no set_axis
    short = pd.Series([True, False])
    assert list(RowPipelineMixin._gfql_on_table_index(table, short).index) == [0, 1]  # length mismatch: untouched
    full = pd.Series([True, False, True])
    assert list(RowPipelineMixin._gfql_on_table_index(table, full).index) == [0, 2, 7]
