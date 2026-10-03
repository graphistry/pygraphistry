"""Negated membership keeps its rows on a frame whose labels are not 0..n-1.

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
