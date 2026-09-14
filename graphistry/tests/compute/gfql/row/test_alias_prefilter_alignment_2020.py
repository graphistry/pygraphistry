"""Single-alias predicate pushdown keeps working after an earlier pushdown narrowed the frame (#2020)."""
import pandas as pd
import pytest

import graphistry

Q = ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] AND e.e_type IN ['sent','transfer'] "
     "AND t.type IN ['transaction','account'] RETURN t.id AS id")


def _graph(engine):
    nodes = pd.DataFrame({"id": ["a", "b", "c", "tx1", "tx2"],
                          "type": ["person", "person", "company", "transaction", "transaction"]})
    edges = pd.DataFrame({"src": ["a", "b", "a", "tx1", "tx2"], "dst": ["b", "c", "tx1", "tx2", "c"],
                          "e_type": ["knows", "works_at", "sent", "transfer", "received"]})
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _ids(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    return sorted(df["id"].tolist())


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_three_in_predicates_across_a_hop_agree_on_every_engine(engine):
    assert _ids(_graph(engine).gfql(Q, engine=engine)) == ["tx1"]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_pushdown_on_a_frame_with_filtered_labels_matches_the_scalar_form(engine):
    g = _graph(engine)
    scalar = Q.replace("t.type IN ['transaction','account']", "t.type = 'transaction'")
    assert _ids(g.gfql(Q, engine=engine)) == _ids(g.gfql(scalar, engine=engine)) == ["tx1"]


@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_pushdown_on_a_non_range_indexed_node_frame(engine):
    g = _graph("pandas")
    nodes = g._nodes.iloc[[4, 2, 0, 3, 1]]  # labels out of order, no RangeIndex
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes = cudf.from_pandas(nodes)
        g = _graph("cudf")
    g = g.nodes(nodes)
    assert _ids(g.gfql(Q, engine=engine)) == ["tx1"]
