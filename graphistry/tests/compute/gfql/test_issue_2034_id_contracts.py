"""Current-master contracts for node-ID multiplicity and null linking (#2034).

The full pipeline now serves these rows positionally. These pins keep both the
seeded fast path and the general path on the same openCypher row contract.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.index import index_trace
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off

ENGINES = ["pandas", "polars", "cudf"]


def _canon(result):
    df = result._nodes
    df = df.to_pandas() if hasattr(df, "to_pandas") else pd.DataFrame(df)
    df.columns = [str(column) for column in df.columns]
    columns = sorted(df.columns)
    return df.sort_values(columns).reset_index(drop=True)[columns] if columns else df


def _run(graph, query, engine, *, full=False):
    if full:
        with routes_off(ROUTES):
            return _canon(graph.gfql(query, engine=engine))
    else:
        return _canon(graph.gfql(query, engine=engine))


def _assert_parity(graph, query, engine, expected):
    seeded = _run(graph, query, engine, full=False)
    general = _run(graph, query, engine, full=True)
    assert len(seeded) == len(expected)
    assert seeded.equals(general)
    return seeded


def _graph(engine, *, ids, names, ages, edge_srcs, edge_dsts):
    nodes = pd.DataFrame({"id": ids, "name": names, "age": ages})
    edges = pd.DataFrame({
        "src": pd.Series(edge_srcs, dtype=nodes["id"].dtype),
        "dst": pd.Series(edge_dsts, dtype=nodes["id"].dtype),
        "type": pd.Series(["KNOWS"] * len(edge_srcs), dtype="str"),
    })
    if engine == "polars":
        polars = pytest.importorskip("polars")
        nodes, edges = polars.from_pandas(nodes), polars.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst")


@pytest.mark.parametrize("engine", ENGINES)
def test_node_lookup_duplicate_and_null_ids_follow_the_row_contract(engine):
    duplicate_graph = _graph(
        engine,
        ids=[7, 7, 8],
        names=["a", "b", "c"],
        ages=[1, 2, 3],
        edge_srcs=[],
        edge_dsts=[],
    )
    out = _assert_parity(
        duplicate_graph,
        "MATCH (p {id: 7}) RETURN p.age AS age",
        engine,
        [(1,), (2,)],
    )
    assert out["age"].tolist() == [1, 2]

    null_graph = _graph(
        engine,
        ids=pd.Series([None, None, 8], dtype="Int64"),
        names=["n3", "other", "other"],
        ages=[4, 5, 6],
        edge_srcs=[],
        edge_dsts=[],
    )
    out = _assert_parity(
        null_graph,
        "MATCH (p {name: 'n3'}) RETURN p.id AS pid, p.age AS age",
        engine,
        [(None, 4)],
    )
    assert out["pid"].isna().all()
    assert out["age"].tolist() == [4]

    with index_trace():
        out = _run(null_graph, "MATCH (p {id: null}) RETURN p.id AS pid", engine)
    assert out.empty


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_typed_hop_trace_reports_duplicate_decline_and_unique_serve(engine):
    duplicate_graph = _graph(
        engine,
        ids=[7, 7, 8],
        names=["a", "b", "c"],
        ages=[1, 2, 3],
        edge_srcs=[7, 7],
        edge_dsts=[8, 8],
    )
    query = "MATCH (p {id: 7})-[:KNOWS]->(d) RETURN d.age AS age"
    _assert_parity(duplicate_graph, query, engine, [(3,), (3,), (3,), (3,)])
    with index_trace() as steps:
        duplicate_graph.gfql(query, engine=engine)
    assert any(
        step.get("op") == "fast_path"
        and step.get("seam") == "seeded_typed_hop"
        and step.get("served") is False
        for step in steps
    ), steps

    unique_graph = _graph(
        engine,
        ids=[7, 8],
        names=["a", "b"],
        ages=[1, 2],
        edge_srcs=[7],
        edge_dsts=[8],
    )
    with index_trace() as steps:
        out = unique_graph.gfql(query, engine=engine)
    assert _canon(out)["age"].tolist() == [2]
    assert any(
        step.get("op") == "fast_path"
        and step.get("seam") == "seeded_typed_hop"
        and step.get("served") is True
        for step in steps
    ), steps


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_typed_hop_duplicate_seed_rows_match_the_general_path(engine):
    duplicate_graph = _graph(
        engine,
        ids=[7, 7, 8],
        names=["a", "b", "c"],
        ages=[1, 2, 3],
        edge_srcs=[7, 7],
        edge_dsts=[8, 8],
    )
    out = _assert_parity(
        duplicate_graph,
        "MATCH (p {id: 7})-[:KNOWS]->(d) RETURN d.age AS age",
        engine,
        [(3,), (3,), (3,), (3,)],
    )
    assert out["age"].tolist() == [3, 3, 3, 3]

    reverse_graph = _graph(
        engine,
        ids=[7, 7, 8],
        names=["a", "b", "c"],
        ages=[1, 2, 3],
        edge_srcs=[8, 8],
        edge_dsts=[7, 7],
    )
    out = _assert_parity(
        reverse_graph,
        "MATCH (p {id: 7})<-[:KNOWS]-(d) RETURN d.age AS age",
        engine,
        [(1,), (1,), (1,), (1,)],
    )
    assert out["age"].tolist() == [3, 3, 3, 3]

    unique_graph = _graph(
        engine,
        ids=[7, 8],
        names=["a", "b"],
        ages=[1, 2],
        edge_srcs=[7],
        edge_dsts=[8],
    )
    _assert_parity(
        unique_graph,
        "MATCH (p {id: 7})-[:KNOWS]->(d) RETURN d.age AS age",
        engine,
        [(2,)],
    )

    no_edge_graph = _graph(
        engine,
        ids=[7, 8],
        names=["a", "b"],
        ages=[1, 2],
        edge_srcs=[],
        edge_dsts=[],
    )
    _assert_parity(
        no_edge_graph,
        "MATCH (p {id: 7})-[:KNOWS]->(d) RETURN d.age AS age",
        engine,
        [],
    )


@pytest.mark.parametrize("engine", ENGINES)
def test_seeded_typed_hop_null_seed_ids_do_not_link(engine):
    null_seed_graph = _graph(
        engine,
        ids=pd.Series([None, 8], dtype="Int64"),
        names=["n0", "b"],
        ages=[1, 2],
        edge_srcs=[None],
        edge_dsts=[8],
    )
    _assert_parity(
        null_seed_graph,
        "MATCH (p {name: 'n0'})-[:KNOWS]->(d) RETURN d.age AS age",
        engine,
        [],
    )
