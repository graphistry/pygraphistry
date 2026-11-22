import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError


def _nodes(engine):
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "type": ["P", "Q"],
        "label__Foo": [True, False],
        "name": ["x", "y"],
    })
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(nodes)
    if engine == "polars":
        import polars as pl

        return pl.from_pandas(nodes)
    return nodes


def _rows(frame):
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame.to_dict("records")


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


@pytest.mark.parametrize("key,expected", [
    ("type", ["P", "Q"]),
    ("id", ["a", "b"]),
    ("label__Foo", [True, False]),
    ("name", ["x", "y"]),
], ids=["type", "id", "label-flag", "property"])
def test_constant_string_subscript_matches_dotted_property(engine, key, expected):
    graph = graphistry.nodes(_nodes(engine), "id")
    rows = _rows(graph.gfql(
        f"MATCH (n) RETURN n['{key}'] AS v, n.{key} AS dotted",
        engine=engine,
    )._nodes)
    assert [(row["v"], row["dotted"]) for row in rows] == list(zip(expected, expected))


@pytest.mark.parametrize("key_expr,expected", [
    ("'nam' + 'e'", ["x", "y"]),
    ("$key", ["x", "y"]),
], ids=["constant-concatenation", "parameter"])
def test_nonliteral_constant_string_subscript_matches_dotted_property(engine, key_expr, expected):
    graph = graphistry.nodes(_nodes(engine), "id")
    rows = _rows(graph.gfql(
        f"MATCH (n) RETURN n[{key_expr}] AS v",
        engine=engine,
        params={"key": "name"},
    )._nodes)
    assert [row["v"] for row in rows] == expected


def _rows_or_native_decline(engine, graph, query):
    """Rows, or None when native Polars declines (parity-or-error); any other engine must answer."""
    try:
        return _rows(graph.gfql(query, engine=engine)._nodes)
    except (NotImplementedError, GFQLUnsupportedError):
        assert engine in {"polars", "polars-gpu"}, query
        return None


def _edge_graph(engine):
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "name": ["x", "y", "z"]})
    edges = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "c"], "eid": ["e1", "e2"], "kind": ["K1", "K2"]})
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst", edge="eid")


@pytest.mark.parametrize("bound_edge_id", [True, False], ids=["edge-id-bound", "edge-id-unbound"])
@pytest.mark.parametrize("alias,key,expected", [
    ("a", "name", ["x", "y"]),
    ("b", "name", ["y", "z"]),
    ("r", "kind", ["K1", "K2"]),
], ids=["source-node", "target-node", "relationship"])
def test_binding_row_string_subscripts_match_dotted_properties(engine, bound_edge_id, alias, key, expected):
    """A relationship pattern yields binding rows with one prefixed column group per alias; each reads its own group."""
    graph = _edge_graph(engine)
    if not bound_edge_id:
        graph = graph.bind(edge=None)
    results = {}
    for form, expr in [("subscript", f"{alias}['{key}']"), ("dotted", f"{alias}.{key}")]:
        rows = _rows_or_native_decline(engine, graph, f"MATCH (a)-[r]->(b) RETURN {expr} AS v")
        if rows is None:
            return
        results[form] = sorted(row["v"] for row in rows)
    assert results["subscript"] == results["dotted"] == expected


def test_binding_row_subscript_of_an_absent_key_is_null_like_the_dotted_property(engine):
    """pandas/cuDF return null for both spellings; native Polars may decline a spelling (parity-or-error), never answer differently."""
    graph = _edge_graph(engine)
    for expr in ("a['nope']", "a.nope"):
        try:
            rows = _rows(graph.gfql(f"MATCH (a)-[r]->(b) RETURN {expr} AS v", engine=engine)._nodes)
        except (NotImplementedError, GFQLUnsupportedError):
            assert engine in {"polars", "polars-gpu"}, expr
            continue
        assert len(rows) == 2
        assert all(pd.isna(row["v"]) for row in rows), expr


def test_cartesian_node_subscripts_read_their_own_alias(engine):
    """MATCH (a),(b) builds binding rows with one prefixed column group per node alias."""
    graph = _edge_graph(engine)
    for alias, expected in (("a", ["x", "x", "x", "y", "y", "y", "z", "z", "z"]), ("b", ["x", "x", "x", "y", "y", "y", "z", "z", "z"])):
        rows = _rows_or_native_decline(engine, graph, f"MATCH (a),(b) RETURN {alias}['name'] AS v")
        if rows is not None:
            assert sorted(row["v"] for row in rows) == expected, alias


def test_cartesian_node_subscript_of_an_absent_key_is_null(engine):
    rows = _rows_or_native_decline(engine, _edge_graph(engine), "MATCH (a),(b) RETURN a['nope'] AS v")
    if rows is not None:
        assert len(rows) == 9
        assert all(pd.isna(row["v"]) for row in rows)


def test_cross_alias_where_reads_node_and_relationship_subscripts(engine):
    """A cross-alias WHERE keeps binding rows for every alias, including the relationship's edge id."""
    graph = _edge_graph(engine)
    rows = _rows_or_native_decline(engine, graph, "MATCH (a)-[r]->(b) WHERE a['name'] <> b['name'] RETURN r['kind'] AS v")
    if rows is not None:
        assert sorted(row["v"] for row in rows) == ["K1", "K2"]
    rows = _rows_or_native_decline(engine, graph, "MATCH (a)-[r]->(b) WHERE a['name'] = 'x' RETURN b.name AS v")
    if rows is not None:
        assert [row["v"] for row in rows] == ["y"]
    rows = _rows_or_native_decline(engine, graph, "MATCH (a)-[r]->(b) WHERE a['nope'] = 'x' RETURN b.name AS v")
    if rows is not None:
        assert rows == []


def test_subscript_of_a_map_column_is_not_read_as_a_node_property(engine):
    """A map value named like nothing in the table, projected beside a node column, still answers from the map."""
    graph = _edge_graph(engine)
    for query, expected in (
        ("MATCH (n) WITH {name: 'z'} AS m, n.name AS name RETURN m['name'] AS v", ["z", "z", "z"]),
        ("MATCH (n) WITH n, {id: 9} AS m RETURN m['id'] AS v", [9, 9, 9]),
    ):
        rows = _rows_or_native_decline(engine, graph, query)
        if rows is not None:
            assert [row["v"] for row in rows] == expected, query


def test_subscript_of_an_optional_alias_is_null_where_the_alias_is_absent(engine):
    """Rows with no optional match give null for the subscript, exactly like the dotted spelling."""
    graph = _edge_graph(engine)
    results = {}
    for form, expr in (("subscript", "b['name']"), ("dotted", "b.name")):
        rows = _rows_or_native_decline(engine, graph, f"MATCH (a) OPTIONAL MATCH (a)-[r]->(b) RETURN {expr} AS v")
        if rows is None:
            return
        results[form] = sorted(("" if pd.isna(row["v"]) else row["v"]) for row in rows)
    assert results["subscript"] == results["dotted"] == ["", "y", "z"]
