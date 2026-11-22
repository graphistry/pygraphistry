import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError, GFQLValidationError


def _engine_frame(engine):
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "tags": ["left", "right"],
        "shared": ["n-left", "n-right"],
    })
    edges = pd.DataFrame({
        "s": ["a", "b"],
        "d": ["b", "a"],
        "type": ["KNOWS", "KNOWS"],
        "weight": [5, 7],
        "shared": ["e-left", "e-right"],
    })
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return nodes, edges


def _records(frame):
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


def test_keys_and_properties_of_node_alias_ignore_other_bindings(engine):
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    query = "MATCH (n),(m) RETURN DISTINCT keys(n) AS k, properties(n) AS p"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _records(graph.gfql(query, engine=engine)._nodes)
    assert rows == [
        {"k": ["tags", "shared"], "p": "{tags: 'left', shared: 'n-left'}"},
        {"k": ["tags", "shared"], "p": "{tags: 'right', shared: 'n-right'}"},
    ]


def test_keys_and_properties_of_relationship_alias_ignore_other_bindings(engine):
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")

    query = "MATCH (n)-[r:KNOWS]->(m) RETURN DISTINCT keys(r) AS k, properties(r) AS p"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return

    rows = _records(graph.gfql(query, engine=engine)._nodes)
    assert rows == [
        {"k": ["weight", "shared"], "p": "{weight: 5, shared: 'e-left'}"},
        {"k": ["weight", "shared"], "p": "{weight: 7, shared: 'e-right'}"},
    ]


@pytest.mark.parametrize("query", [
    "MATCH (n),(k) WITH n, k, {a: 1} AS m RETURN properties(m) AS v",
    "MATCH (n) WITH n.tags AS t RETURN properties(t) AS v",
], ids=["map-after-two-alias-with", "string-column"])
def test_properties_of_a_non_alias_is_never_read_as_a_node(engine, query):
    """Only a real alias may be rebuilt as an entity: a map or plain column raises (as before) instead of answering with node properties."""
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    with pytest.raises((GFQLValidationError, NotImplementedError)):
        graph.gfql(query, engine=engine)


def test_properties_of_a_map_beside_a_node_alias_is_the_map(engine):
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return
    assert [row["v"] for row in _records(graph.gfql(query, engine=engine)._nodes)] == ["{a: 1}", "{a: 1}"]


def test_multi_alias_metadata_keeps_one_row_per_binding(engine):
    """Forcing binding rows must not collapse or multiply bindings when DISTINCT is absent."""
    nodes, edges = _engine_frame(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    cases = [
        ("MATCH (n),(m) RETURN keys(n) AS v", [["tags", "shared"]] * 4),
        ("MATCH (n)-[r]->(m) RETURN properties(n) AS v", ["{tags: 'left', shared: 'n-left'}", "{tags: 'right', shared: 'n-right'}"]),
    ]
    for query, expected in cases:
        if engine in {"polars", "polars-gpu"}:
            with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
                graph.gfql(query, engine=engine)
            continue
        assert [row["v"] for row in _records(graph.gfql(query, engine=engine)._nodes)] == expected, query
