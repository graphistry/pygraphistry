"""Pin for #2019: a WHERE on one alias of a multi-alias MATCH filters that alias and projects
another, on pandas, cuDF and polars, plain and with indexes resident.

Still declined, each with its own ``field``: ORDER BY on a non-returned alias, and projecting the
edge alias from a node-alias WHERE (the open half of #2019).
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

ENGINES = ["pandas", "polars", "cudf"]

_NODES = pd.DataFrame({
    "id": ["a", "b", "c", "tx1", "tx2"],
    "type": ["person", "person", "company", "transaction", "transaction"],
    "score": [1, 2, 3, 4, 5],
})
_EDGES = pd.DataFrame({
    "src": ["a", "b", "a", "tx1", "tx2"],
    "dst": ["b", "c", "tx1", "tx2", "c"],
    "e_type": ["knows", "works_at", "sent", "transfer", "received"],
})


def _graphs(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "polars":
        pytest.importorskip("polars")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        pytest.importorskip("cupy")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    if engine == "cudf":
        return [("plain", g)]  # string ids cannot be indexed on cuDF, see the pin below
    return [("plain", g), ("indexed", g.gfql_index_all())]


def _ids(result):
    col = result._nodes["id"]
    return sorted(col.to_pandas().tolist() if hasattr(col, "to_pandas") else list(col))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query,expected", [
    # the issue's repro: filter on a, project t
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id", ["b", "c", "tx1"]),
    # the same with the edge alias, the destination alias, a comparison, and NOT
    ("MATCH (a)-[e]->(t) WHERE e.e_type IN ['sent','transfer'] RETURN t.id AS id", ["tx1", "tx2"]),
    ("MATCH (a)-[e]->(t) WHERE t.type IN ['transaction'] RETURN a.id AS id", ["a", "tx1"]),
    ("MATCH (a)-[e]->(t) WHERE a.score > 1 RETURN t.id AS id", ["c", "c", "tx2"]),
    ("MATCH (a)-[e]->(t) WHERE NOT (a.type IN ['person']) RETURN t.id AS id", ["c", "tx2"]),
    # projection shapes over the filtered pattern; DISTINCT collapses the duplicated c
    ("MATCH (a)-[e]->(t) WHERE a.score > 1 RETURN DISTINCT t.id AS id", ["c", "tx2"]),
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] WITH t RETURN t.id AS id", ["b", "c", "tx1"]),
    ("MATCH (a)-[e]->(t)-[f]->(u) WHERE a.type IN ['person','company'] RETURN u.id AS id", ["c", "tx2"]),
])
def test_single_alias_where_filters_that_alias_and_projects_another(engine, query, expected):
    for label, g in _graphs(engine):
        assert _ids(g.gfql(query, engine=engine)) == expected, (label, engine, query)


@pytest.mark.parametrize("engine", ENGINES)
def test_two_alias_where_still_works(engine):
    q = "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] AND e.e_type IN ['sent','transfer'] RETURN t.id AS id"
    for label, g in _graphs(engine):
        assert _ids(g.gfql(q, engine=engine)) == ["tx1"], (label, engine)


@pytest.mark.parametrize("engine", ENGINES)
def test_order_by_on_a_non_returned_alias_is_declined_as_order_by(engine):
    q = "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id ORDER BY a.id"
    for label, g in _graphs(engine):
        with pytest.raises(GFQLValidationError) as exc:
            g.gfql(q, engine=engine)
        assert exc.value.code == ErrorCode.E108 and exc.value.context["field"] == "order_by", (label, engine)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN e.e_type AS et",
    "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id, e.e_type AS et",
])
def test_projecting_the_edge_alias_declines_or_answers_the_frame_truth(engine, query):
    # #2019's open half, and the shape is not settled: it declines here and answered on CI
    # python 3.9, so the contract is pinned rather than the verdict.
    persons = set(_NODES.loc[_NODES["type"].isin(["person", "company"]), "id"])
    truth = sorted(_EDGES.loc[_EDGES["src"].isin(persons), "e_type"])
    for label, g in _graphs(engine):
        try:
            out = g.gfql(query, engine=engine)
        except GFQLValidationError as exc:
            assert exc.code == ErrorCode.E108 and exc.context["field"] == "where", (label, engine)
            continue
        col = out._nodes["et"] if "et" in out._nodes.columns else out._edges["et"]
        col = col.to_pandas() if hasattr(col, "to_pandas") else col
        assert sorted(col.tolist()) == truth, (label, engine)


def test_string_ids_cannot_be_indexed_on_cudf_yet():
    # measured on a GB10 with cudf 26.02: gfql_index_all() over a string-keyed cuDF graph raises a
    # raw `TypeError: cupy does not support object` instead of declining, so the pins above run
    # unindexed there. The plain query answers normally. Unchanged from master.
    cudf = pytest.importorskip("cudf")
    pytest.importorskip("cupy")
    g = graphistry.edges(cudf.from_pandas(_EDGES), "src", "dst").nodes(cudf.from_pandas(_NODES), "id")
    assert _ids(g.gfql("MATCH (a)-[e]->(t) WHERE t.type IN ['transaction'] RETURN a.id AS id", engine="cudf")) == ["a", "tx1"]
    with pytest.raises(TypeError, match="cupy does not support object"):
        g.gfql_index_all()
