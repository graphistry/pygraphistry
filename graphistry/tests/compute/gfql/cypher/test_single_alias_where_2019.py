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
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
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
@pytest.mark.parametrize("query,field", [
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id ORDER BY a.id", "order_by"),
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN e.e_type AS et", "where"),
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id, e.e_type AS et", "where"),
])
def test_residuals_are_declined_at_their_own_field(engine, query, field):
    for label, g in _graphs(engine):
        with pytest.raises(GFQLValidationError) as exc:
            g.gfql(query, engine=engine)
        assert exc.value.code == ErrorCode.E108 and exc.value.context["field"] == field, (label, engine)
