"""Regression pin for #2019: a WHERE that touches ONE alias of a multi-alias MATCH lowers as
that alias's filter and projects another alias.

The issue's exact repro (``MATCH (a)-[e]->(t) WHERE a.type IN [...] RETURN t.id``) does not
raise at its own stated commit 3fb216dd or at master in this suite's environment; this pin
keeps the shape working on every engine, plain and indexed, and keeps the two residuals that
SHOULD still be rejected rejected with their own messages (not as "multi-source").
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars  # noqa: F401
    _HAS_POLARS = True
except Exception:  # pragma: no cover - depends on test env
    _HAS_POLARS = False

_ENGINES = ["pandas"] + (["polars"] if _HAS_POLARS else [])

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


def _graphs():
    g = graphistry.edges(_EDGES, "src", "dst").nodes(_NODES, "id")
    return [("plain", g), ("indexed", g.gfql_index_all())]


def _ids(result):
    col = result._nodes["id"]
    return sorted(col.to_list() if hasattr(col, "to_list") else list(col))


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query,expected", [
    # the issue's repro: filter on a, project t
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id", ["b", "c", "tx1"]),
    # the same with the edge alias, the destination alias, a comparison, and NOT
    ("MATCH (a)-[e]->(t) WHERE e.e_type IN ['sent','transfer'] RETURN t.id AS id", ["tx1", "tx2"]),
    ("MATCH (a)-[e]->(t) WHERE t.type IN ['transaction'] RETURN a.id AS id", ["a", "tx1"]),
    ("MATCH (a)-[e]->(t) WHERE a.score > 1 RETURN t.id AS id", ["c", "c", "tx2"]),
    ("MATCH (a)-[e]->(t) WHERE NOT (a.type IN ['person']) RETURN t.id AS id", ["c", "tx2"]),
    # projection shapes over the filtered pattern
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN DISTINCT t.id AS id", ["b", "c", "tx1"]),
    ("MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] WITH t RETURN t.id AS id", ["b", "c", "tx1"]),
    ("MATCH (a)-[e]->(t)-[f]->(u) WHERE a.type IN ['person','company'] RETURN u.id AS id", ["c", "tx2"]),
])
def test_single_alias_where_filters_that_alias_and_projects_another(engine, query, expected):
    for label, g in _graphs():
        assert _ids(g.gfql(query, engine=engine)) == expected, (label, engine, query)


@pytest.mark.parametrize("engine", _ENGINES)
def test_two_alias_where_still_works(engine):
    q = "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] AND e.e_type IN ['sent','transfer'] RETURN t.id AS id"
    for label, g in _graphs():
        assert _ids(g.gfql(q, engine=engine)) == ["tx1"], (label, engine)


@pytest.mark.parametrize("engine", _ENGINES)
def test_order_by_on_a_non_returned_alias_is_rejected_as_order_by_not_multi_source(engine):
    q = "MATCH (a)-[e]->(t) WHERE a.type IN ['person','company'] RETURN t.id AS id ORDER BY a.id"
    g = _graphs()[0][1]
    with pytest.raises(GFQLValidationError, match="ORDER BY expressions must reference"):
        g.gfql(q, engine=engine)
