from collections import Counter
import os

import pandas as pd
import pytest

import graphistry


ENGINES = [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1")),
]


def _graph(edge_rows, engine="pandas"):
    nodes = pd.DataFrame({
        "id": ["c1", "c2", "a", "b", "z"],
        "name": ["c1", "c2", "same", "other", "same"],
        "label__Country": [True, True, False, False, False],
        "label__City": [False, False, True, True, True],
    })
    edges = pd.DataFrame(edge_rows, columns=["s", "d", "type"])
    return _graph_frames(nodes, edges, engine)


def _graph_frames(nodes, edges, engine):
    if engine in {"polars", "polars-gpu"}:
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(graph, query, *, engine="pandas"):
    out = graph.gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return Counter(tuple(row) for row in out.itertuples(index=False, name=None))


_REENTRY_QUERY = (
    "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
    "WITH country, collect(city) AS cities "
    "MATCH (country)-[:CAND]->(candidate:City) "
    "WHERE candidate IN cities "
    "RETURN country.id AS countryId, candidate.id AS cityId"
)


@pytest.mark.parametrize(
    ("edges", "expected"),
    [
        ([ ("c1", "a", "HAS_CITY"), ("c1", "a", "CAND") ], Counter({("c1", "a"): 1})),
        ([ ("c1", "a", "HAS_CITY"), ("c1", "z", "CAND") ], Counter()),
        ([ ("c1", "a", "CAND") ], Counter()),
        ([ ("c1", "a", "HAS_CITY"), ("c1", "a", "HAS_CITY"), ("c1", "a", "CAND") ], Counter({("c1", "a"): 1})),
        ([
            ("c1", "a", "HAS_CITY"), ("c2", "b", "HAS_CITY"),
            ("c1", "b", "CAND"), ("c2", "a", "CAND"),
        ], Counter()),
    ],
    ids=["positive", "identity-collision", "empty", "duplicate-is-existential", "cross-group"],
)
@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_collected_node_membership_uses_identity(edges, expected, engine):
    assert _rows(_graph(edges, engine), _REENTRY_QUERY, engine=engine) == expected


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_direct_list_membership_and_unary_not_use_identity(engine):
    graph = _graph([
        ("a", "a", "CAND"),
        ("a", "z", "CAND"),
        ("a", "b", "CAND"),
    ])
    in_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a] "
        "RETURN a.id AS a, b.id AS b"
    )
    not_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN [a]) "
        "RETURN a.id AS a, b.id AS b"
    )
    empty_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [] "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, in_query, engine=engine) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query, engine=engine) == Counter({("a", "z"): 1, ("a", "b"): 1})
    assert _rows(graph, empty_query, engine=engine) == Counter()


@pytest.mark.parametrize(
    ("rhs", "expected_not"),
    [
        ("[a, null]", Counter()),
        ("[a, 1]", Counter({("a", "z"): 1, ("a", "b"): 1})),
        ("[a, 'x']", Counter({("a", "z"): 1, ("a", "b"): 1})),
        ("[a, 'z']", Counter({("a", "z"): 1, ("a", "b"): 1})),
    ],
)
@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_mixed_entity_lists_keep_identity_and_null_semantics(rhs, expected_not, engine):
    graph = _graph([
        ("a", "a", "CAND"),
        ("a", "z", "CAND"),
        ("a", "b", "CAND"),
    ])
    in_query = (
        f"MATCH (a:City)-[:CAND]->(b:City) WHERE b IN {rhs} "
        "RETURN a.id AS a, b.id AS b"
    )
    not_query = (
        f"MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN {rhs}) "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, in_query, engine=engine) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query, engine=engine) == expected_not


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_numeric_node_id_does_not_match_same_scalar_list_value(engine):
    nodes = pd.DataFrame({
        "id": [1, 2],
        "name": ["same", "same"],
        "label__City": [True, True],
    })
    edges = pd.DataFrame({"s": [1, 1], "d": [1, 2], "type": ["CAND", "CAND"]})
    graph = _graph_frames(nodes, edges, engine)
    in_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a, 2] "
        "RETURN a.id AS a, b.id AS b"
    )
    not_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN [a, 2]) "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, in_query, engine=engine) == Counter({(1, 1): 1})
    assert _rows(graph, not_query, engine=engine) == Counter({(1, 2): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_dynamic_scalar_and_null_rhs_items_preserve_tri_values(engine):
    graph = _graph([
        ("a", "a", "CAND"),
        ("a", "z", "CAND"),
        ("a", "b", "CAND"),
    ])
    dynamic_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a, b.name] "
        "RETURN a.id AS a, b.id AS b"
    )
    null_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN [a, b.missing]) "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, dynamic_query, engine=engine) == Counter({("a", "a"): 1})
    assert _rows(graph, null_query, engine=engine) == Counter()


@pytest.mark.parametrize(
    "entity_expr",
    [
        "head([a])",
        "head([a, null])",
        "head([a, a])",
        "head([a, 1])",
        "head([a, b.name])",
    ],
)
@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_nested_head_entity_membership_uses_identity(entity_expr, engine):
    graph = _graph([
        ("a", "a", "CAND"),
        ("a", "z", "CAND"),
    ])
    in_query = (
        f"MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [{entity_expr}] "
        "RETURN a.id AS a, b.id AS b"
    )
    not_query = (
        f"MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN [{entity_expr}]) "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, in_query, engine=engine) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query, engine=engine) == Counter({("a", "z"): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_generated_identity_carry_avoids_trailing_alias_collision(engine):
    graph = _graph([("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], engine)
    query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(__gfql_node_identity_list_0:City) "
        "WHERE __gfql_node_identity_list_0 IN cities "
        "RETURN country.id AS countryId, __gfql_node_identity_list_0.id AS candidateId"
    )
    assert _rows(graph, query, engine=engine) == Counter({("c1", "a"): 1})

    relationship_alias_query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[__gfql_node_identity_list_0:CAND]->(candidate:City) "
        "WHERE candidate IN cities "
        "RETURN country.id AS countryId, candidate.id AS candidateId"
    )
    assert _rows(graph, relationship_alias_query, engine=engine) == Counter({("c1", "a"): 1})


@pytest.mark.parametrize("engine", ENGINES)
def test_issue_2258_preserves_alias_and_property_equality_and_collect_display(engine):
    graph = _graph([
        ("a", "a", "CAND"),
        ("a", "z", "CAND"),
        ("a", "b", "CAND"),
        ("c1", "a", "HAS_CITY"),
        ("c1", "z", "HAS_CITY"),
    ])
    alias_eq = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE a = b "
        "RETURN a.id AS a, b.id AS b"
    )
    property_eq = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE a.name = b.name "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, alias_eq, engine=engine) == Counter({("a", "a"): 1})
    assert _rows(graph, property_eq, engine=engine) == Counter({("a", "a"): 1, ("a", "z"): 1})
    visible = graph.gfql("MATCH (n:City) RETURN collect(n) AS cities", engine=engine)._nodes
    if hasattr(visible, "to_pandas"):
        visible = visible.to_pandas()
    assert visible["cities"].iloc[0] == [
        "(:City {name: 'same'})",
        "(:City {name: 'other'})",
        "(:City {name: 'same'})",
    ]
    reentry_graph = _graph([("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")])
    visible_after_reentry = reentry_graph.gfql(
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(candidate:City) "
        "WHERE candidate IN cities "
        "RETURN cities, candidate.id AS candidateId"
    )._nodes
    if hasattr(visible_after_reentry, "to_pandas"):
        visible_after_reentry = visible_after_reentry.to_pandas()
    assert visible_after_reentry.to_dict(orient="records") == [{
        "cities": ["(:City {name: 'same'})"],
        "candidateId": "a",
    }]


def test_issue_2258_polars_retains_explicit_reentry_capability_decline():
    pytest.importorskip("polars")
    with pytest.raises(NotImplementedError, match="row op 'with_'"):
        _rows(
            _graph([("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")]),
            _REENTRY_QUERY,
            engine="polars",
        )
    with pytest.raises(NotImplementedError, match="row op 'where_rows'"):
        _rows(
            _graph([("a", "z", "CAND")]),
            "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a] RETURN a.id AS a, b.id AS b",
            engine="polars",
        )
