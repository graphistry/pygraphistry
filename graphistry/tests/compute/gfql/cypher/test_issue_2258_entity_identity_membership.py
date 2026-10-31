from collections import Counter

import pandas as pd
import pytest

import graphistry


def _graph(edge_rows, engine="pandas"):
    nodes = pd.DataFrame({
        "id": ["c1", "c2", "a", "b", "z"],
        "name": ["c1", "c2", "same", "other", "same"],
        "label__Country": [True, True, False, False, False],
        "label__City": [False, False, True, True, True],
    })
    edges = pd.DataFrame(edge_rows, columns=["s", "d", "type"])
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(graph, query, *, engine="pandas"):
    out = graph.gfql(query, engine=engine)._nodes
    if engine == "polars":
        return Counter(tuple(row[column] for column in out.columns) for row in out.to_dicts())
    return Counter(tuple(row) for row in out.itertuples(index=False, name=None))


_REENTRY_QUERY = (
    "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
    "WITH country, collect(city) AS cities "
    "MATCH (country)-[:CAND]->(candidate:City) "
    "WHERE candidate IN cities "
    "RETURN country.id AS countryId, candidate.id AS cityId"
)


@pytest.mark.parametrize(
    ("engine", "edges", "expected"),
    [
        ("pandas", [("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], Counter({("c1", "a"): 1})),
        ("polars", [("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], Counter({("c1", "a"): 1})),
        ("pandas", [("c1", "a", "HAS_CITY"), ("c1", "z", "CAND")], Counter()),
        ("polars", [("c1", "a", "HAS_CITY"), ("c1", "z", "CAND")], Counter()),
        ("pandas", [("c1", "a", "CAND")], Counter()),
        ("polars", [("c1", "a", "CAND")], Counter()),
        ("pandas", [("c1", "a", "HAS_CITY"), ("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], Counter({("c1", "a"): 1})),
        ("polars", [("c1", "a", "HAS_CITY"), ("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], Counter({("c1", "a"): 1})),
        ("pandas", [
            ("c1", "a", "HAS_CITY"), ("c2", "b", "HAS_CITY"),
            ("c1", "b", "CAND"), ("c2", "a", "CAND"),
        ], Counter()),
        ("polars", [
            ("c1", "a", "HAS_CITY"), ("c2", "b", "HAS_CITY"),
            ("c1", "b", "CAND"), ("c2", "a", "CAND"),
        ], Counter()),
    ],
    ids=["pandas-positive", "polars-positive", "pandas-identity-collision", "polars-identity-collision",
         "pandas-empty", "polars-empty", "pandas-duplicate-is-existential", "polars-duplicate-is-existential",
         "pandas-cross-group", "polars-cross-group"],
)
def test_issue_2258_collected_node_membership_uses_identity(engine, edges, expected):
    assert _rows(_graph(edges, engine), _REENTRY_QUERY, engine=engine) == expected


def test_issue_2258_direct_list_membership_and_unary_not_use_identity():
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
    assert _rows(graph, in_query) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query) == Counter({("a", "z"): 1, ("a", "b"): 1})
    assert _rows(graph, empty_query) == Counter()


@pytest.mark.parametrize(
    ("rhs", "expected_not"),
    [
        ("[a, null]", Counter()),
        ("[a, 1]", Counter({("a", "z"): 1, ("a", "b"): 1})),
        ("[a, 'x']", Counter({("a", "z"): 1, ("a", "b"): 1})),
        ("[a, 'z']", Counter({("a", "z"): 1, ("a", "b"): 1})),
    ],
)
def test_issue_2258_mixed_entity_lists_keep_identity_and_null_semantics(rhs, expected_not):
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
    assert _rows(graph, in_query) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query) == expected_not


def test_issue_2258_numeric_node_id_does_not_match_same_scalar_list_value():
    nodes = pd.DataFrame({
        "id": [1, 2],
        "name": ["same", "same"],
        "label__City": [True, True],
    })
    edges = pd.DataFrame({"s": [1, 1], "d": [1, 2], "type": ["CAND", "CAND"]})
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    in_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a, 2] "
        "RETURN a.id AS a, b.id AS b"
    )
    not_query = (
        "MATCH (a:City)-[:CAND]->(b:City) WHERE NOT (b IN [a, 2]) "
        "RETURN a.id AS a, b.id AS b"
    )
    assert _rows(graph, in_query) == Counter({(1, 1): 1})
    assert _rows(graph, not_query) == Counter({(1, 2): 1})


def test_issue_2258_dynamic_scalar_and_null_rhs_items_preserve_tri_values():
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
    assert _rows(graph, dynamic_query) == Counter({("a", "a"): 1})
    assert _rows(graph, null_query) == Counter()


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
def test_issue_2258_nested_head_entity_membership_uses_identity(entity_expr):
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
    assert _rows(graph, in_query) == Counter({("a", "a"): 1})
    assert _rows(graph, not_query) == Counter({("a", "z"): 1})


def test_issue_2258_generated_identity_carry_avoids_trailing_alias_collision():
    graph = _graph([("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")])
    query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(__gfql_node_identity_list_0:City) "
        "WHERE __gfql_node_identity_list_0 IN cities "
        "RETURN country.id AS countryId, __gfql_node_identity_list_0.id AS candidateId"
    )
    assert _rows(graph, query) == Counter({("c1", "a"): 1})

    relationship_alias_query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[__gfql_node_identity_list_0:CAND]->(candidate:City) "
        "WHERE candidate IN cities "
        "RETURN country.id AS countryId, candidate.id AS candidateId"
    )
    assert _rows(graph, relationship_alias_query) == Counter({("c1", "a"): 1})


def test_issue_2258_preserves_alias_and_property_equality_and_collect_display():
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
    assert _rows(graph, alias_eq) == Counter({("a", "a"): 1})
    assert _rows(graph, property_eq) == Counter({("a", "a"): 1, ("a", "z"): 1})
    visible = graph.gfql("MATCH (n:City) RETURN collect(n) AS cities")._nodes
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
    assert visible_after_reentry.to_dict(orient="records") == [{
        "cities": ["(:City {name: 'same'})"],
        "candidateId": "a",
    }]


def test_issue_2258_polars_serves_collected_identity_and_keeps_direct_list_decline():
    pytest.importorskip("polars")
    graph = _graph([("c1", "a", "HAS_CITY"), ("c1", "a", "CAND")], "polars")
    result = graph.gfql(_REENTRY_QUERY, engine="polars")._nodes
    assert _rows(graph, _REENTRY_QUERY, engine="polars") == Counter({("c1", "a"): 1})
    assert not any(str(column).startswith(("__gfql_", "__cypher_reentry_")) for column in result.columns)
    visible = graph.gfql(
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(candidate:City) WHERE candidate IN cities "
        "RETURN cities, candidate.id AS cityId",
        engine="polars",
    )._nodes.to_dicts()
    assert visible == [{"cities": ["(:City {name: 'same'})"], "cityId": "a"}]
    not_graph = _graph([
        ("c1", "a", "HAS_CITY"),
        ("c1", "a", "CAND"),
        ("c1", "z", "CAND"),
    ], "polars")
    not_query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(candidate:City) WHERE NOT (candidate IN cities) "
        "RETURN candidate.id AS cityId"
    )
    assert not_graph.gfql(not_query, engine="polars")._nodes.to_dicts() == [{"cityId": "z"}]
    with pytest.raises(NotImplementedError, match="row op 'where_rows'"):
        _rows(
            _graph([("a", "z", "CAND")], "polars"),
            "MATCH (a:City)-[:CAND]->(b:City) WHERE b IN [a] RETURN a.id AS a, b.id AS b",
            engine="polars",
        )


def test_issue_2258_polars_collected_identity_keeps_cycle_bag_and_collision_negative():
    pl = pytest.importorskip("polars")
    nodes = pd.DataFrame({
        "id": ["c1", "z"],
        "name": ["same", "same"],
        "label__Country": [True, False],
        "label__City": [True, True],
    })
    edges = pd.DataFrame({
        "s": ["c1"] * 5,
        "d": ["c1", "c1", "c1", "c1", "z"],
        "type": ["HAS_CITY", "HAS_CITY", "CAND", "CAND", "CAND"],
    })
    graph = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(candidate:City) WHERE candidate IN cities "
        "RETURN cities, candidate.id AS cityId"
    )
    rows = graph.gfql(query, engine="polars")._nodes.to_dicts()
    expected_cities = ["(:Country:City {name: 'same'})"] * 2
    assert rows == [
        {"cities": expected_cities, "cityId": "c1"},
        {"cities": expected_cities, "cityId": "c1"},
    ]
    assert all(not any(str(column).startswith(("__gfql_", "__cypher_reentry_")) for column in row) for row in rows)


def test_issue_2258_polars_collected_identity_membership_keeps_null_as_unknown():
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.lazy.engine.polars.row_pipeline import _lower_with_schema, lower_expr_str

    table = pl.DataFrame({
        "candidate": ["a", None, "z"],
        "country": ["c1", "c1", "c1"],
        "country.__cypher_reentry___gfql_node_identity_list_0__": [["a"], ["a"], ["a"]],
    })
    expr = _lower_with_schema(
        table,
        lambda: lower_expr_str(
            "candidate.__gfql_node_id__ IN country.__cypher_reentry___gfql_node_identity_list_0__",
            table.columns,
        ),
        node_id="candidate",
    )
    assert expr is not None
    assert table.select(expr.alias("member")).get_column("member").to_list() == [True, None, False]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    ("same_name_value", "expected_entity"),
    [
        (None, "(:City {name: 'same'})"),
        ("inner", "(:City {name: 'same', city: 'inner'})"),
    ],
    ids=["same-name-null", "same-name-non-null"],
)
def test_issue_2258_collected_entity_same_name_property_keeps_presence_and_identity(
    engine, same_name_value, expected_entity,
):
    if engine == "polars":
        pl = pytest.importorskip("polars")
    nodes = pd.DataFrame({
        "id": ["c1", "a", "z"],
        "name": ["country", "same", "same"],
        # This property deliberately has the same spelling as the `city` node alias.
        "city": ["outer", same_name_value, same_name_value],
        "label__Country": [True, False, False],
        "label__City": [False, True, True],
    })
    edges = pd.DataFrame({
        "s": ["c1", "c1", "c1", "c1"],
        "d": ["a", "a", "a", "z"],
        "type": ["HAS_CITY", "CAND", "CAND", "CAND"],
    })
    if engine == "polars":
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "WITH country, collect(city) AS cities "
        "MATCH (country)-[:CAND]->(candidate:City) WHERE candidate IN cities "
        "RETURN cities, candidate.id AS cityId"
    )
    result = graph.gfql(query, engine=engine)._nodes
    rows = result.to_dicts() if engine == "polars" else result.to_dict(orient="records")
    expected = [
        {"cities": [expected_entity], "cityId": "a"},
        {"cities": [expected_entity], "cityId": "a"},
    ]
    assert rows == expected
    assert all(
        not any(str(column).startswith(("__gfql_", "__cypher_reentry_")) for column in row)
        for row in rows
    )

    direct_nodes = pd.DataFrame({
        "id": ["a", "b"],
        "name": ["same", "same"],
        "n": [same_name_value, same_name_value],
        "label__City": [True, True],
    })
    if engine == "polars":
        direct_nodes = pl.from_pandas(direct_nodes)
    direct_graph = graphistry.nodes(direct_nodes, "id")
    direct_query = "MATCH (n:City) RETURN collect(n) AS xs"
    direct_result = direct_graph.gfql(direct_query, engine=engine)._nodes
    direct_rows = (
        direct_result.to_dicts()
        if engine == "polars"
        else direct_result.to_dict(orient="records")
    )
    expected_direct_entity = (
        "(:City {name: 'same'})"
        if same_name_value is None
        else f"(:City {{name: 'same', n: '{same_name_value}'}})"
    )
    assert direct_rows == [{"xs": [expected_direct_entity, expected_direct_entity]}]
    noncolliding_alias_rows = (
        direct_graph.gfql("MATCH (m:City) RETURN collect(m) AS xs", engine=engine)._nodes
    )
    noncolliding_rows = (
        noncolliding_alias_rows.to_dicts()
        if engine == "polars"
        else noncolliding_alias_rows.to_dict(orient="records")
    )
    assert noncolliding_rows == [{"xs": [expected_direct_entity, expected_direct_entity]}]

    # Ordinary alias/property access remains mapped to its actual prefixed property column.
    property_query = (
        "MATCH (country:Country)-[:HAS_CITY]->(city:City) "
        "RETURN country.city AS countryCity, city.name AS cityName"
    )
    property_result = graph.gfql(property_query, engine=engine)._nodes
    property_rows = (
        property_result.to_dicts()
        if engine == "polars"
        else property_result.to_dict(orient="records")
    )
    assert property_rows == [{"countryCity": "outer", "cityName": "same"}]
