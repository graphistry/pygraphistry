from graphistry.compute import e_forward, e_undirected, n

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.parse_cypher import graph_fixture_from_create

from tests.cypher_tck.scenarios.fixtures import (
    MATCH5_GRAPH,
    MATCH7_GRAPH_SINGLE,
    MATCH7_GRAPH_AB,
    MATCH7_GRAPH_ABC,
    MATCH7_GRAPH_REL,
    MATCH7_GRAPH_X,
    MATCH7_GRAPH_AB_X,
    MATCH7_GRAPH_LABELS,
    MATCH7_GRAPH_PLAYER_TEAM_BOTH,
    MATCH7_GRAPH_PLAYER_TEAM_SINGLE,
    MATCH7_GRAPH_PLAYER_TEAM_DIFF,
    WITH_ORDERBY4_GRAPH,
    BINARY_TREE_1_GRAPH,
    BINARY_TREE_2_GRAPH,
)


SCENARIOS = [
    Scenario(
        key="with-where1-1",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[1] Filter node with property predicate on a single variable with multiple bindings",
        cypher="MATCH (a)\nWITH a\nWHERE a.name = 'B'\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'B'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and WHERE filtering are not supported",
        tags=("with", "where", "xfail"),
    ),

    Scenario(
        key="with-where1-2",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[2] Filter node with property predicate on a single variable with multiple distinct bindings",
        cypher="MATCH (a)\nWITH DISTINCT a.name2 AS name\nWHERE a.name2 = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'A'}),
                   ({name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections and WHERE filtering are not supported",
        tags=("with", "distinct", "where", "xfail"),
    ),

    Scenario(
        key="with-where1-3",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[3] Filter for an unbound relationship variable",
        cypher="MATCH (a:A), (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE r IS NULL\nRETURN other",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"other": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),

    Scenario(
        key="with-where1-4",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[4] Filter for an unbound node variable",
        cypher="MATCH (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE a IS NULL\nRETURN other",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"other": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),
]
