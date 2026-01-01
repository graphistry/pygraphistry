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
        key="return3-1",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[1] Returning multiple expressions",
        cypher="MATCH (a)\nRETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b",
        graph=GraphFixture(
            nodes=[
                {"node_id": "n1", "labels": []},
            ],
            edges=[],
            node_id="node_id",
            node_columns=("node_id", "labels"),
        ),
        expected=Expected(
            rows=[
                {"a": "false", "b": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expression projections and null predicates are not supported",
        tags=("return", "expression", "null", "xfail"),
    ),

    Scenario(
        key="return3-2",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[2] Returning multiple node property values",
        cypher="MATCH (a)\nRETURN a.name, a.age, a.seasons",
        graph=GraphFixture(
            nodes=[
                {
                    "id": "n1",
                    "labels": [],
                    "name": "Philip J. Fry",
                    "age": 2046,
                    "seasons": [1, 2, 3, 4, 5, 6, 7],
                }
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {
                    "a.name": "'Philip J. Fry'",
                    "a.age": 2046,
                    "a.seasons": "[1, 2, 3, 4, 5, 6, 7]",
                },
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "xfail"),
    ),

    Scenario(
        key="return3-3",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[3] Projecting nodes and relationships",
        cypher="MATCH (a)-[r]->()\nRETURN a AS foo, r AS bar",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"foo": "(:A)", "bar": "[:T]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN projections of nodes and relationships are not supported",
        tags=("return", "projection", "xfail"),
    ),
]
