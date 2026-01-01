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
        key="with3-1",
        feature_path="tck/features/clauses/with/With3.feature",
        scenario="[1] Forwarding multiple node and relationship variables",
        cypher="MATCH (a)-[r]->(b:X)\nWITH a, r, b\nMATCH (a)-[r]->(b)\nRETURN r AS rel\n  ORDER BY rel.id",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {id: 0}]->(:X),
                   ()-[:T2 {id: 1}]->(:X),
                   ()-[:T2 {id: 2}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1 {id: 0}]"},
                {"rel": "[:T2 {id: 1}]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, and relationship projections are not supported",
        tags=("with", "orderby", "relationship", "xfail"),
    ),
]
