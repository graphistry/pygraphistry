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
        key="with-where6-1",
        feature_path="tck/features/clauses/with-where/WithWhere6.feature",
        scenario="[1] Filter a single aggregate",
        cypher="MATCH (a)-->()\nWITH a, count(*) AS relCount\nWHERE relCount > 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}),
                   (b {name: 'B'})
            CREATE (a)-[:REL]->(),
                   (a)-[:REL]->(),
                   (a)-[:REL]->(),
                   (b)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'A'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and WHERE filtering are not supported",
        tags=("with", "aggregation", "where", "xfail"),
    ),
]
