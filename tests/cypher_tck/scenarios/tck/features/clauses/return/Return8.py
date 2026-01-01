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
        key="return8-1",
        feature_path="tck/features/clauses/return/Return8.feature",
        scenario="[1] Return aggregation after With filtering",
        cypher="MATCH (n)\nWITH n\nWHERE n.num = 42\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 43}), ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, WHERE filtering, and aggregations are not supported",
        tags=("return", "with", "aggregation", "xfail"),
    ),
]
