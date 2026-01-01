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
        key="return-orderby5-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy5.feature",
        scenario="[1] Renaming columns before ORDER BY should return results in ascending order",
        cypher="MATCH (n)\nRETURN n.num AS n\nORDER BY n + 2",
        graph=graph_fixture_from_create(
            """
            CREATE (n1 {num: 1}),
              (n2 {num: 3}),
              (n3 {num: -5})
            """
        ),
        expected=Expected(
            rows=[
                {"n": -5},
                {"n": 1},
                {"n": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY expression evaluation is not supported",
        tags=("return", "orderby", "expression", "xfail"),
    ),
]
