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
        key="return-orderby4-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy4.feature",
        scenario="[1] ORDER BY of a column introduced in RETURN should return salient results in ascending order",
        cypher="WITH [0, 1] AS prows, [[2], [3, 4]] AS qrows\nUNWIND prows AS p\nUNWIND qrows[p] AS q\nWITH p, count(q) AS rng\nRETURN p\nORDER BY rng",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"p": 0},
                {"p": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and ORDER BY are not supported",
        tags=("return", "orderby", "with", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby4-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy4.feature",
        scenario="[2] Handle projections with ORDER BY",
        cypher="MATCH (c:Crew {name: 'Neo'})\nWITH c, 0 AS relevance\nRETURN c.rank AS rank\nORDER BY relevance, c.rank",
        graph=graph_fixture_from_create(
            """
            CREATE (c1:Crew {name: 'Neo', rank: 1}),
              (c2:Crew {name: 'Neo', rank: 2}),
              (c3:Crew {name: 'Neo', rank: 3}),
              (c4:Crew {name: 'Neo', rank: 4}),
              (c5:Crew {name: 'Neo', rank: 5})
            """
        ),
        expected=Expected(
            rows=[
                {"rank": 1},
                {"rank": 2},
                {"rank": 3},
                {"rank": 4},
                {"rank": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and ORDER BY are not supported",
        tags=("return", "orderby", "with", "xfail"),
    ),
]
