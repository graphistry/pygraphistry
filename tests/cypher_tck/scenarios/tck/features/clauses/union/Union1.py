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
        key="union1-1",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[1] Two elements, both unique, distinct",
        cypher="RETURN 1 AS x\nUNION\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),

    Scenario(
        key="union1-2",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[2] Three elements, two unique, distinct",
        cypher="RETURN 2 AS x\nUNION\nRETURN 1 AS x\nUNION\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),

    Scenario(
        key="union1-3",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[3] Two single-column inputs, one with duplicates, distinct",
        cypher="UNWIND [2, 1, 2, 3] AS x\nRETURN x\nUNION\nUNWIND [3, 4] AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
                {"x": 3},
                {"x": 4},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION and UNWIND are not supported",
        tags=("union", "unwind", "xfail"),
    ),

    Scenario(
        key="union1-4",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[4] Should be able to create text output from union queries",
        cypher="MATCH (a:A)\nRETURN a AS a\nUNION\nMATCH (b:B)\nRETURN b AS a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)"},
                {"a": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),

    Scenario(
        key="union1-5",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[5] Failing when UNION has different columns",
        cypher="RETURN 1 AS a\nUNION\nRETURN 2 AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),
]
