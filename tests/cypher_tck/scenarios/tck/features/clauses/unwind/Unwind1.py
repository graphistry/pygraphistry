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
        key="unwind1-1",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[1] Unwinding a list",
        cypher="UNWIND [1, 2, 3] AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),

    Scenario(
        key="unwind1-2",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[2] Unwinding a range",
        cypher="UNWIND range(1, 3) AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and range() expressions are not supported",
        tags=("unwind", "expression", "xfail"),
    ),

    Scenario(
        key="unwind1-3",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[3] Unwinding a concatenation of lists",
        cypher="WITH [1, 2, 3] AS first, [4, 5, 6] AS second\nUNWIND (first + second) AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
                {"x": 4},
                {"x": 5},
                {"x": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and list concatenation are not supported",
        tags=("unwind", "with", "expression", "xfail"),
    ),

    Scenario(
        key="unwind1-4",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[4] Unwinding a collected unwound expression",
        cypher="UNWIND RANGE(1, 2) AS row\nWITH collect(row) AS rows\nUNWIND rows AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, aggregation, and range() expressions are not supported",
        tags=("unwind", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="unwind1-5",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[5] Unwinding a collected expression",
        cypher="MATCH (row)\nWITH collect(row) AS rows\nUNWIND rows AS node\nRETURN node.id",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1}), ({id: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"node.id": 1},
                {"node.id": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, aggregation, and projection evaluation are not supported",
        tags=("unwind", "aggregation", "projection", "xfail"),
    ),

    Scenario(
        key="unwind1-6",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[6] Creating nodes from an unwound parameter list",
        cypher="UNWIND $events AS event\nMATCH (y:Year {year: event.year})\nMERGE (e:Event {id: event.id})\nMERGE (y)<-[:IN]-(e)\nRETURN e.id AS x\nORDER BY x",
        graph=graph_fixture_from_create(
            """
            CREATE (:Year {year: 2016})
            """
        ),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, parameters, MERGE, ORDER BY, and side-effect validation are not supported",
        tags=("unwind", "params", "merge", "orderby", "xfail"),
    ),

    Scenario(
        key="unwind1-7",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[7] Double unwinding a list of lists",
        cypher="WITH [[1, 2, 3], [4, 5, 6]] AS lol\nUNWIND lol AS x\nUNWIND x AS y\nRETURN y",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"y": 1},
                {"y": 2},
                {"y": 3},
                {"y": 4},
                {"y": 5},
                {"y": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and UNWIND are not supported",
        tags=("unwind", "with", "xfail"),
    ),

    Scenario(
        key="unwind1-8",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[8] Unwinding the empty list",
        cypher="UNWIND [] AS empty\nRETURN empty",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),

    Scenario(
        key="unwind1-9",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[9] Unwinding null",
        cypher="UNWIND null AS nil\nRETURN nil",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),

    Scenario(
        key="unwind1-10",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[10] Unwinding list with duplicates",
        cypher="UNWIND [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] AS duplicate\nRETURN duplicate",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"duplicate": 1},
                {"duplicate": 1},
                {"duplicate": 2},
                {"duplicate": 2},
                {"duplicate": 3},
                {"duplicate": 3},
                {"duplicate": 4},
                {"duplicate": 4},
                {"duplicate": 5},
                {"duplicate": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),

    Scenario(
        key="unwind1-11",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[11] Unwind does not prune context",
        cypher="WITH [1, 2, 3] AS list\nUNWIND list AS x\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"list": "[1, 2, 3]", "x": 1},
                {"list": "[1, 2, 3]", "x": 2},
                {"list": "[1, 2, 3]", "x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and projection expansion are not supported",
        tags=("unwind", "with", "projection", "xfail"),
    ),

    Scenario(
        key="unwind1-12",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[12] Unwind does not remove variables from scope",
        cypher="MATCH (a:S)-[:X]->(b1)\nWITH a, collect(b1) AS bees\nUNWIND bees AS b2\nMATCH (a)-[:Y]->(b2)\nRETURN a, b2",
        graph=graph_fixture_from_create(
            """
            CREATE (s:S),
              (n),
              (e:E),
              (s)-[:X]->(e),
              (s)-[:Y]->(e),
              (n)-[:Y]->(e)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:S)", "b2": "(:E)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, aggregation, and multi-step MATCH are not supported",
        tags=("unwind", "with", "aggregation", "match", "xfail"),
    ),

    Scenario(
        key="unwind1-13",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[13] Multiple unwinds after each other",
        cypher="WITH [1, 2] AS xs, [3, 4] AS ys, [5, 6] AS zs\nUNWIND xs AS x\nUNWIND ys AS y\nUNWIND zs AS z\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and UNWIND are not supported",
        tags=("unwind", "with", "xfail"),
    ),
]
