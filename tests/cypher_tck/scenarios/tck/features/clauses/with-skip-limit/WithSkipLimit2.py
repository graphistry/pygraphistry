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
        key="with-skip-limit2-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[1] ORDER BY and LIMIT can be used",
        cypher="MATCH (a:A)\nWITH a\nORDER BY a.name\nLIMIT 1\nMATCH (a)-->(b)\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (), (), (),
                   (a)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, and LIMIT are not supported",
        tags=("with", "limit", "orderby", "xfail"),
    ),

    Scenario(
        key="with-skip-limit2-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[2] Handle dependencies across WITH with LIMIT",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\n  LIMIT 1\nMATCH (b)\nWHERE b.id = property\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End {num: 42, id: 0}),
                   (:End {num: 3}),
                   (:Begin {num: a.id})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and joins are not supported",
        tags=("with", "limit", "join", "xfail"),
    ),

    Scenario(
        key="with-skip-limit2-3",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[3] Connected components succeeding WITH with LIMIT",
        cypher="MATCH (n:A)\nWITH n\nLIMIT 1\nMATCH (m:B), (n)-->(x:X)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:X)
            CREATE (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"m": "(:B)", "n": "(:A)", "x": "(:X)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and RETURN * projections are not supported",
        tags=("with", "limit", "return-star", "xfail"),
    ),

    Scenario(
        key="with-skip-limit2-4",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[4] Ordering and limiting on aggregate",
        cypher="MATCH ()-[r1]->(x)\nWITH x, sum(r1.num) AS c\n  ORDER BY c LIMIT 1\nRETURN x, c",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {num: 3}]->(x:X),
                   ()-[:T2 {num: 2}]->(x),
                   ()-[:T3 {num: 1}]->(:Y)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:Y)", "c": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, ORDER BY, and LIMIT are not supported",
        tags=("with", "limit", "orderby", "aggregation", "xfail"),
    ),
]
