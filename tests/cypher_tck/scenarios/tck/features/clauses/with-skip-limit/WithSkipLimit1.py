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
        key="with-skip-limit1-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit1.feature",
        scenario="[1] Handle dependencies across WITH with SKIP",
        cypher="MATCH (a)\nWITH a.name AS property, a.num AS idToUse\n  ORDER BY property\n  SKIP 1\nMATCH (b)\nWHERE b.id = idToUse\nRETURN DISTINCT b",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A', num: 0, id: 0}),
                   ({name: 'B', num: a.id, id: 1}),
                   ({name: 'C', num: 0, id: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "({name: 'A', num: 0, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, SKIP, and DISTINCT are not supported",
        tags=("with", "skip", "orderby", "distinct", "xfail"),
    ),

    Scenario(
        key="with-skip-limit1-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit1.feature",
        scenario="[2] Ordering and skipping on aggregate",
        cypher="MATCH ()-[r1]->(x)\nWITH x, sum(r1.num) AS c\n  ORDER BY c SKIP 1\nRETURN x, c",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {num: 3}]->(x:X),
                   ()-[:T2 {num: 2}]->(x),
                   ()-[:T3 {num: 1}]->(:Y)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:X)", "c": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, ORDER BY, and SKIP are not supported",
        tags=("with", "skip", "orderby", "aggregation", "xfail"),
    ),
]
