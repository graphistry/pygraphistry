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
        key="call3-1",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[1] Standalone call to procedure with argument of type NUMBER accepts value of type INTEGER",
        cypher="CALL test.my.proc(42)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'wisdom'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call3-2",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[2] In-query call to procedure with argument of type NUMBER accepts value of type INTEGER",
        cypher="CALL test.my.proc(42) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'wisdom'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call3-3",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[3] Standalone call to procedure with argument of type NUMBER accepts value of type FLOAT",
        cypher="CALL test.my.proc(42.3)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'about right'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call3-4",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[4] In-query call to procedure with argument of type NUMBER accepts value of type FLOAT",
        cypher="CALL test.my.proc(42.3) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'about right'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call3-5",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[5] Standalone call to procedure with argument of type FLOAT accepts value of type INTEGER",
        cypher="CALL test.my.proc(42)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'close enough'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call3-6",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[6] In-query call to procedure with argument of type FLOAT accepts value of type INTEGER",
        cypher="CALL test.my.proc(42) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'close enough'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
]
