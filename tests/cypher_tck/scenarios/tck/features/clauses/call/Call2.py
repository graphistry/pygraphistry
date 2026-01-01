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
        key="call2-1",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[1] In-query call to procedure with explicit arguments",
        cypher="CALL test.my.proc('Stefan', 1) YIELD city, country_code\nRETURN city, country_code",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call2-2",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[2] Standalone call to procedure with explicit arguments",
        cypher="CALL test.my.proc('Stefan', 1)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call2-3",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[3] Standalone call to procedure with implicit arguments",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and parameter binding are not supported",
        tags=("call", "params", "xfail"),
    ),

    Scenario(
        key="call2-4",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[4] In-query call to procedure that takes arguments fails when trying to pass them implicitly",
        cypher="CALL test.my.proc YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call2-5",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[5] Standalone call to procedure should fail if input type is wrong",
        cypher="CALL test.my.proc(true)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call2-6",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[6] In-query call to procedure should fail if input type is wrong",
        cypher="CALL test.my.proc(true) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
]
