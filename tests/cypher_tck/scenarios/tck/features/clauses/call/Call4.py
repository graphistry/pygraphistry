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
        key="call4-1",
        feature_path="tck/features/clauses/call/Call4.feature",
        scenario="[1] Standalone call to procedure with null argument",
        cypher="CALL test.my.proc(null)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call4-2",
        feature_path="tck/features/clauses/call/Call4.feature",
        scenario="[2] In-query call to procedure with null argument",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
]
