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
        key="call6-1",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[1] Calling the same STRING procedure twice using the same outputs in each call",
        cypher="CALL test.labels() YIELD label\nWITH count(*) AS c\nCALL test.labels() YIELD label\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 3, "label": "'A'"},
                {"c": 3, "label": "'B'"},
                {"c": 3, "label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call6-2",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[2] Project procedure results between query scopes with WITH clause",
        cypher="CALL test.my.proc(null) YIELD out\nWITH out RETURN out",
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
        key="call6-3",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[3] Project procedure results between query scopes with WITH clause and rename the projection",
        cypher="CALL test.my.proc(null) YIELD out\nWITH out AS a RETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
]
