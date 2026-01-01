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
        key="union3-1",
        feature_path="tck/features/clauses/union/Union3.feature",
        scenario="[1] Failing when mixing UNION and UNION ALL",
        cypher="RETURN 1 AS a\nUNION\nRETURN 2 AS a\nUNION ALL\nRETURN 3 AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION clause composition validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),

    Scenario(
        key="union3-2",
        feature_path="tck/features/clauses/union/Union3.feature",
        scenario="[2] Failing when mixing UNION ALL and UNION",
        cypher="RETURN 1 AS a\nUNION ALL\nRETURN 2 AS a\nUNION\nRETURN 3 AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION clause composition validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),
]
