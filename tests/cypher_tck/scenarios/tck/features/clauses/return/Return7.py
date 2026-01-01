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
        key="return7-1",
        feature_path="tck/features/clauses/return/Return7.feature",
        scenario="[1] Return all variables",
        cypher="MATCH p = (a:Start)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:Start)-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:Start)", "b": "()", "p": "<(:Start)-[:T]->()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN * projections and named path returns are not supported",
        tags=("return", "projection", "path", "xfail"),
    ),

    Scenario(
        key="return7-2",
        feature_path="tck/features/clauses/return/Return7.feature",
        scenario="[2] Fail when using RETURN * without variables in scope",
        cypher="MATCH ()\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for RETURN * without scoped variables is not enforced",
        tags=("return", "syntax-error", "xfail"),
    ),
]
