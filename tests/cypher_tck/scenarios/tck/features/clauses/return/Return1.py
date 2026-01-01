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
        key="return1-1",
        feature_path="tck/features/clauses/return/Return1.feature",
        scenario="[1] Returning a list property",
        cypher="MATCH (n)\nRETURN n",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "numbers": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            node_ids=["n1"],
            rows=[
                {"n": "({numbers: [1, 2, 3]})"},
            ],
        ),
        gfql=[n()],
        tags=("return", "list-property"),
    ),

    Scenario(
        key="return1-2",
        feature_path="tck/features/clauses/return/Return1.feature",
        scenario="[2] Fail when returning an undefined variable",
        cypher="MATCH ()\nRETURN foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for undefined variables is not enforced",
        tags=("return", "syntax-error", "xfail"),
    ),
]
