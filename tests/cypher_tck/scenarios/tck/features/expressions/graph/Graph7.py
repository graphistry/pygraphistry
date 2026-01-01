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
        key='expr-graph7-1',
        feature_path='tck/features/expressions/graph/Graph7.feature',
        scenario="[1] Execute n['name'] in read queries",
        cypher="MATCH (n {name: 'Apa'})\n      RETURN n['nam' + 'e'] AS value",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'Apa'})
            """
        ),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph7-2',
        feature_path='tck/features/expressions/graph/Graph7.feature',
        scenario="[2] Execute n['name'] in update queries",
        cypher="CREATE (n {name: 'Apa'})\n      RETURN n['nam' + 'e'] AS value",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph7-3',
        feature_path='tck/features/expressions/graph/Graph7.feature',
        scenario='[3] Use dynamic property lookup based on parameters when there is lhs type information',
        cypher="CREATE (n {name: 'Apa'})\n      RETURN n[$idx] AS value",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'params', 'xfail'),
    ),
]
