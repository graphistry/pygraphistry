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
        key='expr-literals1-1',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[1] Return a boolean true lower case',
        cypher='RETURN true AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals1-2',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[2] Return a boolean true upper case',
        cypher='RETURN TRUE AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals1-3',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[3] Return a boolean false lower case',
        cypher='RETURN false AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals1-4',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[4] Return a boolean false upper case',
        cypher='RETURN FALSE AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals1-5',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[5] Return null lower case',
        cypher='RETURN null AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals1-6',
        feature_path='tck/features/expressions/literals/Literals1.feature',
        scenario='[6] Return null upper case',
        cypher='RETURN NULL AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),
]
