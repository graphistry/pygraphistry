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
        key='expr-literals2-1',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[1] Return a short positive integer',
        cypher='RETURN 1 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-2',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[2] Return a long positive integer',
        cypher='RETURN 372036854 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 372036854}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-3',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[3] Return the largest integer',
        cypher='RETURN 9223372036854775807 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 9223372036854775807}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-4',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[4] Return a positive zero',
        cypher='RETURN 0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-5',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[5] Return a negative zero',
        cypher='RETURN -0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-6',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[6] Return a short negative integer',
        cypher='RETURN -1 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-7',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[7] Return a long negative integer',
        cypher='RETURN -372036854 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -372036854}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-8',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[8] Return the smallest integer',
        cypher='RETURN -9223372036854775808 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -9223372036854775808}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-9',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[9] Fail on a too large integer',
        cypher='RETURN 9223372036854775808 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-10',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[10] Fail on a too small integer',
        cypher='RETURN -9223372036854775809 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-11',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[11] Fail on an integer containing a alphabetic character',
        cypher='RETURN 9223372h54775808 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals2-12',
        feature_path='tck/features/expressions/literals/Literals2.feature',
        scenario='[12] Fail on an integer containing a invalid symbol character',
        cypher='RETURN 9223372#54775808 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
