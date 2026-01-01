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
        key='expr-literals3-1',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[1] Return a short positive hexadecimal integer',
        cypher='RETURN 0x1 AS literal',
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
        key='expr-literals3-2',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[2] Return a long positive hexadecimal integer',
        cypher='RETURN 0x162CD4F6 AS literal',
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
        key='expr-literals3-3',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[3] Return the largest hexadecimal integer',
        cypher='RETURN 0x7FFFFFFFFFFFFFFF AS literal',
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
        key='expr-literals3-4',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[4] Return a positive hexadecimal zero',
        cypher='RETURN 0x0 AS literal',
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
        key='expr-literals3-5',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[5] Return a negative hexadecimal zero',
        cypher='RETURN -0x0 AS literal',
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
        key='expr-literals3-6',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[6] Return a short negative hexadecimal integer',
        cypher='RETURN -0x1 AS literal',
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
        key='expr-literals3-7',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[7] Return a long negative hexadecimal integer',
        cypher='RETURN -0x162CD4F6 AS literal',
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
        key='expr-literals3-8',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[8] Return the smallest hexadecimal integer',
        cypher='RETURN -0x8000000000000000 AS literal',
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
        key='expr-literals3-9',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[9] Return a lower case hexadecimal integer',
        cypher='RETURN 0x1a2b3c4d5e6f7 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 460367961908983}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-10',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[10] Return a upper case hexadecimal integer',
        cypher='RETURN 0x1A2B3C4D5E6F7 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 460367961908983}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-11',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[11] Return a mixed case hexadecimal integer',
        cypher='RETURN 0x1A2b3c4D5E6f7 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 460367961908983}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-12',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[12] Fail on an incomplete hexadecimal integer',
        cypher='RETURN 0x AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-13',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[13] Fail on an hexadecimal literal containing a lower case invalid alphanumeric character',
        cypher='RETURN 0x1A2b3j4D5E6f7 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-14',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[14] Fail on an hexadecimal literal containing a upper case invalid alphanumeric character',
        cypher='RETURN 0x1A2b3c4Z5E6f7 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-16',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[16] Fail on a too large hexadecimal integer',
        cypher='RETURN 0x8000000000000000 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-literals3-17',
        feature_path='tck/features/expressions/literals/Literals3.feature',
        scenario='[17] Fail on a too small hexadecimal integer',
        cypher='RETURN -0x8000000000000001 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
