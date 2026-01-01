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
        key='expr-literals5-1',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[1] Return a short positive float',
        cypher='RETURN 1.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-2',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[2] Return a short positive float without integer digits',
        cypher='RETURN .1 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-3',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[3] Return a long positive float',
        cypher='RETURN 3985764.3405892687 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 3985764.3405892686}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-4',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[4] Return a long positive float without integer digits',
        cypher='RETURN .3405892687 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.3405892687}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-5',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[5] Return a very long positive float',
        cypher='RETURN 126354186523812635418263552340512384016094862983471987543918591348961093487896783409268730945879405123840160948812635418265234051238401609486298347198754391859134896109348789678340926873094587962983471812635265234051238401609486298348126354182652340512384016094862983471987543918591348961093487896783409218.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '1.2635418652381264e305'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-6',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[6] Return a very long positive float without integer digits',
        cypher='RETURN .00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '1e-305'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-7',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[7] Return a positive zero float',
        cypher='RETURN 0.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-8',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[8] Return a positive zero float without integer digits',
        cypher='RETURN .0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-9',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[9] Return a negative zero float',
        cypher='RETURN -0.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-10',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[10] Return a negative zero float without integer digits',
        cypher='RETURN -.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 0.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-11',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[11] Return a very long negative float',
        cypher='RETURN -126354186523812635418263552340512384016094862983471987543918591348961093487896783409268730945879405123840160948812635418265234051238401609486298347198754391859134896109348789678340926873094587962983471812635265234051238401609486298348126354182652340512384016094862983471987543918591348961093487896783409218.0 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '-1.2635418652381264e305'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-12',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[12] Return a very long negative float without integer digits',
        cypher='RETURN -.00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '-1e-305'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-13',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[13] Return a positive float with positive lower case exponent',
        cypher='RETURN 1e9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1000000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-14',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[14] Return a positive float with positive upper case exponent',
        cypher='RETURN 1E9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1000000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-15',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[15] Return a positive float with positive lower case exponent without integer digits',
        cypher='RETURN .1e9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 100000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-16',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[16] Return a positive float with negative lower case exponent',
        cypher='RETURN 1e-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1e-05}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-17',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[17] Return a positive float with negative lower case exponent without integer digits',
        cypher='RETURN .1e-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1e-06}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-18',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[18] Return a positive float with negative upper case exponent without integer digits',
        cypher='RETURN .1E-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': 1e-06}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-19',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[19] Return a negative float in with positive lower case exponent',
        cypher='RETURN -1e9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1000000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-20',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[20] Return a negative float in with positive upper case exponent',
        cypher='RETURN -1E9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1000000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-21',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[21] Return a negative float with positive lower case exponent without integer digits',
        cypher='RETURN -.1e9 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -100000000.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-22',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[22] Return a negative float with negative lower case exponent',
        cypher='RETURN -1e-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1e-05}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-23',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[23] Return a negative float with negative lower case exponent without integer digits',
        cypher='RETURN -.1e-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1e-06}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-24',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[24] Return a negative float with negative upper case exponent without integer digits',
        cypher='RETURN -.1E-5 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': -1e-06}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-25',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[25] Return a positive float with one integer digit and maximum positive exponent',
        cypher='RETURN 1e308 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '1e308'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-26',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[26] Return a positive float with nine integer digit and maximum positive exponent',
        cypher='RETURN 123456789e300 AS literal',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'literal': '1.23456789e308'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-literals5-27',
        feature_path='tck/features/expressions/literals/Literals5.feature',
        scenario='[27] Fail when float value is too large',
        cypher='RETURN 1.34E999',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'literals', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
