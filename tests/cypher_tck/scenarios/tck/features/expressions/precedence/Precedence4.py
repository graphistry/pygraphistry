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
        key='expr-precedence4-1-1',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[1] Null predicate takes precedence over comparison operator (example 1)',
        cypher='RETURN null IS NOT NULL = null IS NULL AS a,\n             (null IS NOT NULL) = (null IS NULL) AS b,\n             (null IS NOT NULL = null) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-1-2',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[1] Null predicate takes precedence over comparison operator (example 2)',
        cypher='RETURN null IS NULL <> null IS NULL AS a,\n             (null IS NULL) <> (null IS NULL) AS b,\n             (null IS NULL <> null) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-1-3',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[1] Null predicate takes precedence over comparison operator (example 3)',
        cypher='RETURN null IS NULL <> null IS NOT NULL AS a,\n             (null IS NULL) <> (null IS NOT NULL) AS b,\n             (null IS NULL <> null) IS NOT NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-2',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[2] Null predicate takes precedence over boolean negation',
        cypher='RETURN NOT null IS NULL AS a,\n             NOT (null IS NULL) AS b,\n             (NOT null) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-1',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 1)',
        cypher='RETURN null AND null IS NULL AS a,\n             null AND (null IS NULL) AS b,\n             (null AND null) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-2',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 2)',
        cypher='RETURN null AND true IS NULL AS a,\n             null AND (true IS NULL) AS b,\n             (null AND true) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-3',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 3)',
        cypher='RETURN false AND false IS NOT NULL AS a,\n             false AND (false IS NOT NULL) AS b,\n             (false AND false) IS NOT NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-4',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 4)',
        cypher='RETURN null OR false IS NULL AS a,\n             null OR (false IS NULL) AS b,\n             (null OR false) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null', 'b': 'null', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-5',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 5)',
        cypher='RETURN true OR null IS NULL AS a,\n             true OR (null IS NULL) AS b,\n             (true OR null) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-6',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 6)',
        cypher='RETURN true XOR null IS NOT NULL AS a,\n             true XOR (null IS NOT NULL) AS b,\n             (true XOR null) IS NOT NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-3-7',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[3] Null predicate takes precedence over binary boolean operator (example 7)',
        cypher='RETURN true XOR false IS NULL AS a,\n             true XOR (false IS NULL) AS b,\n             (true XOR false) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence4-4',
        feature_path='tck/features/expressions/precedence/Precedence4.feature',
        scenario='[4] String predicate takes precedence over binary boolean operator',
        cypher="RETURN ('abc' STARTS WITH null OR true) = (('abc' STARTS WITH null) OR true) AS a,\n             ('abc' STARTS WITH null OR true) <> ('abc' STARTS WITH (null OR true)) AS b,\n             (true OR null STARTS WITH 'abc') = (true OR (null STARTS WITH 'abc')) AS c,\n             (true OR null STARTS WITH 'abc') <> ((true OR null) STARTS WITH 'abc') AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'null', 'c': 'true', 'd': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),
]
