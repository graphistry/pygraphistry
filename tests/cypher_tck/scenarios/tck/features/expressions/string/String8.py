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
        key='expr-string8-1',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[1] Finding exact matches with non-proper prefix',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH 'ABCDEF'\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'ABCDEF'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-2',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[2] Finding beginning of string',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH 'ABC'\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'ABCDEF'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-3',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[3] Finding the empty prefix',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH ''\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'ABCDEF'})"},
            {'a': "(:TheLabel {name: 'AB'})"},
            {'a': "(:TheLabel {name: 'abcdef'})"},
            {'a': "(:TheLabel {name: 'ab'})"},
            {'a': "(:TheLabel {name: ''})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-4',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[4] Finding strings starting with whitespace',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH ' '\n      RETURN a.name AS name",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel),
                         (:TheLabel {name: ' Foo '}),
                         (:TheLabel {name: '\nFoo\n'}),
                         (:TheLabel {name: '\tFoo\t'})
            """
        ),
        expected=Expected(
            rows=[
            {'name': "' Foo '"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-5',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[5] Finding strings starting with newline',
        cypher="MATCH (a)\n      WHERE a.name STARTS WITH '\\n'\n      RETURN a.name AS name",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel),
                         (:TheLabel {name: ' Foo '}),
                         (:TheLabel {name: '\nFoo\n'}),
                         (:TheLabel {name: '\tFoo\t'})
            """
        ),
        expected=Expected(
            rows=[
            {'name': "'\\nFoo\\n'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-6',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[6] No string starts with null',
        cypher='MATCH (a)\n      WHERE a.name STARTS WITH null\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-7',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[7] No string does not start with null',
        cypher='MATCH (a)\n      WHERE NOT a.name STARTS WITH null\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-8',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[8] Handling non-string operands for STARTS WITH',
        cypher='WITH [1, 3.14, true, [], {}, null] AS operands\n      UNWIND operands AS op1\n      UNWIND operands AS op2\n      WITH op1 STARTS WITH op2 AS v\n      RETURN v, count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'v': 'null', 'count(*)': 36}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string8-9',
        feature_path='tck/features/expressions/string/String8.feature',
        scenario='[9] NOT with STARTS WITH',
        cypher="MATCH (a)\n      WHERE NOT a.name STARTS WITH 'ab'\n      RETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a': "(:TheLabel {name: 'ABCDEF'})"},
            {'a': "(:TheLabel {name: 'AB'})"},
            {'a': "(:TheLabel {name: ''})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),
]
