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
        key='expr-string10-1',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[1] Finding exact matches with non-proper substring',
        cypher="MATCH (a)\n      WHERE a.name CONTAINS 'ABCDEF'\n      RETURN a",
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
        key='expr-string10-2',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[2] Finding substring of string',
        cypher="MATCH (a)\n      WHERE a.name CONTAINS 'CD'\n      RETURN a",
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
        key='expr-string10-3',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[3] Finding the empty substring',
        cypher="MATCH (a)\n      WHERE a.name CONTAINS ''\n      RETURN a",
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
        key='expr-string10-4',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[4] Finding strings containing whitespace',
        cypher="MATCH (a)\n      WHERE a.name CONTAINS ' '\n      RETURN a.name AS name",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel),
                         (:TheLabel {name: 'Foo Foo'}),
                         (:TheLabel {name: 'Foo\nFoo'}),
                         (:TheLabel {name: 'Foo\tFoo'})
            """
        ),
        expected=Expected(
            rows=[
            {'name': "'Foo Foo'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string10-5',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[5] Finding strings containing newline',
        cypher="MATCH (a)\n      WHERE a.name CONTAINS '\\n'\n      RETURN a.name AS name",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {name: 'ABCDEF'}), (:TheLabel {name: 'AB'}),
                         (:TheLabel {name: 'abcdef'}), (:TheLabel {name: 'ab'}),
                         (:TheLabel {name: ''}), (:TheLabel),
                         (:TheLabel {name: 'Foo Foo'}),
                         (:TheLabel {name: 'Foo\nFoo'}),
                         (:TheLabel {name: 'Foo\tFoo'})
            """
        ),
        expected=Expected(
            rows=[
            {'name': "'Foo\\nFoo'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'string', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-string10-6',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[6] No string contains null',
        cypher='MATCH (a)\n      WHERE a.name CONTAINS null\n      RETURN a',
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
        key='expr-string10-7',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[7] No string does not contain null',
        cypher='MATCH (a)\n      WHERE NOT a.name CONTAINS null\n      RETURN a',
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
        key='expr-string10-8',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[8] Handling non-string operands for CONTAINS',
        cypher='WITH [1, 3.14, true, [], {}, null] AS operands\n      UNWIND operands AS op1\n      UNWIND operands AS op2\n      WITH op1 CONTAINS op2 AS v\n      RETURN v, count(*)',
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
        key='expr-string10-9',
        feature_path='tck/features/expressions/string/String10.feature',
        scenario='[9] NOT with CONTAINS',
        cypher="MATCH (a)\n      WHERE NOT a.name CONTAINS 'b'\n      RETURN a",
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
