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
        key='expr-typeconversion1-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[1] `toBoolean()` on booleans',
        cypher='UNWIND [true, false] AS b\n      RETURN toBoolean(b) AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'b': 'true'},
            {'b': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[2] `toBoolean()` on valid literal string',
        cypher="RETURN toBoolean('true') AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[3] `toBoolean()` on variables with valid string values',
        cypher="UNWIND ['true', 'false'] AS s\n      RETURN toBoolean(s) AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'b': 'true'},
            {'b': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[4] `toBoolean()` on invalid strings',
        cypher="UNWIND [null, '', ' tru ', 'f alse'] AS things\n      RETURN toBoolean(things) AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'b': 'null'},
            {'b': 'null'},
            {'b': 'null'},
            {'b': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 1)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, []] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 2)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, {}] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 3)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, 1.0] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 4)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, n] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 5)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, r] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion1-5-6',
        feature_path='tck/features/expressions/typeConversion/TypeConversion1.feature',
        scenario='[5] Fail `toBoolean()` on invalid types #Example: <exampleName> (example 6)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [true, p] | toBoolean(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'runtime-error', 'xfail'),
    ),
]
