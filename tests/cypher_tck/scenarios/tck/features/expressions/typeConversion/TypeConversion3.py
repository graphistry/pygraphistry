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
        key='expr-typeconversion3-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[1] `toFloat()` on mixed number types',
        cypher='WITH [3.4, 3] AS numbers\n      RETURN [n IN numbers | toFloat(n)] AS float_numbers',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'float_numbers': '[3.4, 3.0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion3-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[2] `toFloat()` returning null on non-numerical string',
        cypher="WITH 'foo' AS foo_string, '' AS empty_string\n      RETURN toFloat(foo_string) AS foo, toFloat(empty_string) AS empty",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'foo': 'null', 'empty': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion3-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[3] `toFloat()` handling Any type',
        cypher="WITH [3.4, 3, '5'] AS numbers\n      RETURN [n IN numbers | toFloat(n)] AS float_numbers",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'float_numbers': '[3.4, 3.0, 5.0]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion3-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[4] `toFloat()` on a list of strings',
        cypher="WITH ['1', '2', 'foo'] AS numbers\n      RETURN [n IN numbers | toFloat(n)] AS float_numbers",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'float_numbers': '[1.0, 2.0, null]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion3-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[5] `toFloat()` on node property',
        cypher='MATCH (m:Movie { rating: 4 })\n      WITH *\n      MATCH (n)\n      RETURN toFloat(n.rating) AS float',
        graph=graph_fixture_from_create(
            """
            CREATE (:Movie {rating: 4})
            """
        ),
        expected=Expected(
            rows=[
            {'float': 4.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion3-6-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 1)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, true] | toFloat(x) ] AS list',
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
        key='expr-typeconversion3-6-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 2)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, []] | toFloat(x) ] AS list',
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
        key='expr-typeconversion3-6-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 3)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, {}] | toFloat(x) ] AS list',
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
        key='expr-typeconversion3-6-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 4)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, n] | toFloat(x) ] AS list',
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
        key='expr-typeconversion3-6-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 5)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, r] | toFloat(x) ] AS list',
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
        key='expr-typeconversion3-6-6',
        feature_path='tck/features/expressions/typeConversion/TypeConversion3.feature',
        scenario='[6] Fail `toFloat()` on invalid types #Example: <exampleName> (example 6)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1.0, p] | toFloat(x) ] AS list',
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
