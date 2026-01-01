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
        key='expr-typeconversion2-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[1] `toInteger()` on float',
        cypher='WITH 82.9 AS weight\n      RETURN toInteger(weight)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'toInteger(weight)': 82}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[2] `toInteger()` returning null on non-numerical string',
        cypher="WITH 'foo' AS foo_string, '' AS empty_string\n      RETURN toInteger(foo_string) AS foo, toInteger(empty_string) AS empty",
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
        key='expr-typeconversion2-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[3] `toInteger()` handling mixed number types',
        cypher='WITH [2, 2.9] AS numbers\n      RETURN [n IN numbers | toInteger(n)] AS int_numbers',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'int_numbers': '[2, 2]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[4] `toInteger()` handling Any type',
        cypher="WITH [2, 2.9, '1.7'] AS things\n      RETURN [n IN things | toInteger(n)] AS int_numbers",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'int_numbers': '[2, 2, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[5] `toInteger()` on a list of strings',
        cypher="WITH ['2', '2.9', 'foo'] AS numbers\n      RETURN [n IN numbers | toInteger(n)] AS int_numbers",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'int_numbers': '[2, 2, null]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-6',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[6] `toInteger()` on a complex-typed expression',
        cypher='RETURN toInteger(1 - $param) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-7',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[7] `toInteger()` on node property',
        cypher="MATCH (p:Person { name: '42' })\n      WITH *\n      MATCH (n)\n      RETURN toInteger(n.name) AS name",
        graph=graph_fixture_from_create(
            """
            CREATE (:Person {name: '42'})
            """
        ),
        expected=Expected(
            rows=[
            {'name': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion2-8-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[8] Fail `toInteger()` on invalid types #Example: <exampleName> (example 1)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, []] | toInteger(x) ] AS list',
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
        key='expr-typeconversion2-8-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[8] Fail `toInteger()` on invalid types #Example: <exampleName> (example 2)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, {}] | toInteger(x) ] AS list',
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
        key='expr-typeconversion2-8-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[8] Fail `toInteger()` on invalid types #Example: <exampleName> (example 3)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, n] | toInteger(x) ] AS list',
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
        key='expr-typeconversion2-8-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[8] Fail `toInteger()` on invalid types #Example: <exampleName> (example 4)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, r] | toInteger(x) ] AS list',
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
        key='expr-typeconversion2-8-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion2.feature',
        scenario='[8] Fail `toInteger()` on invalid types #Example: <exampleName> (example 5)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, p] | toInteger(x) ] AS list',
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
