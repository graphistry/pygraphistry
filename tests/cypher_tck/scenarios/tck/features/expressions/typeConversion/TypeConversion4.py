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
        key='expr-typeconversion4-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[1] `toString()` handling integer literal',
        cypher='RETURN toString(42) AS bool',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'bool': "'42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[2] `toString()` handling boolean literal',
        cypher='RETURN toString(true) AS bool',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'bool': "'true'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[3] `toString()` handling inlined boolean',
        cypher='RETURN toString(1 < 0) AS bool',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'bool': "'false'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[4] `toString()` handling boolean properties',
        cypher='MATCH (m:Movie)\n      RETURN toString(m.watched)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Movie {watched: true})
            """
        ),
        expected=Expected(
            rows=[
            {'toString(m.watched)': "'true'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[5] `toString()` should work on Any type',
        cypher="RETURN [x IN [1, 2.3, true, 'apa'] | toString(x) ] AS list",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'list': "['1', '2.3', 'true', 'apa']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-6',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[6] `toString()` on a list of integers',
        cypher='WITH [1, 2, 3] AS numbers\n      RETURN [n IN numbers | toString(n)] AS string_numbers',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'string_numbers': "['1', '2', '3']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-7',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[7] `toString()` on node property',
        cypher='MATCH (m:Movie { rating: 4 })\n      WITH *\n      MATCH (n)\n      RETURN toString(n.rating)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Movie {rating: 4})
            """
        ),
        expected=Expected(
            rows=[
            {'toString(n.rating)': "'4'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-8',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[8] `toString()` should accept potentially correct types 1',
        cypher="UNWIND ['male', 'female', null] AS gen\n      RETURN coalesce(toString(gen), 'x') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'male'"},
            {'result': "'female'"},
            {'result': "'x'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-9',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[9] `toString()` should accept potentially correct types 2',
        cypher="UNWIND ['male', 'female', null] AS gen\n      RETURN toString(coalesce(gen, 'x')) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'male'"},
            {'result': "'female'"},
            {'result': "'x'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'typeConversion', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-typeconversion4-10-1',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[10] Fail `toString()` on invalid types #Example: <exampleName> (example 1)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, '', []] | toString(x) ] AS list",
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
        key='expr-typeconversion4-10-2',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[10] Fail `toString()` on invalid types #Example: <exampleName> (example 2)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, '', {}] | toString(x) ] AS list",
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
        key='expr-typeconversion4-10-3',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[10] Fail `toString()` on invalid types #Example: <exampleName> (example 3)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, '', n] | toString(x) ] AS list",
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
        key='expr-typeconversion4-10-4',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[10] Fail `toString()` on invalid types #Example: <exampleName> (example 4)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, '', r] | toString(x) ] AS list",
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
        key='expr-typeconversion4-10-5',
        feature_path='tck/features/expressions/typeConversion/TypeConversion4.feature',
        scenario='[10] Fail `toString()` on invalid types #Example: <exampleName> (example 5)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [1, '', p] | toString(x) ] AS list",
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
