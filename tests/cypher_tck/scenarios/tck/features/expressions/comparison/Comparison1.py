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
        key='expr-comparison1-1',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[1] Number-typed integer comparison',
        cypher='WITH collect([0, 0.0]) AS numbers\n      UNWIND numbers AS arr\n      WITH arr[0] AS expected\n      MATCH (n) WHERE toInteger(n.id) = expected\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[
            {'n': '({id: 0})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-2',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[2] Number-typed float comparison',
        cypher='WITH collect([0.5, 0]) AS numbers\n      UNWIND numbers AS arr\n      WITH arr[0] AS expected\n      MATCH (n) WHERE toInteger(n.id) = expected\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-3',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[3] Any-typed string comparison',
        cypher="WITH collect(['0', 0]) AS things\n      UNWIND things AS arr\n      WITH arr[0] AS expected\n      MATCH (n) WHERE toInteger(n.id) = expected\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-4',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[4] Comparing nodes to nodes',
        cypher='MATCH (a)\n      WITH a\n      MATCH (b)\n      WHERE a = b\n      RETURN count(b)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'count(b)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-5',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[5] Comparing relationships to relationships',
        cypher='MATCH ()-[a]->()\n      WITH a\n      MATCH ()-[b]->()\n      WHERE a = b\n      RETURN count(b)',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'count(b)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-1',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 1)',
        cypher='RETURN [1, 2] = [1] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-2',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 2)',
        cypher='RETURN [null] = [1] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-3',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 3)',
        cypher="RETURN ['a'] = [1] AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-4',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 4)',
        cypher='RETURN [[1]] = [[1], [null]] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-5',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 5)',
        cypher='RETURN [[1], [2]] = [[1], [null]] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-6-6',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[6] Comparing lists to lists (example 6)',
        cypher='RETURN [[1], [2, 3]] = [[1], [null]] AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-1',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 1)',
        cypher='RETURN {} = {} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-2',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 2)',
        cypher='RETURN {k: true} = {k: true} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-3',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 3)',
        cypher='RETURN {k: 1} = {k: 1} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-4',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 4)',
        cypher='RETURN {k: 1.0} = {k: 1.0} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-5',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 5)',
        cypher="RETURN {k: 'abc'} = {k: 'abc'} AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-6',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 6)',
        cypher="RETURN {k: 'a', l: 2} = {k: 'a', l: 2} AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-7',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 7)',
        cypher='RETURN {} = {k: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-8',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 8)',
        cypher='RETURN {k: null} = {} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-9',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 9)',
        cypher='RETURN {k: 1} = {k: 1, l: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-10',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 10)',
        cypher='RETURN {k: null, l: 1} = {l: 1} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-11',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 11)',
        cypher='RETURN {k: null} = {k: null, l: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-12',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 12)',
        cypher='RETURN {k: null} = {k: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-13',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 13)',
        cypher='RETURN {k: 1} = {k: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-14',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 14)',
        cypher='RETURN {k: 1, l: null} = {k: null, l: null} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-15',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 15)',
        cypher='RETURN {k: 1, l: null} = {k: null, l: 1} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-7-16',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[7] Comparing maps to maps (example 16)',
        cypher='RETURN {k: 1, l: null} = {k: 1, l: 1} AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-8-1',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[8] Equality and inequality of NaN (example 1)',
        cypher='RETURN 0.0 / 0.0 = 1 AS isEqual, 0.0 / 0.0 <> 1 AS isNotEqual',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'isEqual': 'false', 'isNotEqual': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-8-2',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[8] Equality and inequality of NaN (example 2)',
        cypher='RETURN 0.0 / 0.0 = 1.0 AS isEqual, 0.0 / 0.0 <> 1.0 AS isNotEqual',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'isEqual': 'false', 'isNotEqual': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-8-3',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[8] Equality and inequality of NaN (example 3)',
        cypher='RETURN 0.0 / 0.0 = 0.0 / 0.0 AS isEqual, 0.0 / 0.0 <> 0.0 / 0.0 AS isNotEqual',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'isEqual': 'false', 'isNotEqual': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-8-4',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[8] Equality and inequality of NaN (example 4)',
        cypher="RETURN 0.0 / 0.0 = 'a' AS isEqual, 0.0 / 0.0 <> 'a' AS isNotEqual",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'isEqual': 'false', 'isNotEqual': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-9-1',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[9] Equality between strings and numbers (example 1)',
        cypher='RETURN 1.0 = 1.0 AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-9-2',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[9] Equality between strings and numbers (example 2)',
        cypher='RETURN 1 = 1.0 AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-9-3',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[9] Equality between strings and numbers (example 3)',
        cypher="RETURN '1.0' = 1.0 AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-9-4',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[9] Equality between strings and numbers (example 4)',
        cypher="RETURN '1' = 1 AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-10',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[10] Handling inlined equality of large integer',
        cypher='MATCH (p:TheLabel {id: 4611686018427387905})\n      RETURN p.id',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[
            {'p.id': 4611686018427387905}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-11',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[11] Handling explicit equality of large integer',
        cypher='MATCH (p:TheLabel)\n      WHERE p.id = 4611686018427387905\n      RETURN p.id',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[
            {'p.id': 4611686018427387905}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-12',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[12] Handling inlined equality of large integer, non-equal values',
        cypher='MATCH (p:TheLabel {id : 4611686018427387900})\n      RETURN p.id',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-13',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[13] Handling explicit equality of large integer, non-equal values',
        cypher='MATCH (p:TheLabel)\n      WHERE p.id = 4611686018427387900\n      RETURN p.id',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-14',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[14] Direction of traversed relationship is not significant for path equality, simple',
        cypher='MATCH p1 = (:A)-->()\n      MATCH p2 = (:A)<--()\n      RETURN p1 = p2',
        graph=graph_fixture_from_create(
            """
            CREATE (n:A)-[:LOOP]->(n)
            """
        ),
        expected=Expected(
            rows=[
            {'p1 = p2': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-15',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[15] It is unknown - i.e. null - if a null is equal to a null',
        cypher='RETURN null = null AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-16',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[16] It is unknown - i.e. null - if a null is not equal to a null',
        cypher='RETURN null <> null AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-comparison1-17',
        feature_path='tck/features/expressions/comparison/Comparison1.feature',
        scenario='[17] Failing when comparing to an undefined variable',
        cypher='MATCH (s)\n      WHERE s.name = undefinedVariable\n        AND s.age = 10\n      RETURN s',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'comparison', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
