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
        key='expr-list6-1',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[1] Return list size',
        cypher='RETURN size([1, 2, 3]) AS n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'n': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-2',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[2] Setting and returning the size of a list property',
        cypher='MATCH (n:TheLabel)\n      SET n.numbers = [1, 2, 3]\n      RETURN size(n.numbers)',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'size(n.numbers)': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-3',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[3] Concatenating and returning the size of literal lists',
        cypher='RETURN size([[], []] + [[]]) AS l',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'l': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-4',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[4] `size()` on null list',
        cypher='WITH null AS l\n      RETURN size(l), size(null)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'size(l)': 'null', 'size(null)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-5',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[5] Fail for `size()` on paths',
        cypher='MATCH p = (a)-[*]->(b)\n      RETURN size(p)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-1',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 1)',
        cypher='MATCH (a), (b), (c)\n      RETURN size(()--())',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-2',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 2)',
        cypher='MATCH (a), (b), (c)\n      RETURN size(()--(a))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-3',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 3)',
        cypher='MATCH (a), (b), (c)\n      RETURN size((a)-->())',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-4',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 4)',
        cypher='MATCH (a), (b), (c)\n      RETURN size((a)<--(a {}))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-5',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 5)',
        cypher='MATCH (a), (b), (c)\n      RETURN size((a)-[:REL]->(b))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-6',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 6)',
        cypher='MATCH (a), (b), (c)\n      RETURN size((a)-[:REL]->(b))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-7',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 7)',
        cypher='MATCH (a), (b), (c)\n      RETURN size((a)-[:REL]->(:C)<-[:REL]-(a {num: 5}))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-6-8',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[6] Fail for `size()` on pattern predicates (example 8)',
        cypher='MATCH (a), (b), (c)\n      RETURN size(()-[:REL*0..2]->()<-[:REL]-(:A {num: 5}))',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-list6-7',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[7] Using size of pattern comprehension to test existence',
        cypher='MATCH (n:X)\n      RETURN n, size([(n)--() | 1]) > 0 AS b',
        graph=graph_fixture_from_create(
            """
            CREATE (a:X {num: 42}), (:X {num: 43})
                  CREATE (a)-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:X {num: 42})', 'b': 'true'},
            {'n': '(:X {num: 43})', 'b': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-8',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[8] Get node degree via size of pattern comprehension',
        cypher='MATCH (a:X)\n      RETURN size([(a)-->() | 1]) AS length',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X),
                    (x)-[:T]->(),
                    (x)-[:T]->(),
                    (x)-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'length': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-9',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[9] Get node degree via size of pattern comprehension that specifies a relationship type',
        cypher='MATCH (a:X)\n      RETURN size([(a)-[:T]->() | 1]) AS length',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X),
                    (x)-[:T]->(),
                    (x)-[:T]->(),
                    (x)-[:T]->(),
                    (x)-[:OTHER]->()
            """
        ),
        expected=Expected(
            rows=[
            {'length': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list6-10',
        feature_path='tck/features/expressions/list/List6.feature',
        scenario='[10] Get node degree via size of pattern comprehension that specifies multiple relationship types',
        cypher='MATCH (a:X)\n      RETURN size([(a)-[:T|OTHER]->() | 1]) AS length',
        graph=graph_fixture_from_create(
            """
            CREATE (x:X),
                    (x)-[:T]->(),
                    (x)-[:T]->(),
                    (x)-[:T]->(),
                    (x)-[:OTHER]->()
            """
        ),
        expected=Expected(
            rows=[
            {'length': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),
]
