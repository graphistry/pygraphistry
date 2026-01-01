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
        key='merge1-1',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[1] Merge node when no nodes exist',
        cypher='MERGE (a)\n      RETURN count(*) AS n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'n': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-2',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[2] Merge node with label',
        cypher='MERGE (a:TheLabel)\n      RETURN labels(a)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'labels(a)': "['TheLabel']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-3',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[3] Merge node with label when it exists',
        cypher='MERGE (a:TheLabel)\n      RETURN a.id',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 1})
            """
        ),
        expected=Expected(
            rows=[
            {'a.id': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-4',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario="[4] Merge node should create when it doesn't match, properties",
        cypher='MERGE (a {num: 43})\n      RETURN a.num',
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'a.num': 43}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-5',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario="[5] Merge node should create when it doesn't match, properties and label",
        cypher='MERGE (a:TheLabel {num: 43})\n      RETURN a.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'a.num': 43}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-6',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[6] Merge node with prop and label',
        cypher='MERGE (a:TheLabel {num: 42})\n      RETURN a.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'a.num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-7',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[7] Merge should work when finding multiple elements',
        cypher='CREATE (:X)\n      CREATE (:X)\n      MERGE (:X)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-8',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[8] Merge should handle argument properly',
        cypher='WITH 42 AS var\n      MERGE (c:N {var: var})',
        graph=graph_fixture_from_create(
            """
            CREATE ({var: 42}),
                    ({var: 'not42'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-9',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[9] Merge should support updates while merging',
        cypher='MATCH (foo)\n      WITH foo.x AS x, foo.y AS y\n      MERGE (:N {x: x, y: y + 1})\n      MERGE (:N {x: x, y: y})\n      MERGE (:N {x: x + 1, y: y})\n      RETURN x, y',
        graph=graph_fixture_from_create(
            """
            UNWIND [0, 1, 2] AS x
                  UNWIND [0, 1, 2] AS y
                  CREATE ({x: x, y: y})
            """
        ),
        expected=Expected(
            rows=[
            {'x': 0, 'y': 0},
            {'x': 0, 'y': 1},
            {'x': 0, 'y': 2},
            {'x': 1, 'y': 0},
            {'x': 1, 'y': 1},
            {'x': 1, 'y': 2},
            {'x': 2, 'y': 0},
            {'x': 2, 'y': 1},
            {'x': 2, 'y': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-10',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[10] Merge must properly handle multiple labels',
        cypher='MERGE (test:L:B {num: 42})\n      RETURN labels(test) AS labels',
        graph=graph_fixture_from_create(
            """
            CREATE (:L:A {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'labels': "['L', 'B']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-11',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[11] Merge should be able to merge using property of bound node',
        cypher='MATCH (person:Person)\n      MERGE (city:City {name: person.bornIn})',
        graph=graph_fixture_from_create(
            """
            CREATE (:Person {name: 'A', bornIn: 'New York'})
                  CREATE (:Person {name: 'B', bornIn: 'Ohio'})
                  CREATE (:Person {name: 'C', bornIn: 'New Jersey'})
                  CREATE (:Person {name: 'D', bornIn: 'New York'})
                  CREATE (:Person {name: 'E', bornIn: 'Ohio'})
                  CREATE (:Person {name: 'F', bornIn: 'New Jersey'})
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-12',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[12] Merge should be able to merge using property of freshly created node',
        cypher='CREATE (a {num: 1})\n      MERGE ({v: a.num})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-13',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[13] Merge should bind a path',
        cypher='MERGE p = (a {num: 1})\n      RETURN p',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'p': '<({num: 1})>'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-14',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[14] Merges should not be able to match on deleted nodes',
        cypher='MATCH (a:A)\n      DELETE a\n      MERGE (a2:A)\n      RETURN a2.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 1}),
                    (:A {num: 2})
            """
        ),
        expected=Expected(
            rows=[
            {'a2.num': 'null'},
            {'a2.num': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge1-15',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[15] Fail when merge a node that is already bound',
        cypher='MATCH (a)\n      MERGE (a)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge1-16',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[16] Fail when using parameter as node predicate in MERGE',
        cypher='MERGE (n $param)\n      RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge1-17',
        feature_path='tck/features/clauses/merge/Merge1.feature',
        scenario='[17] Fail on merging node with null property',
        cypher='MERGE ({num: null})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('merge', 'runtime-error', 'xfail'),
    ),
]
