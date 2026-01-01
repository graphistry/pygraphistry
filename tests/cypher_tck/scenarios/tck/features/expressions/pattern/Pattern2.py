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
        key='expr-pattern2-1',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[1] Return a pattern comprehension',
        cypher='MATCH (n)\n      RETURN [p = (n)-->() | p] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:T]->(b),
                         (b)-[:T]->(:C)
            """
        ),
        expected=Expected(
            rows=[
            {'list': '[<(:A)-[:T]->(:B)>]'},
            {'list': '[<(:B)-[:T]->(:C)>]'},
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-2',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[2] Return a pattern comprehension with label predicate',
        cypher='MATCH (n:A)\n      RETURN [p = (n)-->(:B) | p] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C), (d:D)
                  CREATE (a)-[:T]->(b),
                         (a)-[:T]->(c),
                         (a)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
            {'list': '[<(:A)-[:T]->(:B)>]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-3',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[3] Return a pattern comprehension with bound nodes',
        cypher='MATCH (a:A), (b:B)\n      RETURN [p = (a)-->(b) | p] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'list': '[<(:A)-[:T]->(:B)>]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-4',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[4] Introduce a new node variable in pattern comprehension',
        cypher='MATCH (n)\n      RETURN [(n)-[:T]->(b) | b.name] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b {name: 'val'}), (c)
                  CREATE (a)-[:T]->(b),
                         (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            rows=[
            {'list': "['val']"},
            {'list': '[null]'},
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-5',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[5] Introduce a new relationship variable in pattern comprehension',
        cypher='MATCH (n)\n      RETURN [(n)-[r:T]->() | r.name] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (c)
                  CREATE (a)-[:T {name: 'val'}]->(b),
                         (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            rows=[
            {'list': "['val']"},
            {'list': '[null]'},
            {'list': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-6',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[6] Aggregate on a pattern comprehension',
        cypher='MATCH (n:A)\n      RETURN count([p = (n)-[:HAS]->() | p]) AS c',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (:A), (:A)
                  CREATE (a)-[:HAS]->()
            """
        ),
        expected=Expected(
            rows=[
            {'c': 3}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-7',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[7] Use a pattern comprehension inside a list comprehension',
        cypher='MATCH p = (n:X)-->()\n      RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE (n1:X {n: 1}), (m1:Y), (i1:Y), (i2:Y)
                  CREATE (n1)-[:T]->(m1),
                         (m1)-[:T]->(i1),
                         (m1)-[:T]->(i2)
                  CREATE (n2:X {n: 2}), (m2), (i3:L), (i4:Y)
                  CREATE (n2)-[:T]->(m2),
                         (m2)-[:T]->(i3),
                         (m2)-[:T]->(i4)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:X {n: 1})', 'list': '[1, 2]'},
            {'n': '(:X {n: 2})', 'list': '[0, 1]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-8',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[8] Use a pattern comprehension in WITH',
        cypher='MATCH (n)-->(b)\n      WITH [p = (n)-->() | p] AS ps, count(b) AS c\n      RETURN ps, c',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:T]->(b),
                         (b)-[:T]->(:C)
            """
        ),
        expected=Expected(
            rows=[
            {'ps': '[<(:A)-[:T]->(:B)>]', 'c': 1},
            {'ps': '[<(:B)-[:T]->(:C)>]', 'c': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-9',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[9] Use a variable-length pattern comprehension in WITH',
        cypher='MATCH (a:A), (b:B)\n      WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c\n      RETURN paths, c',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T]->(:B)
            """
        ),
        expected=Expected(
            rows=[
            {'paths': '[<(:A)-[:T]->(:B)>]', 'c': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-10',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[10] Use a pattern comprehension in RETURN',
        cypher='MATCH (n:A)\n      RETURN [p = (n)-[:HAS]->() | p] AS ps',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (:A), (:A)
                  CREATE (a)-[:HAS]->()
            """
        ),
        expected=Expected(
            rows=[
            {'ps': '[<(:A)-[:HAS]->()>]'},
            {'ps': '[]'},
            {'ps': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern2-11',
        feature_path='tck/features/expressions/pattern/Pattern2.feature',
        scenario='[11] Use a pattern comprehension and ORDER BY',
        cypher='MATCH (liker)\n      RETURN [p = (liker)--() | p] AS isNew\n        ORDER BY liker.time',
        graph=graph_fixture_from_create(
            """
            CREATE (a {time: 10}), (b {time: 20})
                  CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'isNew': '[<({time: 10})-[:T]->({time: 20})>]'},
            {'isNew': '[<({time: 20})<-[:T]-({time: 10})>]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),
]
