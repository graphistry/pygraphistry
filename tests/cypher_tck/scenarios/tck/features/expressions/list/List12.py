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
        key='expr-list12-1',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[1] Collect and extract using a list comprehension',
        cypher="MATCH (a:Label1)\n      WITH collect(a) AS nodes\n      WITH nodes, [x IN nodes | x.name] AS oldNames\n      UNWIND nodes AS n\n      SET n.name = 'newName'\n      RETURN n.name, oldNames",
        graph=graph_fixture_from_create(
            """
            CREATE (:Label1 {name: 'original'})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'newName'", 'oldNames': "['original']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-2',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[2] Collect and filter using a list comprehension',
        cypher="MATCH (a:Label1)\n      WITH collect(a) AS nodes\n      WITH nodes, [x IN nodes WHERE x.name = 'original'] AS noopFiltered\n      UNWIND nodes AS n\n      SET n.name = 'newName'\n      RETURN n.name, size(noopFiltered)",
        graph=graph_fixture_from_create(
            """
            CREATE (:Label1 {name: 'original'})
            """
        ),
        expected=Expected(
            rows=[
            {'n.name': "'newName'", 'size(noopFiltered)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-3',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[3] Size of list comprehension',
        cypher='MATCH (n)\n      OPTIONAL MATCH (n)-[r]->(m)\n      RETURN size([x IN collect(r) WHERE x <> null]) AS cn',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'cn': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-4',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[4] Returning a list comprehension',
        cypher='MATCH p = (n)-->()\n      RETURN [x IN collect(p) | head(nodes(x))] AS p',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)
                  CREATE (a)-[:T]->(:B),
                         (a)-[:T]->(:C)
            """
        ),
        expected=Expected(
            rows=[
            {'p': '[(:A), (:A)]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-5',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[5] Using a list comprehension in a WITH',
        cypher='MATCH p = (n:A)-->()\n      WITH [x IN collect(p) | head(nodes(x))] AS p, count(n) AS c\n      RETURN p, c',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)
                  CREATE (a)-[:T]->(:B),
                         (a)-[:T]->(:C)
            """
        ),
        expected=Expected(
            rows=[
            {'p': '[(:A), (:A)]', 'c': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-6',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[6] Using a list comprehension in a WHERE',
        cypher='MATCH (n)-->(b)\n      WHERE n.name IN [x IN labels(b) | toLower(x)]\n      RETURN b',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'c'})
                  CREATE (a)-[:T]->(:B),
                         (a)-[:T]->(:C)
            """
        ),
        expected=Expected(
            rows=[
            {'b': '(:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list12-7',
        feature_path='tck/features/expressions/list/List12.feature',
        scenario='[7] Fail when using aggregation in list comprehension',
        cypher='MATCH (n)\n      RETURN [x IN [1, 2, 3, 4, 5] | count(*)]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
