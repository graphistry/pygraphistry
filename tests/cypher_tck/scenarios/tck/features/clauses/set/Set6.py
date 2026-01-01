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
        key='set6-1',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[1] Limiting to zero results after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = 43\n      RETURN n\n      LIMIT 0',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-2',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[2] Skipping all results after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = 43\n      RETURN n\n      SKIP 1',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-3',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[3] Skipping and limiting to a few results after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = 42\n      RETURN n.num AS num\n      SKIP 2 LIMIT 2',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-4',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[4] Skipping zero results and limiting to all results after setting a property on nodes does not affect the result set nor the side effects',
        cypher='MATCH (n:N)\n      SET n.num = 42\n      RETURN n.num AS num\n      SKIP 0 LIMIT 5',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-5',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[5] Filtering after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = n.num + 1\n      WITH n\n      WHERE n.num % 2 = 0\n      RETURN n.num AS num',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 2},
            {'num': 4},
            {'num': 6}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-6',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[6] Aggregating in `RETURN` after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = n.num + 1\n      RETURN sum(n.num) AS sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 20}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-7',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[7] Aggregating in `WITH` after setting a property on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n.num = n.num + 1\n      WITH sum(n.num) AS sum\n      RETURN sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 20}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-8',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[8] Limiting to zero results after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      RETURN n\n      LIMIT 0',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-9',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[9] Skipping all results after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      RETURN n\n      SKIP 1',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-10',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[10] Skipping and limiting to a few results after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      RETURN n.num AS num\n      SKIP 2 LIMIT 2',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-11',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[11] Skipping zero result and limiting to all results after adding a label on nodes does not affect the result set nor the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      RETURN n.num AS num\n      SKIP 0 LIMIT 5',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
                  CREATE (:N {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-12',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[12] Filtering after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      WITH n\n      WHERE n.num % 2 = 0\n      RETURN n.num AS num',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'num': 2},
            {'num': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-13',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[13] Aggregating in `RETURN` after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      RETURN sum(n.num) AS sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-14',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[14] Aggregating in `WITH` after adding a label on nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      SET n:Foo\n      WITH sum(n.num) AS sum\n      RETURN sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {num: 1})
                  CREATE (:N {num: 2})
                  CREATE (:N {num: 3})
                  CREATE (:N {num: 4})
                  CREATE (:N {num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-15',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[15] Limiting to zero results after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = 43\n      RETURN r\n      LIMIT 0',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[r:R {num: 42}]->()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-16',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[16] Skipping all results after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = 43\n      RETURN r\n      SKIP 1',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[r:R {num: 42}]->()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-17',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[17] Skipping and limiting to a few results after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = 42\n      RETURN r.num AS num\n      SKIP 2 LIMIT 2',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {num: 1}]->()
                  CREATE ()-[:R {num: 2}]->()
                  CREATE ()-[:R {num: 3}]->()
                  CREATE ()-[:R {num: 4}]->()
                  CREATE ()-[:R {num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-18',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[18] Skipping zero result and limiting to all results after setting a property on relationships does not affect the result set nor the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = 42\n      RETURN r.num AS num\n      SKIP 0 LIMIT 5',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {num: 1}]->()
                  CREATE ()-[:R {num: 2}]->()
                  CREATE ()-[:R {num: 3}]->()
                  CREATE ()-[:R {num: 4}]->()
                  CREATE ()-[:R {num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-19',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[19] Filtering after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = r.num + 1\n      WITH r\n      WHERE r.num % 2 = 0\n      RETURN r.num AS num',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {num: 1}]->()
                  CREATE ()-[:R {num: 2}]->()
                  CREATE ()-[:R {num: 3}]->()
                  CREATE ()-[:R {num: 4}]->()
                  CREATE ()-[:R {num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'num': 2},
            {'num': 4},
            {'num': 6}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-20',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[20] Aggregating in `RETURN` after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = r.num + 1\n      RETURN sum(r.num) AS sum',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {num: 1}]->()
                  CREATE ()-[:R {num: 2}]->()
                  CREATE ()-[:R {num: 3}]->()
                  CREATE ()-[:R {num: 4}]->()
                  CREATE ()-[:R {num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 20}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set6-21',
        feature_path='tck/features/clauses/set/Set6.feature',
        scenario='[21] Aggregating in `WITH` after setting a property on relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      SET r.num = r.num + 1\n      WITH sum(r.num) AS sum\n      RETURN sum',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {num: 1}]->()
                  CREATE ()-[:R {num: 2}]->()
                  CREATE ()-[:R {num: 3}]->()
                  CREATE ()-[:R {num: 4}]->()
                  CREATE ()-[:R {num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 20}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),
]
