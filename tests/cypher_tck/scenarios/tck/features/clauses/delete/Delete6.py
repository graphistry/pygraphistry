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
        key='delete6-1',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[1] Limiting to zero results after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      DELETE n\n      RETURN 42 AS num\n      LIMIT 0',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-2',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[2] Skipping all results after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      DELETE n\n      RETURN 42 AS num\n      SKIP 1',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-3',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[3] Skipping and limiting to a few results after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      DELETE n\n      RETURN 42 AS num\n      SKIP 2 LIMIT 2',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-4',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[4] Skipping zero results and limiting to all results after deleting nodes does not affect the result set nor the side effects',
        cypher='MATCH (n:N)\n      DELETE n\n      RETURN 42 AS num\n      SKIP 0 LIMIT 5',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-5',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[5] Filtering after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      WITH n, n.num AS num\n      DELETE n\n      WITH num\n      WHERE num % 2 = 0\n      RETURN num',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-6',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[6] Aggregating in `RETURN` after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      WITH n, n.num AS num\n      DELETE n\n      RETURN sum(num) AS sum',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-7',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[7] Aggregating in `WITH` after deleting nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      WITH n, n.num AS num\n      DELETE n\n      WITH sum(num) AS sum\n      RETURN sum',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-8',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[8] Limiting to zero results after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      DELETE r\n      RETURN 42 AS num\n      LIMIT 0',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-9',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[9] Skipping all results after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      DELETE r\n      RETURN 42 AS num\n      SKIP 1',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-10',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[10] Skipping and limiting to a few results after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      DELETE r\n      RETURN 42 AS num\n      SKIP 2 LIMIT 2',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-11',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[11] Skipping zero result and limiting to all results after deleting relationships does not affect the result set nor the side effects',
        cypher='MATCH ()-[r:R]->()\n      DELETE r\n      RETURN 42 AS num\n      SKIP 0 LIMIT 5',
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
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-12',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[12] Filtering after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      WITH r, r.num AS num\n      DELETE r\n      WITH num\n      WHERE num % 2 = 0\n      RETURN num',
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
            {'num': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-13',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[13] Aggregating in `RETURN` after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      WITH r, r.num AS num\n      DELETE r\n      RETURN sum(num) AS sum',
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
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete6-14',
        feature_path='tck/features/clauses/delete/Delete6.feature',
        scenario='[14] Aggregating in `WITH` after deleting relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      WITH r, r.num AS num\n      DELETE r\n      WITH sum(num) AS sum\n      RETURN sum',
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
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),
]
