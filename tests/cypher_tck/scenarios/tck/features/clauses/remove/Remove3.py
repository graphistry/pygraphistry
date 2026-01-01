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
        key='remove3-1',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[1] Limiting to zero results after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.num\n      RETURN n\n      LIMIT 0',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-2',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[2] Skipping all results after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.num\n      RETURN n\n      SKIP 1',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-3',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[3] Skipping and limiting to a few results after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.name\n      RETURN n.num AS num\n      SKIP 2 LIMIT 2',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-4',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[4] Skipping zero results and limiting to all results after removing a property from nodes does not affect the result set nor the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.name\n      RETURN n.num AS num\n      SKIP 0 LIMIT 5',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
                  CREATE (:N {name: 'a', num: 42})
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-5',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[5] Filtering after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.name\n      WITH n\n      WHERE n.num % 2 = 0\n      RETURN n.num AS num',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {name: 'a', num: 1})
                  CREATE (:N {name: 'a', num: 2})
                  CREATE (:N {name: 'a', num: 3})
                  CREATE (:N {name: 'a', num: 4})
                  CREATE (:N {name: 'a', num: 5})
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-6',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[6] Aggregating in `RETURN` after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.name\n      RETURN sum(n.num) AS sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {name: 'a', num: 1})
                  CREATE (:N {name: 'a', num: 2})
                  CREATE (:N {name: 'a', num: 3})
                  CREATE (:N {name: 'a', num: 4})
                  CREATE (:N {name: 'a', num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-7',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[7] Aggregating in `WITH` after removing a property from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n.name\n      WITH sum(n.num) AS sum\n      RETURN sum',
        graph=graph_fixture_from_create(
            """
            CREATE (:N {name: 'a', num: 1})
                  CREATE (:N {name: 'a', num: 2})
                  CREATE (:N {name: 'a', num: 3})
                  CREATE (:N {name: 'a', num: 4})
                  CREATE (:N {name: 'a', num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-8',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[8] Limiting to zero results after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      RETURN n\n      LIMIT 0',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-9',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[9] Skipping all results after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      RETURN n\n      SKIP 1',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-10',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[10] Skipping and limiting to a few results after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      RETURN n.num AS num\n      SKIP 2 LIMIT 2',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-11',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[11] Skipping zero result and limiting to all results after removing a label from nodes does not affect the result set nor the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      RETURN n.num AS num\n      SKIP 0 LIMIT 5',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-12',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[12] Filtering after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      WITH n\n      WHERE n.num % 2 = 0\n      RETURN n.num AS num',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-13',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[13] Aggregating in `RETURN` after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      RETURN sum(n.num) AS sum',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-14',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[14] Aggregating in `WITH` after removing a label from nodes affects the result set but not the side effects',
        cypher='MATCH (n:N)\n      REMOVE n:N\n      WITH sum(n.num) AS sum\n      RETURN sum',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-15',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[15] Limiting to zero results after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.num\n      RETURN r\n      LIMIT 0',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-16',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[16] Skipping all results after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.num\n      RETURN r\n      SKIP 1',
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-17',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[17] Skipping and limiting to a few results after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.name\n      RETURN r.num AS num\n      SKIP 2 LIMIT 2',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-18',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[18] Skipping zero result and limiting to all results after removing a property from relationships does not affect the result set nor the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.name\n      RETURN r.num AS num\n      SKIP 0 LIMIT 5',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
                  CREATE ()-[:R {name: 'a', num: 42}]->()
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-19',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[19] Filtering after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.name\n      WITH r\n      WHERE r.num % 2 = 0\n      RETURN r.num AS num',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {name: 'a', num: 1}]->()
                  CREATE ()-[:R {name: 'a', num: 2}]->()
                  CREATE ()-[:R {name: 'a', num: 3}]->()
                  CREATE ()-[:R {name: 'a', num: 4}]->()
                  CREATE ()-[:R {name: 'a', num: 5}]->()
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
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-20',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[20] Aggregating in `RETURN` after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.name\n      RETURN sum(r.num) AS sum',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {name: 'a', num: 1}]->()
                  CREATE ()-[:R {name: 'a', num: 2}]->()
                  CREATE ()-[:R {name: 'a', num: 3}]->()
                  CREATE ()-[:R {name: 'a', num: 4}]->()
                  CREATE ()-[:R {name: 'a', num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove3-21',
        feature_path='tck/features/clauses/remove/Remove3.feature',
        scenario='[21] Aggregating in `WITH` after removing a property from relationships affects the result set but not the side effects',
        cypher='MATCH ()-[r:R]->()\n      REMOVE r.name\n      WITH sum(r.num) AS sum\n      RETURN sum',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:R {name: 'a', num: 1}]->()
                  CREATE ()-[:R {name: 'a', num: 2}]->()
                  CREATE ()-[:R {name: 'a', num: 3}]->()
                  CREATE ()-[:R {name: 'a', num: 4}]->()
                  CREATE ()-[:R {name: 'a', num: 5}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),
]
