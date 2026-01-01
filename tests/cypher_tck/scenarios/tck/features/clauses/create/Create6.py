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
        key='create6-1',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[1] Limiting to zero results after creating nodes affects the result set but not the side effects',
        cypher='CREATE (n:N {num: 42})\n      RETURN n\n      LIMIT 0',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-2',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[2] Skipping all results after creating nodes affects the result set but not the side effects',
        cypher='CREATE (n:N {num: 42})\n      RETURN n\n      SKIP 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-3',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[3] Skipping and limiting to a few results after creating nodes does not affect the result set nor the side effects',
        cypher='UNWIND [42, 42, 42, 42, 42] AS x\n      CREATE (n:N {num: x})\n      RETURN n.num AS num\n      SKIP 2 LIMIT 2',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-4',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[4] Skipping zero result and limiting to all results after creating nodes does not affect the result set nor the side effects',
        cypher='UNWIND [42, 42, 42, 42, 42] AS x\n      CREATE (n:N {num: x})\n      RETURN n.num AS num\n      SKIP 0 LIMIT 5',
        graph=GraphFixture(nodes=[], edges=[]),
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
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-5',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[5] Filtering after creating nodes affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE (n:N {num: x})\n      WITH n\n      WHERE n.num % 2 = 0\n      RETURN n.num AS num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'num': 2},
            {'num': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-6',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[6] Aggregating in `RETURN` after creating nodes affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE (n:N {num: x})\n      RETURN sum(n.num) AS sum',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-7',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[7] Aggregating in `WITH` after creating nodes affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE (n:N {num: x})\n      WITH sum(n.num) AS sum\n      RETURN sum',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-8',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[8] Limiting to zero results after creating relationships affects the result set but not the side effects',
        cypher='CREATE ()-[r:R {num: 42}]->()\n      RETURN r\n      LIMIT 0',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-9',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[9] Skipping all results after creating relationships affects the result set but not the side effects',
        cypher='CREATE ()-[r:R {num: 42}]->()\n      RETURN r\n      SKIP 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-10',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[10] Skipping and limiting to a few results after creating relationships does not affect the result set nor the side effects',
        cypher='UNWIND [42, 42, 42, 42, 42] AS x\n      CREATE ()-[r:R {num: x}]->()\n      RETURN r.num AS num\n      SKIP 2 LIMIT 2',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'num': 42},
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-11',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[11] Skipping zero result and limiting to all results after creating relationships does not affect the result set nor the side effects',
        cypher='UNWIND [42, 42, 42, 42, 42] AS x\n      CREATE ()-[r:R {num: x}]->()\n      RETURN r.num AS num\n      SKIP 0 LIMIT 5',
        graph=GraphFixture(nodes=[], edges=[]),
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
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-12',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[12] Filtering after creating relationships affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE ()-[r:R {num: x}]->()\n      WITH r\n      WHERE r.num % 2 = 0\n      RETURN r.num AS num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'num': 2},
            {'num': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-13',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[13] Aggregating in `RETURN` after creating relationships affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE ()-[r:R {num: x}]->()\n      RETURN sum(r.num) AS sum',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create6-14',
        feature_path='tck/features/clauses/create/Create6.feature',
        scenario='[14] Aggregating in `WITH` after creating relationships affects the result set but not the side effects',
        cypher='UNWIND [1, 2, 3, 4, 5] AS x\n      CREATE ()-[r:R {num: x}]->()\n      WITH sum(r.num) AS sum\n      RETURN sum',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'sum': 15}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),
]
