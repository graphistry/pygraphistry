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
        key='create3-1',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[1] MATCH-CREATE',
        cypher='MATCH ()\n      CREATE ()',
        graph=graph_fixture_from_create(
            """
            CREATE (), ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-2',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[2] WITH-CREATE',
        cypher='MATCH ()\n      CREATE ()\n      WITH *\n      CREATE ()',
        graph=graph_fixture_from_create(
            """
            CREATE (), ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-3',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[3] MATCH-CREATE-WITH-CREATE',
        cypher='MATCH ()\n      CREATE ()\n      WITH *\n      MATCH ()\n      CREATE ()',
        graph=graph_fixture_from_create(
            """
            CREATE (), ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-4',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[4] MATCH-CREATE: Newly-created nodes not visible to preceding MATCH',
        cypher='MATCH ()\n      CREATE ()',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-5',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[5] WITH-CREATE: Nodes are not created when aliases are applied to variable names',
        cypher='MATCH (n)\n      MATCH (m)\n      WITH n AS a, m AS b\n      CREATE (a)-[:T]->(b)\n      RETURN a, b',
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
            {'a': '({num: 1})', 'b': '({num: 1})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-6',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[6] WITH-CREATE: Only a single node is created when an alias is applied to a variable name',
        cypher='MATCH (n)\n      WITH n AS a\n      CREATE (a)-[:T]->()\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:X)
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:X)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-7',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[7] WITH-CREATE: Nodes are not created when aliases are applied to variable names multiple times',
        cypher='MATCH (n)\n      MATCH (m)\n      WITH n AS a, m AS b\n      CREATE (a)-[:T]->(b)\n      WITH a AS x, b AS y\n      CREATE (x)-[:T]->(y)\n      RETURN x, y',
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'})
            """
        ),
        expected=Expected(
            rows=[
            {'x': "({name: 'A'})", 'y': "({name: 'A'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-8',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[8] WITH-CREATE: Only a single node is created when an alias is applied to a variable name multiple times',
        cypher='MATCH (n)\n      WITH n AS a\n      CREATE (a)-[:T]->()\n      WITH a AS x\n      CREATE (x)-[:T]->()\n      RETURN x',
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 5})
            """
        ),
        expected=Expected(
            rows=[
            {'x': '({num: 5})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-9',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[9] WITH-CREATE: A bound node should be recognized after projection with WITH + WITH',
        cypher='CREATE (a)\n      WITH a\n      WITH *\n      CREATE (b)\n      CREATE (a)<-[:T]-(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-10',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[10] WITH-UNWIND-CREATE: A bound node should be recognized after projection with WITH + UNWIND',
        cypher='CREATE (a)\n      WITH a\n      UNWIND [0] AS i\n      CREATE (b)\n      CREATE (a)<-[:T]-(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-11',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[11] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE node',
        cypher='CREATE (a)\n      WITH a\n      MERGE ()\n      CREATE (b)\n      CREATE (a)<-[:T]-(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-12',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[12] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE pattern',
        cypher='CREATE (a)\n      WITH a\n      MERGE (x)\n      MERGE (y)\n      MERGE (x)-[:T]->(y)\n      CREATE (b)\n      CREATE (a)<-[:T]-(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create3-13',
        feature_path='tck/features/clauses/create/Create3.feature',
        scenario='[13] Merge followed by multiple creates',
        cypher='MERGE (t:T {id: 42})\n      CREATE (f:R)\n      CREATE (t)-[:REL]->(f)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),
]
