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
        key='create2-1',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[1] Create two nodes and a single relationship in a single pattern',
        cypher='CREATE ()-[:R]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-2',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[2] Create two nodes and a single relationship in separate patterns',
        cypher='CREATE (a), (b),\n             (a)-[:R]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-3',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[3] Create two nodes and a single relationship in separate clauses',
        cypher='CREATE (a)\n      CREATE (b)\n      CREATE (a)-[:R]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-4',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[4] Create two nodes and a single relationship in the reverse direction',
        cypher='CREATE (:A)<-[:R]-(:B)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-5',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[5] Create a single relationship between two existing nodes',
        cypher='MATCH (x:X), (y:Y)\n      CREATE (x)-[:R]->(y)',
        graph=graph_fixture_from_create(
            """
            CREATE (:X)
                  CREATE (:Y)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-6',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[6] Create a single relationship between two existing nodes in the reverse direction',
        cypher='MATCH (x:X), (y:Y)\n      CREATE (x)<-[:R]-(y)',
        graph=graph_fixture_from_create(
            """
            CREATE (:X)
                  CREATE (:Y)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-7',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[7] Create a single node and a single self loop in a single pattern',
        cypher='CREATE (root)-[:LINK]->(root)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-8',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[8] Create a single node and a single self loop in separate patterns',
        cypher='CREATE (root),\n             (root)-[:LINK]->(root)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-9',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[9] Create a single node and a single self loop in separate clauses',
        cypher='CREATE (root)\n      CREATE (root)-[:LINK]->(root)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-10',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[10] Create a single self loop on an existing node',
        cypher='MATCH (root:Root)\n      CREATE (root)-[:LINK]->(root)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Root)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-11',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[11] Create a single relationship and an end node on an existing starting node',
        cypher='MATCH (x:Begin)\n      CREATE (x)-[:TYPE]->(:End)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Begin)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-12',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[12] Create a single relationship and a starting node on an existing end node',
        cypher='MATCH (x:End)\n      CREATE (:Begin)-[:TYPE]->(x)',
        graph=graph_fixture_from_create(
            """
            CREATE (:End)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-13',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[13] Create a single relationship with a property',
        cypher='CREATE ()-[:R {num: 42}]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-14',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[14] Create a single relationship with a property and return it',
        cypher='CREATE ()-[r:R {num: 42}]->()\n      RETURN r.num AS num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-15',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[15] Create a single relationship with two properties',
        cypher="CREATE ()-[:R {id: 12, name: 'foo'}]->()",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-16',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[16] Create a single relationship with two properties and return them',
        cypher="CREATE ()-[r:R {id: 12, name: 'foo'}]->()\n      RETURN r.id AS id, r.name AS name",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'id': 12, 'name': "'foo'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-17',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[17] Create a single relationship with null properties should not return those properties',
        cypher='CREATE ()-[r:X {id: 12, name: null}]->()\n      RETURN r.id, r.name AS name',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r.id': 12, 'name': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create2-18',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[18] Fail when creating a relationship without a type',
        cypher='CREATE ()-->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-19',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[19] Fail when creating a relationship without a direction',
        cypher='CREATE (a)-[:FOO]-(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-20',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[20] Fail when creating a relationship with two directions',
        cypher='CREATE (a)<-[:FOO]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-21',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[21] Fail when creating a relationship with more than one type',
        cypher='CREATE ()-[:A|:B]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-22',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[22] Fail when creating a variable-length relationship',
        cypher='CREATE ()-[:FOO*2]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-23',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[23] Fail when creating a relationship that is already bound',
        cypher='MATCH ()-[r]->()\n      CREATE ()-[r]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='create2-24',
        feature_path='tck/features/clauses/create/Create2.feature',
        scenario='[24] Fail when creating a relationship using undefined variable in pattern',
        cypher='MATCH (a)\n      CREATE (a)-[:KNOWS]->(b {name: missing})\n      RETURN b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('create', 'syntax-error', 'xfail'),
    ),
]
