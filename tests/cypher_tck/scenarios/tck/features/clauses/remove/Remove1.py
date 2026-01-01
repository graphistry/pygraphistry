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
        key='remove1-1',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[1] Remove a single node property',
        cypher='MATCH (n)\n      REMOVE n.num\n      RETURN n.num IS NOT NULL AS still_there',
        graph=graph_fixture_from_create(
            """
            CREATE (:L {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'still_there': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-2',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[2] Remove multiple node properties',
        cypher='MATCH (n)\n      REMOVE n.num, n.name\n      RETURN size(keys(n)) AS props',
        graph=graph_fixture_from_create(
            """
            CREATE (:L {num: 42, name: 'a', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'props': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-3',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[3] Remove a single relationship property',
        cypher='MATCH ()-[r]->()\n      REMOVE r.num\n      RETURN r.num IS NOT NULL AS still_there',
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (a)-[:X {num: 42}]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'still_there': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-4',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[4] Remove multiple relationship properties',
        cypher='MATCH ()-[r]->()\n      REMOVE r.num, r.a\n      RETURN size(keys(r)) AS props',
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (a)-[:X {num: 42, a: 'a', b: 'B'}]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'props': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-5',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[5] Ignore null when removing property from a node',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      REMOVE a.num\n      RETURN a',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-6',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[6] Ignore null when removing property from a relationship',
        cypher='MATCH (n)\n      OPTIONAL MATCH (n)-[r]->()\n      REMOVE r.num\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n': '({num: 42})'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove1-7',
        feature_path='tck/features/clauses/remove/Remove1.feature',
        scenario='[7] Remove a missing node property',
        cypher='MATCH (n)\n      REMOVE n.num\n      RETURN sum(size(keys(n))) AS totalNumberOfProps',
        graph=graph_fixture_from_create(
            """
            CREATE (), (), ()
            """
        ),
        expected=Expected(
            rows=[
            {'totalNumberOfProps': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),
]
