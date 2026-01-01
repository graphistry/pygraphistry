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
        key='delete5-1',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[1] Delete node from a list',
        cypher='MATCH (:User)-[:FRIEND]->(n)\n      WITH collect(n) AS friends\n      DETACH DELETE friends[$friendIndex]',
        graph=graph_fixture_from_create(
            """
            CREATE (u:User)
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('delete', 'params', 'xfail'),
    ),

    Scenario(
        key='delete5-2',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[2] Delete relationship from a list',
        cypher='MATCH (:User)-[r:FRIEND]->()\n      WITH collect(r) AS friendships\n      DETACH DELETE friendships[$friendIndex]',
        graph=graph_fixture_from_create(
            """
            CREATE (u:User)
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
                  CREATE (u)-[:FRIEND]->()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('delete', 'params', 'xfail'),
    ),

    Scenario(
        key='delete5-3',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[3] Delete nodes from a map',
        cypher='MATCH (u:User)\n      WITH {key: u} AS nodes\n      DELETE nodes.key',
        graph=graph_fixture_from_create(
            """
            CREATE (:User), (:User)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete5-4',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[4] Delete relationships from a map',
        cypher='MATCH (:User)-[r]->(:User)\n      WITH {key: r} AS rels\n      DELETE rels.key',
        graph=graph_fixture_from_create(
            """
            CREATE (a:User), (b:User)
                  CREATE (a)-[:R]->(b)
                  CREATE (b)-[:R]->(a)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete5-5',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[5] Detach delete nodes from nested map/list',
        cypher='MATCH (u:User)\n      WITH {key: collect(u)} AS nodeMap\n      DETACH DELETE nodeMap.key[0]',
        graph=graph_fixture_from_create(
            """
            CREATE (a:User), (b:User)
                  CREATE (a)-[:R]->(b)
                  CREATE (b)-[:R]->(a)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete5-6',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[6] Delete relationships from nested map/list',
        cypher='MATCH (:User)-[r]->(:User)\n      WITH {key: {key: collect(r)}} AS rels\n      DELETE rels.key.key[0]',
        graph=graph_fixture_from_create(
            """
            CREATE (a:User), (b:User)
                  CREATE (a)-[:R]->(b)
                  CREATE (b)-[:R]->(a)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete5-7',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[7] Delete paths from nested map/list',
        cypher='MATCH p = (:User)-[r]->(:User)\n      WITH {key: collect(p)} AS pathColls\n      DELETE pathColls.key[0], pathColls.key[1]',
        graph=graph_fixture_from_create(
            """
            CREATE (a:User), (b:User)
                  CREATE (a)-[:R]->(b)
                  CREATE (b)-[:R]->(a)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete5-8',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[8] Failing when using undefined variable in DELETE',
        cypher='MATCH (a)\n      DELETE x',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('delete', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='delete5-9',
        feature_path='tck/features/clauses/delete/Delete5.feature',
        scenario='[9] Failing when deleting an integer expression',
        cypher='MATCH ()\n      DELETE 1 + 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('delete', 'syntax-error', 'xfail'),
    ),
]
