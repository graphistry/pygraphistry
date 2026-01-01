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
        key='delete2-1',
        feature_path='tck/features/clauses/delete/Delete2.feature',
        scenario='[1] Delete relationships',
        cypher='MATCH ()-[r]-()\n      DELETE r',
        graph=graph_fixture_from_create(
            """
            UNWIND range(0, 2) AS i
                  CREATE ()-[:R]->()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete2-2',
        feature_path='tck/features/clauses/delete/Delete2.feature',
        scenario='[2] Delete optionally matched relationship',
        cypher='MATCH (n)\n      OPTIONAL MATCH (n)-[r]-()\n      DELETE n, r',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete2-3',
        feature_path='tck/features/clauses/delete/Delete2.feature',
        scenario='[3] Delete relationship with bidirectional matching',
        cypher='MATCH p = ()-[r:T]-()\n      WHERE r.id = 42\n      DELETE r',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {id: 42}]->()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete2-4',
        feature_path='tck/features/clauses/delete/Delete2.feature',
        scenario='[4] Ignore null when deleting relationship',
        cypher='OPTIONAL MATCH ()-[r:DoesNotExist]-()\n      DELETE r\n      RETURN r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='DELETE clause semantics are not supported',
        tags=('delete', 'xfail'),
    ),

    Scenario(
        key='delete2-5',
        feature_path='tck/features/clauses/delete/Delete2.feature',
        scenario='[5] Failing when deleting a relationship type',
        cypher='MATCH ()-[r:T]-()\n      DELETE r:T',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {id: 42}]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('delete', 'syntax-error', 'xfail'),
    ),
]
