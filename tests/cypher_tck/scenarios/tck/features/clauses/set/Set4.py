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
        key='set4-1',
        feature_path='tck/features/clauses/set/Set4.feature',
        scenario='[1] Set multiple properties with a property map',
        cypher="MATCH (n:X)\n      SET n = {name: 'A', name2: 'B', num: 5}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X)
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'A', name2: 'B', num: 5})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set4-2',
        feature_path='tck/features/clauses/set/Set4.feature',
        scenario='[2] Non-existent values in a property map are removed with SET',
        cypher="MATCH (n:X {name: 'A'})\n      SET n = {name: 'B', baz: 'C'}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'B', baz: 'C'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set4-3',
        feature_path='tck/features/clauses/set/Set4.feature',
        scenario='[3] Null values in a property map are removed with SET',
        cypher="MATCH (n:X {name: 'A'})\n      SET n = {name: 'B', name2: null, baz: 'C'}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'B', baz: 'C'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set4-4',
        feature_path='tck/features/clauses/set/Set4.feature',
        scenario='[4] All properties are removed if node is set to empty property map',
        cypher="MATCH (n:X {name: 'A'})\n      SET n = { }\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:X)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set4-5',
        feature_path='tck/features/clauses/set/Set4.feature',
        scenario='[5] Ignore null when setting properties using an overriding map',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      SET a = {num: 42}\n      RETURN a',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),
]
