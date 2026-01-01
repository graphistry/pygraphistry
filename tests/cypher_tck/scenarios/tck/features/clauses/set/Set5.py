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
        key='set5-1',
        feature_path='tck/features/clauses/set/Set5.feature',
        scenario='[1] Ignore null when setting properties using an appending map',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      SET a += {num: 42}\n      RETURN a',
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

    Scenario(
        key='set5-2',
        feature_path='tck/features/clauses/set/Set5.feature',
        scenario='[2] Overwrite values when using +=',
        cypher="MATCH (n:X {name: 'A'})\n      SET n += {name2: 'C'}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'A', name2: 'C'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set5-3',
        feature_path='tck/features/clauses/set/Set5.feature',
        scenario='[3] Retain old values when using +=',
        cypher="MATCH (n:X {name: 'A'})\n      SET n += {name2: 'B'}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'A', name2: 'B'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set5-4',
        feature_path='tck/features/clauses/set/Set5.feature',
        scenario='[4] Explicit null values in a map remove old values',
        cypher="MATCH (n:X {name: 'A'})\n      SET n += {name: null}\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name2: 'B'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set5-5',
        feature_path='tck/features/clauses/set/Set5.feature',
        scenario='[5] Set an empty map when using += has no effect',
        cypher="MATCH (n:X {name: 'A'})\n      SET n += { }\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:X {name: 'A', name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:X {name: 'A', name2: 'B'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),
]
