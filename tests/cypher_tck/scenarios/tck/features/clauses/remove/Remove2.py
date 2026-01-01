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
        key='remove2-1',
        feature_path='tck/features/clauses/remove/Remove2.feature',
        scenario='[1] Remove a single label from a node with a single label',
        cypher='MATCH (n)\n      REMOVE n:L\n      RETURN n.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:L {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'n.num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove2-2',
        feature_path='tck/features/clauses/remove/Remove2.feature',
        scenario='[2] Remove a single label from a node with two labels',
        cypher='MATCH (n)\n      REMOVE n:Foo\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Foo:Bar)
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['Bar']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove2-3',
        feature_path='tck/features/clauses/remove/Remove2.feature',
        scenario='[3] Remove two labels from a node with three labels',
        cypher='MATCH (n)\n      REMOVE n:L1:L3\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE (:L1:L2:L3 {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['L2']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove2-4',
        feature_path='tck/features/clauses/remove/Remove2.feature',
        scenario='[4] Remove a non-existent node label',
        cypher='MATCH (n)\n      REMOVE n:Bar\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE (:Foo)
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['Foo']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='REMOVE clause semantics are not supported',
        tags=('remove', 'xfail'),
    ),

    Scenario(
        key='remove2-5',
        feature_path='tck/features/clauses/remove/Remove2.feature',
        scenario='[5] Ignore null when removing a node label',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      REMOVE a:L\n      RETURN a',
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
]
