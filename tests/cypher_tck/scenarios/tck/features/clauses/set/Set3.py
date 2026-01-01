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
        key='set3-1',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[1] Add a single label to a node with no label',
        cypher='MATCH (n)\n      SET n:Foo\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:Foo)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-2',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[2] Adding multiple labels to a node with no label',
        cypher='MATCH (n)\n      SET n:Foo:Bar\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:Foo:Bar)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-3',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[3] Add a single label to a node with an existing label',
        cypher='MATCH (n:A)\n      SET n:Foo\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A:Foo)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-4',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[4] Adding multiple labels to a node with an existing label',
        cypher='MATCH (n)\n      SET n:Foo:Bar\n      RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A:Foo:Bar)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-5',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[5] Ignore whitespace before colon 1',
        cypher='MATCH (n)\n      SET n :Foo\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['Foo']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-6',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[6] Ignore whitespace before colon 2',
        cypher='MATCH (n)\n      SET n :Foo :Bar\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['Foo', 'Bar']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-7',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[7] Ignore whitespace before colon 3',
        cypher='MATCH (n)\n      SET n :Foo:Bar\n      RETURN labels(n)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'labels(n)': "['Foo', 'Bar']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set3-8',
        feature_path='tck/features/clauses/set/Set3.feature',
        scenario='[8] Ignore null when setting label',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      SET a:L\n      RETURN a',
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
