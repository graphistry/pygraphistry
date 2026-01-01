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
        key='set1-1',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[1] Set a property',
        cypher="MATCH (n:A)\n      WHERE n.name = 'Andres'\n      SET n.name = 'Michael'\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'Andres'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:A {name: 'Michael'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-2',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[2] Set a property to an expression',
        cypher="MATCH (n:A)\n      WHERE n.name = 'Andres'\n      SET n.name = n.name + ' was here'\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'Andres'})
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:A {name: 'Andres was here'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-3',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[3] Set a property by selecting the node using a simple expression',
        cypher="MATCH (n:A)\n      SET (n).name = 'neo4j'\n      RETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(
            rows=[
            {'n': "(:A {name: 'neo4j'})"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-4',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[4] Set a property by selecting the relationship using a simple expression',
        cypher="MATCH ()-[r:REL]->()\n      SET (r).name = 'neo4j'\n      RETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
            {'r': "[:REL {name: 'neo4j'}]"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-5',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[5] Adding a list property',
        cypher='MATCH (n:A)\n      SET n.numbers = [1, 2, 3]\n      RETURN [i IN n.numbers | i / 2.0] AS x',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(
            rows=[
            {'x': '[0.5, 1.0, 1.5]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-6',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[6] Concatenate elements onto a list property',
        cypher='CREATE (a {numbers: [1, 2, 3]})\n      SET a.numbers = a.numbers + [4, 5]\n      RETURN a.numbers',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a.numbers': '[1, 2, 3, 4, 5]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-7',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[7] Concatenate elements in reverse onto a list property',
        cypher='CREATE (a {numbers: [3, 4, 5]})\n      SET a.numbers = [1, 2] + a.numbers\n      RETURN a.numbers',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a.numbers': '[1, 2, 3, 4, 5]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='SET clause semantics are not supported',
        tags=('set', 'xfail'),
    ),

    Scenario(
        key='set1-8',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[8] Ignore null when setting property',
        cypher='OPTIONAL MATCH (a:DoesNotExist)\n      SET a.num = 42\n      RETURN a',
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
        key='set1-9',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[9] Failing when using undefined variable in SET',
        cypher='MATCH (a)\n      SET a.name = missing\n      RETURN a',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('set', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='set1-10',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[10] Failing when setting a list of maps as a property',
        cypher='CREATE (a)\n      SET a.maplist = [{num: 1}]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('set', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='set1-11',
        feature_path='tck/features/clauses/set/Set1.feature',
        scenario='[11] Set multiple node properties',
        cypher="MATCH (n:X)\n      SET n.name = 'A', n.name2 = 'B', n.num = 5\n      RETURN n",
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
]
