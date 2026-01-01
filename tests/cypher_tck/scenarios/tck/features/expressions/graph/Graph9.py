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
        key='expr-graph9-1',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[1] `properties()` on a node',
        cypher='MATCH (p:Person)\n      RETURN properties(p) AS m',
        graph=graph_fixture_from_create(
            """
            CREATE (n:Person {name: 'Popeye', level: 9001})
            """
        ),
        expected=Expected(
            rows=[
            {'m': "{name: 'Popeye', level: 9001}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-2',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[2] `properties()` on a relationship',
        cypher='MATCH ()-[r:R]->()\n      RETURN properties(r) AS m',
        graph=graph_fixture_from_create(
            """
            CREATE (n)-[:R {name: 'Popeye', level: 9001}]->(n)
            """
        ),
        expected=Expected(
            rows=[
            {'m': "{name: 'Popeye', level: 9001}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-3',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[3] `properties()` on null',
        cypher='OPTIONAL MATCH (n:DoesNotExist)\n      OPTIONAL MATCH (n)-[r:NOT_THERE]->()\n      RETURN properties(n), properties(r), properties(null)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'properties(n)': 'null', 'properties(r)': 'null', 'properties(null)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-4',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[4] `properties()` on a map',
        cypher="RETURN properties({name: 'Popeye', level: 9001}) AS m",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'m': "{name: 'Popeye', level: 9001}"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-5',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[5] `properties()` failing on an integer literal',
        cypher='RETURN properties(1)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-6',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[6] `properties()` failing on a string literal',
        cypher="RETURN properties('Cypher')",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph9-7',
        feature_path='tck/features/expressions/graph/Graph9.feature',
        scenario='[7] `properties()` failing on a list of booleans',
        cypher='RETURN properties([true, false])',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
