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
        key='expr-path3-1',
        feature_path='tck/features/expressions/path/Path3.feature',
        scenario='[1] Return a var length path of length zero',
        cypher='MATCH p = (a)-[*0..1]->(b)\n      RETURN a, b, length(p) AS l',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL]->(b:B)
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A)', 'b': '(:A)', 'l': 0},
            {'a': '(:B)', 'b': '(:B)', 'l': 0},
            {'a': '(:A)', 'b': '(:B)', 'l': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-path3-2',
        feature_path='tck/features/expressions/path/Path3.feature',
        scenario='[2] Failing when using `length()` on a node',
        cypher='MATCH (n)\n      RETURN length(n)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-path3-3',
        feature_path='tck/features/expressions/path/Path3.feature',
        scenario='[3] Failing when using `length()` on a relationship',
        cypher='MATCH ()-[r]->()\n      RETURN length(r)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
