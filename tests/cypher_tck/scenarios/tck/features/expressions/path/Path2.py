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
        key='expr-path2-1',
        feature_path='tck/features/expressions/path/Path2.feature',
        scenario='[1] Return relationships by fetching them from the path',
        cypher='MATCH p = (a:Start)-[:REL*2..2]->(b)\n      RETURN relationships(p)',
        graph=graph_fixture_from_create(
            """
            CREATE (s:Start)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:C)
            """
        ),
        expected=Expected(
            rows=[
            {'relationships(p)': '[[:REL {num: 1}], [:REL {num: 2}]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-path2-2',
        feature_path='tck/features/expressions/path/Path2.feature',
        scenario='[2] Return relationships by fetching them from the path - starting from the end',
        cypher='MATCH p = (a)-[:REL*2..2]->(b:End)\n      RETURN relationships(p)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(e:End)
            """
        ),
        expected=Expected(
            rows=[
            {'relationships(p)': '[[:REL {num: 1}], [:REL {num: 2}]]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-path2-3',
        feature_path='tck/features/expressions/path/Path2.feature',
        scenario='[3] `relationships()` on null path',
        cypher='WITH null AS a\n      OPTIONAL MATCH p = (a)-[r]->()\n      RETURN relationships(p), relationships(null)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'relationships(p)': 'null', 'relationships(null)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'path', 'meta-xfail', 'xfail'),
    ),
]
