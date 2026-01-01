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
        key='expr-aggregation5-1',
        feature_path='tck/features/expressions/aggregation/Aggregation5.feature',
        scenario='[1] `collect()` filtering nulls',
        cypher='MATCH (n)\n      OPTIONAL MATCH (n)-[:NOT_EXIST]->(x)\n      RETURN n, collect(x)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'n': '()', 'collect(x)': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation5-2',
        feature_path='tck/features/expressions/aggregation/Aggregation5.feature',
        scenario='[2] OPTIONAL MATCH and `collect()` on node property',
        cypher='OPTIONAL MATCH (f:DoesExist)\n      OPTIONAL MATCH (n:DoesNotExist)\n      RETURN collect(DISTINCT n.num) AS a, collect(DISTINCT f.num) AS b',
        graph=graph_fixture_from_create(
            """
            CREATE (:DoesExist {num: 42})
                  CREATE (:DoesExist {num: 43})
                  CREATE (:DoesExist {num: 44})
            """
        ),
        expected=Expected(
            rows=[
            {'a': '[]', 'b': '[42, 43, 44]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
