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
        key='expr-graph5-1',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[1] Single-labels expression on nodes',
        cypher='MATCH (a)\n      RETURN a, a:B AS result',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:B:C)', 'result': 'true'},
            {'a': '(:A:B)', 'result': 'true'},
            {'a': '(:A:C)', 'result': 'false'},
            {'a': '(:B:C)', 'result': 'true'},
            {'a': '(:A)', 'result': 'false'},
            {'a': '(:B)', 'result': 'true'},
            {'a': '(:C)', 'result': 'false'},
            {'a': '()', 'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-2',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[2] Single-labels expression on relationships',
        cypher='MATCH ()-[r]->()\n      RETURN r, r:T2 AS result',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(),
                         ()-[:T2]->(),
                         ()-[:t2]->(),
                         (:T2)-[:T3]->(),
                         ()-[:T4]->(:T2)
            """
        ),
        expected=Expected(
            rows=[
            {'r': '[:T1]', 'result': 'false'},
            {'r': '[:T2]', 'result': 'true'},
            {'r': '[:t2]', 'result': 'false'},
            {'r': '[:T3]', 'result': 'false'},
            {'r': '[:T4]', 'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-3',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[3] Conjunctive labels expression on nodes',
        cypher='MATCH (a)\n      RETURN a, a:A:B AS result',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:B:C)', 'result': 'true'},
            {'a': '(:A:B)', 'result': 'true'},
            {'a': '(:A:C)', 'result': 'false'},
            {'a': '(:B:C)', 'result': 'false'},
            {'a': '(:A)', 'result': 'false'},
            {'a': '(:B)', 'result': 'false'},
            {'a': '(:C)', 'result': 'false'},
            {'a': '()', 'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-4-1',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[4] Conjunctive labels expression on nodes with varying order and repeating labels (example 1)',
        cypher='MATCH (a)\n      WHERE a:A:C\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-4-2',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[4] Conjunctive labels expression on nodes with varying order and repeating labels (example 2)',
        cypher='MATCH (a)\n      WHERE a:C:A\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-4-3',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[4] Conjunctive labels expression on nodes with varying order and repeating labels (example 3)',
        cypher='MATCH (a)\n      WHERE a:A:C:A\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-4-4',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[4] Conjunctive labels expression on nodes with varying order and repeating labels (example 4)',
        cypher='MATCH (a)\n      WHERE a:C:C:A\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-4-5',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[4] Conjunctive labels expression on nodes with varying order and repeating labels (example 5)',
        cypher='MATCH (a)\n      WHERE a:C:A:A:C\n      RETURN a',
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B), (:A:C), (:B:C),
                         (:A), (:B), (:C), ()
            """
        ),
        expected=Expected(
            rows=[
            {'a': '(:A:C)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph5-5',
        feature_path='tck/features/expressions/graph/Graph5.feature',
        scenario='[5] Label expression on null',
        cypher='MATCH (n:Single)\n      OPTIONAL MATCH (n)-[r:TYPE]-(m)\n      RETURN m:TYPE',
        graph=graph_fixture_from_create(
            """
            CREATE (s:Single)
            """
        ),
        expected=Expected(
            rows=[
            {'m:TYPE': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),
]
