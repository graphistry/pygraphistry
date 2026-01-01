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
        key='expr-graph6-1',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[1] Statically access a property of a non-null node',
        cypher='MATCH (n)\n      RETURN n.missing, n.missingToo, n.existing',
        graph=graph_fixture_from_create(
            """
            CREATE ({existing: 42, missing: null})
            """
        ),
        expected=Expected(
            rows=[
            {'n.missing': 'null', 'n.missingToo': 'null', 'n.existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-2',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[2] Statically access a property of a optional non-null node',
        cypher='OPTIONAL MATCH (n)\n      RETURN n.missing, n.missingToo, n.existing',
        graph=graph_fixture_from_create(
            """
            CREATE ({existing: 42, missing: null})
            """
        ),
        expected=Expected(
            rows=[
            {'n.missing': 'null', 'n.missingToo': 'null', 'n.existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-3',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[3] Statically access a property of a null node',
        cypher='OPTIONAL MATCH (n)\n      RETURN n.missing',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'n.missing': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-4',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[4] Statically access a property of a node resulting from an expression',
        cypher='MATCH (n)\n      WITH [123, n] AS list\n      RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing',
        graph=graph_fixture_from_create(
            """
            CREATE ({existing: 42, missing: null})
            """
        ),
        expected=Expected(
            rows=[
            {'(list[1]).missing': 'null', '(list[1]).missingToo': 'null', '(list[1]).existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-5',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[5] Statically access a property of a non-null relationship',
        cypher='MATCH ()-[r]->()\n      RETURN r.missing, r.missingToo, r.existing',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:REL {existing: 42, missing: null}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'r.missing': 'null', 'r.missingToo': 'null', 'r.existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-6',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[6] Statically access a property of a optional non-null relationship',
        cypher='OPTIONAL MATCH ()-[r]->()\n      RETURN r.missing, r.missingToo, r.existing',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:REL {existing: 42, missing: null}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'r.missing': 'null', 'r.missingToo': 'null', 'r.existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-7',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[7] Statically access a property of a null relationship',
        cypher='OPTIONAL MATCH ()-[r]->()\n      RETURN r.missing',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'r.missing': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-8',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[8] Statically access a property of a relationship resulting from an expression',
        cypher='MATCH ()-[r]->()\n      WITH [123, r] AS list\n      RETURN (list[1]).missing, (list[1]).missingToo, (list[1]).existing',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:REL {existing: 42, missing: null}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'(list[1]).missing': 'null', '(list[1]).missingToo': 'null', '(list[1]).existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-1',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 1)',
        cypher='WITH 123 AS nonGraphElement\n      RETURN nonGraphElement.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-2',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 2)',
        cypher='WITH 42.45 AS nonGraphElement\n      RETURN nonGraphElement.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-3',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 3)',
        cypher='WITH true AS nonGraphElement\n      RETURN nonGraphElement.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-4',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 4)',
        cypher='WITH false AS nonGraphElement\n      RETURN nonGraphElement.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-5',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 5)',
        cypher="WITH 'string' AS nonGraphElement\n      RETURN nonGraphElement.num",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph6-9-6',
        feature_path='tck/features/expressions/graph/Graph6.feature',
        scenario='[9] Fail when performing property access on a non-graph element (example 6)',
        cypher='WITH [123, true] AS nonGraphElement\n      RETURN nonGraphElement.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),
]
