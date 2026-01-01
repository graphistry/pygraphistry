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
        key='expr-graph4-1',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[1] `type()`',
        cypher='MATCH ()-[r]->()\n      RETURN type(r)',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'type(r)': "'T'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-2',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[2] `type()` on two relationships',
        cypher='MATCH ()-[r1]->()-[r2]->()\n      RETURN type(r1), type(r2)',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->()-[:T2]->()
            """
        ),
        expected=Expected(
            rows=[
            {'type(r1)': "'T1'", 'type(r2)': "'T2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-3',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[3] `type()` on null relationship',
        cypher='MATCH (a)\n      OPTIONAL MATCH (a)-[r:NOT_THERE]->()\n      RETURN type(r), type(null)',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
            {'type(r)': 'null', 'type(null)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-4',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[4] `type()` on mixed null and non-null relationships',
        cypher='MATCH (a)\n      OPTIONAL MATCH (a)-[r:T]->()\n      RETURN type(r)',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'type(r)': "'T'"},
            {'type(r)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-5',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[5] `type()` handling Any type',
        cypher='MATCH (a)-[r]->()\n      WITH [r, 1] AS list\n      RETURN type(list[0])',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'type(list[0])': "'T'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-6-1',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[6] `type()` failing on invalid arguments (example 1)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [r, 0] | type(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-6-2',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[6] `type()` failing on invalid arguments (example 2)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [r, 1.0] | type(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-6-3',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[6] `type()` failing on invalid arguments (example 3)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [r, true] | type(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-6-4',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[6] `type()` failing on invalid arguments (example 4)',
        cypher="MATCH p = (n)-[r:T]->()\n      RETURN [x IN [r, ''] | type(x) ] AS list",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-6-5',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[6] `type()` failing on invalid arguments (example 5)',
        cypher='MATCH p = (n)-[r:T]->()\n      RETURN [x IN [r, []] | type(x) ] AS list',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'graph', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-graph4-7',
        feature_path='tck/features/expressions/graph/Graph4.feature',
        scenario='[7] Failing when using `type()` on a node',
        cypher='MATCH (r)\n      RETURN type(r)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
