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
        key='expr-graph8-1',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[1] Using `keys()` on a single node, non-empty result',
        cypher='MATCH (n)\n      UNWIND keys(n) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'Andres', surname: 'Lopez'})
            """
        ),
        expected=Expected(
            rows=[
            {'theProps': "'name'"},
            {'theProps': "'surname'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-2',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[2] Using `keys()` on multiple nodes, non-empty result',
        cypher='MATCH (n)\n      UNWIND keys(n) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'Andres', surname: 'Lopez'}),
                         ({otherName: 'Andres', otherSurname: 'Lopez'})
            """
        ),
        expected=Expected(
            rows=[
            {'theProps': "'name'"},
            {'theProps': "'surname'"},
            {'theProps': "'otherName'"},
            {'theProps': "'otherSurname'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-3',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[3] Using `keys()` on a single node, empty result',
        cypher='MATCH (n)\n      UNWIND keys(n) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-4',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[4] Using `keys()` on an optionally matched node',
        cypher='OPTIONAL MATCH (n)\n      UNWIND keys(n) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-5',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[5] Using `keys()` on a relationship, non-empty result',
        cypher='MATCH ()-[r:KNOWS]-()\n      UNWIND keys(r) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:KNOWS {status: 'bad', year: '2015'}]->()
            """
        ),
        expected=Expected(
            rows=[
            {'theProps': "'status'"},
            {'theProps': "'year'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-6',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[6] Using `keys()` on a relationship, empty result',
        cypher='MATCH ()-[r:KNOWS]-()\n      UNWIND keys(r) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:KNOWS]->()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-7',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[7] Using `keys()` on an optionally matched relationship',
        cypher='OPTIONAL MATCH ()-[r:KNOWS]-()\n      UNWIND keys(r) AS x\n      RETURN DISTINCT x AS theProps',
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:KNOWS]->()
            """
        ),
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-graph8-8',
        feature_path='tck/features/expressions/graph/Graph8.feature',
        scenario='[8] Using `keys()` and `IN` to check property existence',
        cypher="MATCH (n)\n      RETURN 'exists' IN keys(n) AS a,\n             'missing' IN keys(n) AS b,\n             'missingToo' IN keys(n) AS c",
        graph=graph_fixture_from_create(
            """
            CREATE ({exists: 42, missing: null})
            """
        ),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'false', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'graph', 'meta-xfail', 'xfail'),
    ),
]
