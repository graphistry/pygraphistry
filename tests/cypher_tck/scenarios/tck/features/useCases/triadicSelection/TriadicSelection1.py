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
        key='usecase-triadicselection1-1',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[1] Handling triadic friend of a friend',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"},
            {'c.name': "'b3'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-2',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[2] Handling triadic friend of a friend that is not a friend',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-3',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[3] Handling triadic friend of a friend that is not a friend with different relationship type',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:FOLLOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-4',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[4] Handling triadic friend of a friend that is not a friend with superset of relationship type',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-5',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[5] Handling triadic friend of a friend that is not a friend with implicit subset of relationship type',
        cypher='MATCH (a:A)-->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'b4'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"},
            {'c.name': "'c31'"},
            {'c.name': "'c32'"},
            {'c.name': "'c41'"},
            {'c.name': "'c42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-6',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[6] Handling triadic friend of a friend that is not a friend with explicit subset of relationship type',
        cypher='MATCH (a:A)-[:KNOWS|FOLLOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'b4'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"},
            {'c.name': "'c31'"},
            {'c.name': "'c32'"},
            {'c.name': "'c41'"},
            {'c.name': "'c42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-7',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[7] Handling triadic friend of a friend that is not a friend with same labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c:X)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'c11'"},
            {'c.name': "'c21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-8',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[8] Handling triadic friend of a friend that is not a friend with different labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c:Y)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'c12'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-9',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[9] Handling triadic friend of a friend that is not a friend with implicit subset of labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c:X)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'c11'"},
            {'c.name': "'c21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-10',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[10] Handling triadic friend of a friend that is not a friend with implicit superset of labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"},
            {'c.name': "'c11'"},
            {'c.name': "'c12'"},
            {'c.name': "'c21'"},
            {'c.name': "'c22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-11',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[11] Handling triadic friend of a friend that is a friend',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-12',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[12] Handling triadic friend of a friend that is a friend with different relationship type',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:FOLLOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b3'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-13',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[13] Handling triadic friend of a friend that is a friend with superset of relationship type',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"},
            {'c.name': "'b3'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-14',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[14] Handling triadic friend of a friend that is a friend with implicit subset of relationship type',
        cypher='MATCH (a:A)-->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b1'"},
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-15',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[15] Handling triadic friend of a friend that is a friend with explicit subset of relationship type',
        cypher='MATCH (a:A)-[:KNOWS|FOLLOWS]->(b)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_1_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b1'"},
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-16',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[16] Handling triadic friend of a friend that is a friend with same labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c:X)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-17',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[17] Handling triadic friend of a friend that is a friend with different labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c:Y)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[

            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-18',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[18] Handling triadic friend of a friend that is a friend with implicit subset of labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b)-->(c:X)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-triadicselection1-19',
        feature_path='tck/features/useCases/triadicSelection/TriadicSelection1.feature',
        scenario='[19] Handling triadic friend of a friend that is a friend with implicit superset of labels',
        cypher='MATCH (a:A)-[:KNOWS]->(b:X)-->(c)\n      OPTIONAL MATCH (a)-[r:KNOWS]->(c)\n      WITH c WHERE r IS NOT NULL\n      RETURN c.name',
        graph=BINARY_TREE_2_GRAPH,
        expected=Expected(
            rows=[
            {'c.name': "'b2'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'triadicSelection', 'meta-xfail', 'xfail'),
    ),
]
