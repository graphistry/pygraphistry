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
        key='usecase-countingsubgraphmatches1-1',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[1] Undirected match in self-relationship graph, count',
        cypher='MATCH ()--()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-2',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[2] Undirected match of self-relationship in self-relationship graph, count',
        cypher='MATCH (n)--(n)\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-3',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[3] Undirected match on simple relationship graph, count',
        cypher='MATCH ()--()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:LOOP]->(:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-4',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[4] Directed match on self-relationship graph, count',
        cypher='MATCH ()-->()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-5',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[5] Directed match of self-relationship on self-relationship graph, count',
        cypher='MATCH (n)-->(n)\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-6',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[6] Counting undirected self-relationships in self-relationship graph',
        cypher='MATCH (n)-[r]-(n)\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-7',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[7] Counting distinct undirected self-relationships in self-relationship graph',
        cypher='MATCH (n)-[r]-(n)\n      RETURN count(DISTINCT r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
            {'count(DISTINCT r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-8',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[8] Directed match of a simple relationship, count',
        cypher='MATCH ()-->()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:LOOP]->(:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-9',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[9] Counting directed self-relationships',
        cypher='MATCH (n)-[r]->(n)\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a),
                         ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-10',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[10] Mixing directed and undirected pattern parts with self-relationship, count',
        cypher='MATCH (:A)-->()--()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(l:Looper),
                         (l)-[:LOOP]->(l),
                         (l)-[:T2]->(:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='usecase-countingsubgraphmatches1-11',
        feature_path='tck/features/useCases/countingSubgraphMatches/CountingSubgraphMatches1.feature',
        scenario='[11] Mixing directed and undirected pattern parts with self-relationship, undirected count',
        cypher='MATCH ()-[]-()-[]-()\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(l:Looper),
                         (l)-[:LOOP]->(l),
                         (l)-[:T2]->(:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(*)': 6}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='UseCases suite is not supported',
        tags=('usecase', 'countingSubgraphMatches', 'meta-xfail', 'xfail'),
    ),
]
