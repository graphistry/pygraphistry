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
        key='merge9-1',
        feature_path='tck/features/clauses/merge/Merge9.feature',
        scenario='[1] UNWIND with one MERGE',
        cypher='UNWIND [1, 2, 3, 4] AS int\n      MERGE (n {id: int})\n      RETURN count(*)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'count(*)': 4}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge9-2',
        feature_path='tck/features/clauses/merge/Merge9.feature',
        scenario='[2] UNWIND with multiple MERGE',
        cypher="UNWIND ['Keanu Reeves', 'Hugo Weaving', 'Carrie-Anne Moss', 'Laurence Fishburne'] AS actor\n      MERGE (m:Movie {name: 'The Matrix'})\n      MERGE (p:Person {name: actor})\n      MERGE (p)-[:ACTED_IN]->(m)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge9-3',
        feature_path='tck/features/clauses/merge/Merge9.feature',
        scenario='[3] Mixing MERGE with CREATE',
        cypher='CREATE (a:A), (b:B)\n      MERGE (a)-[:KNOWS]->(b)\n      CREATE (b)-[:KNOWS]->(c:C)\n      RETURN count(*)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'count(*)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge9-4',
        feature_path='tck/features/clauses/merge/Merge9.feature',
        scenario='[4] MERGE after WITH with predicate and WITH with aggregation',
        cypher='UNWIND [42] AS props\n      WITH props WHERE props > 32\n      WITH DISTINCT props AS p\n      MERGE (a:A {num: p})\n      RETURN a.num AS prop',
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 42})
            """
        ),
        expected=Expected(
            rows=[
            {'prop': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),
]
