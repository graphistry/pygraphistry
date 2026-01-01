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
        key='create5-1',
        feature_path='tck/features/clauses/create/Create5.feature',
        scenario='[1] Create a pattern with multiple hops',
        cypher='CREATE (:A)-[:R]->(:B)-[:R]->(:C)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create5-2',
        feature_path='tck/features/clauses/create/Create5.feature',
        scenario='[2] Create a pattern with multiple hops in the reverse direction',
        cypher='CREATE (:A)<-[:R]-(:B)<-[:R]-(:C)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create5-3',
        feature_path='tck/features/clauses/create/Create5.feature',
        scenario='[3] Create a pattern with multiple hops in varying directions',
        cypher='CREATE (:A)-[:R]->(:B)<-[:R]-(:C)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create5-4',
        feature_path='tck/features/clauses/create/Create5.feature',
        scenario='[4] Create a pattern with multiple hops with multiple types and varying directions',
        cypher='CREATE ()-[:R1]->()<-[:R2]-()-[:R3]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),

    Scenario(
        key='create5-5',
        feature_path='tck/features/clauses/create/Create5.feature',
        scenario='[5] Create a pattern with multiple hops and varying directions',
        cypher='CREATE (:A)<-[:R1]-(:B)-[:R2]->(:C)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='CREATE clause semantics are not supported',
        tags=('create', 'xfail'),
    ),
]
