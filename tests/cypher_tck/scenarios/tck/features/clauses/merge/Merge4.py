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
        key='merge4-1',
        feature_path='tck/features/clauses/merge/Merge4.feature',
        scenario='[1] Merge should be able to set labels on match and on create',
        cypher='MATCH ()\n      MERGE (a:L)\n        ON MATCH SET a:M1\n        ON CREATE SET a:M2',
        graph=graph_fixture_from_create(
            """
            CREATE (), ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge4-2',
        feature_path='tck/features/clauses/merge/Merge4.feature',
        scenario='[2] Merge should be able to use properties of bound node in ON MATCH and ON CREATE',
        cypher='MATCH (person:Person)\n        MERGE (city:City)\n          ON MATCH SET city.name = person.bornIn\n          ON CREATE SET city.name = person.bornIn\n        RETURN person.bornIn',
        graph=graph_fixture_from_create(
            """
            CREATE (:Person {bornIn: 'New York'}),
                    (:Person {bornIn: 'Ohio'})
            """
        ),
        expected=Expected(
            rows=[
            {'person.bornIn': "'New York'"},
            {'person.bornIn': "'Ohio'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),
]
