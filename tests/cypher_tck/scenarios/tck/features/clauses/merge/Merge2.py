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
        key='merge2-1',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[1] Merge node with label add label on create',
        cypher='MERGE (a:TheLabel)\n        ON CREATE SET a:Foo\n      RETURN labels(a)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'labels(a)': "['TheLabel', 'Foo']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge2-2',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[2] ON CREATE on created nodes',
        cypher='MERGE (b)\n        ON CREATE SET b.created = 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge2-3',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[3] Merge node with label add property on create',
        cypher='MERGE (a:TheLabel)\n        ON CREATE SET a.num = 42\n      RETURN a.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a.num': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge2-4',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[4] Merge node with label add property on update when it exists',
        cypher='MERGE (a:TheLabel)\n        ON CREATE SET a.num = 42\n      RETURN a.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel)
            """
        ),
        expected=Expected(
            rows=[
            {'a.num': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge2-5',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[5] Merge should be able to use properties of bound node in ON CREATE',
        cypher='MATCH (person:Person)\n      MERGE (city:City)\n        ON CREATE SET city.name = person.bornIn\n      RETURN person.bornIn',
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

    Scenario(
        key='merge2-6',
        feature_path='tck/features/clauses/merge/Merge2.feature',
        scenario='[6] Fail when using undefined variable in ON CREATE',
        cypher='MERGE (n)\n        ON CREATE SET x.num = 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),
]
