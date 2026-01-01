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
        key='merge3-1',
        feature_path='tck/features/clauses/merge/Merge3.feature',
        scenario='[1] Merge should be able to set labels on match',
        cypher='MERGE (a)\n        ON MATCH SET a:L',
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge3-2',
        feature_path='tck/features/clauses/merge/Merge3.feature',
        scenario='[2] Merge node with label add label on match when it exists',
        cypher='MERGE (a:TheLabel)\n        ON MATCH SET a:Foo\n      RETURN labels(a)',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel)
            """
        ),
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
        key='merge3-3',
        feature_path='tck/features/clauses/merge/Merge3.feature',
        scenario='[3] Merge node and set property on match',
        cypher='MERGE (a:TheLabel)\n        ON MATCH SET a.num = 42\n      RETURN a.num',
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel)
            """
        ),
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
        key='merge3-4',
        feature_path='tck/features/clauses/merge/Merge3.feature',
        scenario='[4] Merge should be able to use properties of bound node in ON MATCH',
        cypher='MATCH (person:Person)\n      MERGE (city:City)\n        ON MATCH SET city.name = person.bornIn\n      RETURN person.bornIn',
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
        key='merge3-5',
        feature_path='tck/features/clauses/merge/Merge3.feature',
        scenario='[5] Fail when using undefined variable in ON MATCH',
        cypher='MERGE (n)\n        ON MATCH SET x.num = 1',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),
]
