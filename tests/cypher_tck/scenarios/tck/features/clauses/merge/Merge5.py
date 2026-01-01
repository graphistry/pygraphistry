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
        key='merge5-1',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[1] Creating a relationship',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE]->(b)\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
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
        key='merge5-2',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[2] Matching a relationship',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE]->(b)\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:TYPE]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-3',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[3] Matching two relationships',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE]->(b)\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:TYPE]->(b)
                  CREATE (a)-[:TYPE]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-4',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[4] Using bound variables from other updating clause',
        cypher='CREATE (a), (b)\n      MERGE (a)-[:X]->(b)\n      RETURN count(a)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'count(a)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-5',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[5] Filtering relationships',
        cypher="MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE {name: 'r2'}]->(b)\n      RETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:TYPE {name: 'r1'}]->(b)
                  CREATE (a)-[:TYPE {name: 'r2'}]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-6',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[6] Creating relationship when all matches filtered out',
        cypher="MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE {name: 'r2'}]->(b)\n      RETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:TYPE {name: 'r1'}]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-7',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[7] Matching incoming relationship',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)<-[r:TYPE]-(b)\n      RETURN count(r)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (b)-[:TYPE]->(a)
                  CREATE (a)-[:TYPE]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-8',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[8] Creating relationship with property',
        cypher="MATCH (a:A), (b:B)\n      MERGE (a)-[r:TYPE {name: 'Lola'}]->(b)\n      RETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            """
        ),
        expected=Expected(
            rows=[
            {'count(r)': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-9',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[9] Creating relationship using merged nodes',
        cypher='MERGE (a:A)\n      MERGE (b:B)\n      MERGE (a)-[:FOO]->(b)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-10',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[10] Merge should bind a path',
        cypher='MERGE (a {num: 1})\n      MERGE (b {num: 2})\n      MERGE p = (a)-[:R]->(b)\n      RETURN p',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'p': '<({num: 1})-[:R]->({num: 2})>'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-11',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[11] Use outgoing direction when unspecified',
        cypher='CREATE (a {id: 2}), (b {id: 1})\n      MERGE (a)-[r:KNOWS]-(b)\n      RETURN startNode(r).id AS s, endNode(r).id AS e',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'s': 2, 'e': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-12',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[12] Match outgoing relationship when direction unspecified',
        cypher='MATCH (a {id: 2}), (b {id: 1})\n      MERGE (a)-[r:KNOWS]-(b)\n      RETURN r',
        graph=graph_fixture_from_create(
            """
            CREATE (a {id: 1}), (b {id: 2})
                  CREATE (a)-[:KNOWS]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'r': '[:KNOWS]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-13',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[13] Match both incoming and outgoing relationships when direction unspecified',
        cypher='MATCH (a {id: 2})--(b {id: 1})\n      MERGE (a)-[r:KNOWS]-(b)\n      RETURN r',
        graph=graph_fixture_from_create(
            """
            CREATE (a {id: 2}), (b {id: 1}), (c {id: 1}), (d {id: 2})
                  CREATE (a)-[:KNOWS {name: 'ab'}]->(b)
                  CREATE (c)-[:KNOWS {name: 'cd'}]->(d)
            """
        ),
        expected=Expected(
            rows=[
            {'r': "[:KNOWS {name: 'ab'}]"},
            {'r': "[:KNOWS {name: 'cd'}]"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-14',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[14] Using list properties via variable',
        cypher="CREATE (a:Foo), (b:Bar)\n      WITH a, b\n      UNWIND ['a,b', 'a,b'] AS str\n      WITH a, b, split(str, ',') AS roles\n      MERGE (a)-[r:FB {foobar: roles}]->(b)\n      RETURN count(*)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'count(*)': 2}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-15',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[15] Matching using list property',
        cypher='MATCH (a:A), (b:B)\n      MERGE (a)-[r:T {numbers: [42, 43]}]->(b)\n      RETURN count(*)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:T {numbers: [42, 43]}]->(b)
            """
        ),
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
        key='merge5-16',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[16] Aliasing of existing nodes 1',
        cypher='MATCH (n)\n      MATCH (m)\n      WITH n AS a, m AS b\n      MERGE (a)-[r:T]->(b)\n      RETURN a.id AS a, b.id AS b',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[
            {'a': 0, 'b': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-17',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[17] Aliasing of existing nodes 2',
        cypher='MATCH (n)\n      WITH n AS a, n AS b\n      MERGE (a)-[r:T]->(b)\n      RETURN a.id AS a',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[
            {'a': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-18',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[18] Double aliasing of existing nodes 1',
        cypher='MATCH (n)\n      MATCH (m)\n      WITH n AS a, m AS b\n      MERGE (a)-[:T]->(b)\n      WITH a AS x, b AS y\n      MERGE (a)\n      MERGE (b)\n      MERGE (a)-[:T]->(b)\n      RETURN x.id AS x, y.id AS y',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[
            {'x': 0, 'y': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-19',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[19] Double aliasing of existing nodes 2',
        cypher='MATCH (n)\n      WITH n AS a\n      MERGE (c)\n      MERGE (a)-[:T]->(c)\n      WITH a AS x\n      MERGE (c)\n      MERGE (x)-[:T]->(c)\n      RETURN x.id AS x',
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 0})
            """
        ),
        expected=Expected(
            rows=[
            {'x': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-20',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[20] Do not match on deleted entities',
        cypher='MATCH (a:A)-[ab]->(b:B)-[bc]->(c:C)\n      DELETE ab, bc, b, c\n      MERGE (newB:B {num: 1})\n      MERGE (a)-[:REL]->(newB)\n      MERGE (newC:C)\n      MERGE (newB)-[:REL]->(newC)',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)
                  CREATE (b1:B {num: 0}), (b2:B {num: 1})
                  CREATE (c1:C), (c2:C)
                  CREATE (a)-[:REL]->(b1),
                         (a)-[:REL]->(b2),
                         (b1)-[:REL]->(c1),
                         (b2)-[:REL]->(c2)
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-21',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[21] Do not match on deleted relationships',
        cypher="MATCH (a)-[t:T]->(b)\n      DELETE t\n      MERGE (a)-[t2:T {name: 'rel3'}]->(b)\n      RETURN t2.name",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
                  CREATE (a)-[:T {name: 'rel1'}]->(b),
                         (a)-[:T {name: 'rel2'}]->(b)
            """
        ),
        expected=Expected(
            rows=[
            {'t2.name': "'rel3'"},
            {'t2.name': "'rel3'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='MERGE clause semantics are not supported',
        tags=('merge', 'xfail'),
    ),

    Scenario(
        key='merge5-22',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[22] Fail when imposing new predicates on a variable that is already bound',
        cypher='CREATE (a:Foo)\n      MERGE (a)-[r:KNOWS]->(a:Bar)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-23',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[23] Fail when merging relationship without type',
        cypher='CREATE (a), (b)\n      MERGE (a)-->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-24',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[24] Fail when merging relationship without type, no colon',
        cypher='MATCH (a), (b)\n      MERGE (a)-[NO_COLON]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-25',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[25] Fail when merging relationship with more than one type',
        cypher='CREATE (a), (b)\n      MERGE (a)-[:A|:B]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-26',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[26] Fail when merging relationship that is already bound',
        cypher='MATCH (a)-[r]->(b)\n      MERGE (a)-[r]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-27',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[27] Fail when using parameter as relationship predicate in MERGE',
        cypher='MERGE (a)\n      MERGE (b)\n      MERGE (a)-[r:FOO $param]->(b)\n      RETURN r',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-28',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[28] Fail when using variable length relationship in MERGE',
        cypher='MERGE (a)\n      MERGE (b)\n      MERGE (a)-[:FOO*2]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('merge', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='merge5-29',
        feature_path='tck/features/clauses/merge/Merge5.feature',
        scenario='[29] Fail on merging relationship with null property',
        cypher='CREATE (a), (b)\n      MERGE (a)-[r:X {num: null}]->(b)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('merge', 'runtime-error', 'xfail'),
    ),
]
