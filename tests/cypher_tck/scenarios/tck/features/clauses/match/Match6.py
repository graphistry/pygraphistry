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
        key="match6-1",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[1] Zero-length named path",
        cypher="MATCH p = (a)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "xfail"),
    ),

    Scenario(
        key="match6-2",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[2] Return a simple path",
        cypher="MATCH p = (a {name: 'A'})-->(b)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A {name: 'A'})-[:KNOWS]->(:B {name: 'B'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "xfail"),
    ),

    Scenario(
        key="match6-3",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[3] Return a three node path",
        cypher="MATCH p = (a {name: 'A'})-[rel1]->(b)-[rel2]->(c)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})-[:KNOWS]->(c:C {name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A {name: 'A'})-[:KNOWS]->(:B {name: 'B'})-[:KNOWS]->(:C {name: 'C'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "xfail"),
    ),

    Scenario(
        key="match6-4",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[4] Respecting direction when matching non-existent path",
        cypher="MATCH p = ({name: 'a'})<--({name: 'b'})\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'a'}), (b {name: 'b'})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "direction", "xfail"),
    ),

    Scenario(
        key="match6-5",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[5] Path query should return results in written order",
        cypher="MATCH p = (a:Label1)<--(:Label2)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (:Label1)<-[:TYPE]-(:Label2)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:Label1)<-[:TYPE]-(:Label2)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "direction", "xfail"),
    ),

    Scenario(
        key="match6-6",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[6] Handling direction of named paths",
        cypher="MATCH p = (b)<--(a)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:T]->(b:B)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:B)<-[:T]-(:A)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "direction", "xfail"),
    ),

    Scenario(
        key="match6-7",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[7] Respecting direction when matching existing path",
        cypher="MATCH p = ({name: 'a'})-->({name: 'b'})\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'a'}), (b {name: 'b'})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<({name: 'a'})-[:T]->({name: 'b'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "direction", "xfail"),
    ),

    Scenario(
        key="match6-8",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[8] Respecting direction when matching non-existent path with multiple directions",
        cypher="MATCH p = (n)-->(k)<--(n)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b)
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "direction", "xfail"),
    ),

    Scenario(
        key="match6-9",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[9] Longer path query should return results in written order",
        cypher="MATCH p = (a:Label1)<--(:Label2)--()\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (:Label1)<-[:T1]-(:Label2)-[:T2]->(:Label3)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:Label1)<-[:T1]-(:Label2)-[:T2]->(:Label3)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "xfail"),
    ),

    Scenario(
        key="match6-10",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[10] Named path with alternating directed/undirected relationships",
        cypher="MATCH p = (n)-->(m)--(o)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (b)-[:T]->(a),
                   (c)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:C)-[:T]->(:B)-[:T]->(:A)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "undirected", "xfail"),
    ),

    Scenario(
        key="match6-11",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[11] Named path with multiple alternating directed/undirected relationships",
        cypher="MATCH path = (n)-->(m)--(o)--(p)\nRETURN path",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C), (d:D)
            CREATE (b)-[:T]->(a),
                   (c)-[:T]->(b),
                   (d)-[:T]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"path": "<(:D)-[:T]->(:C)-[:T]->(:B)-[:T]->(:A)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "undirected", "xfail"),
    ),

    Scenario(
        key="match6-12",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[12] Matching path with multiple bidirectional relationships",
        cypher="MATCH p=(n)<-->(k)<-->(n)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T1]->(b),
                   (b)-[:T2]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A)<-[:T2]-(:B)<-[:T1]-(:A)>"},
                {"p": "<(:A)-[:T1]->(:B)-[:T2]->(:A)>"},
                {"p": "<(:B)<-[:T1]-(:A)<-[:T2]-(:B)>"},
                {"p": "<(:B)-[:T2]->(:A)-[:T1]->(:B)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "undirected", "xfail"),
    ),

    Scenario(
        key="match6-13",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[13] Matching path with both directions should respect other directions",
        cypher="MATCH p = (n)<-->(k)<--(n)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T1]->(b),
                   (b)-[:T2]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A)<-[:T2]-(:B)<-[:T1]-(:A)>"},
                {"p": "<(:B)<-[:T1]-(:A)<-[:T2]-(:B)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns are not supported in the harness",
        tags=("match", "path", "undirected", "xfail"),
    ),

    Scenario(
        key="match6-14",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[14] Named path with undirected fixed variable length pattern",
        cypher="MATCH topRoute = (:Start)<-[:CONNECTED_TO]-()-[:CONNECTED_TO*3..3]-(:End)\nRETURN topRoute",
        graph=graph_fixture_from_create(
            """
            CREATE (db1:Start), (db2:End), (mid), (other)
            CREATE (mid)-[:CONNECTED_TO]->(db1),
                   (mid)-[:CONNECTED_TO]->(db2),
                   (mid)-[:CONNECTED_TO]->(db2),
                   (mid)-[:CONNECTED_TO]->(other),
                   (mid)-[:CONNECTED_TO]->(other)
            """
        ),
        expected=Expected(
            rows=[
                {"topRoute": "<(:Start)<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->()<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->(:End)>"},
                {"topRoute": "<(:Start)<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->()<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->(:End)>"},
                {"topRoute": "<(:Start)<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->()<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->(:End)>"},
                {"topRoute": "<(:Start)<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->()<-[:CONNECTED_TO]-()-[:CONNECTED_TO]->(:End)>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-15",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[15] Variable-length named path",
        cypher="MATCH p = ()-[*0..]->()\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-16",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[16] Return a var length path",
        cypher="MATCH p = (n {name: 'A'})-[:KNOWS*1..2]->(x)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS {num: 1}]->(b:B {name: 'B'})-[:KNOWS {num: 2}]->(c:C {name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A {name: 'A'})-[:KNOWS {num: 1}]->(:B {name: 'B'})>"},
                {"p": "<(:A {name: 'A'})-[:KNOWS {num: 1}]->(:B {name: 'B'})-[:KNOWS {num: 2}]->(:C {name: 'C'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-17",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[17] Return a named var length path of length zero",
        cypher="MATCH p = (a {name: 'A'})-[:KNOWS*0..1]->(b)-[:FRIEND*0..1]->(c)\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})-[:FRIEND]->(c:C {name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:A {name: 'A'})>"},
                {"p": "<(:A {name: 'A'})-[:KNOWS]->(:B {name: 'B'})>"},
                {"p": "<(:A {name: 'A'})-[:KNOWS]->(:B {name: 'B'})-[:FRIEND]->(:C {name: 'C'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-18",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[18] Undirected named path",
        cypher="MATCH p = (n:Movie)--(m)\nRETURN p\n  LIMIT 1",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Movie), (b)
            CREATE (b)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<(:Movie)<-[:T]-()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Named path returns and LIMIT handling are not supported",
        tags=("match", "path", "limit", "xfail"),
    ),

    Scenario(
        key="match6-19",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[19] Variable length relationship without lower bound",
        cypher="MATCH p = ({name: 'A'})-[:KNOWS*..2]->()\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (c {name: 'C'})
            CREATE (a)-[:KNOWS]->(b),
                   (b)-[:KNOWS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<({name: 'A'})-[:KNOWS]->({name: 'B'})>"},
                {"p": "<({name: 'A'})-[:KNOWS]->({name: 'B'})-[:KNOWS]->({name: 'C'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-20",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[20] Variable length relationship without bounds",
        cypher="MATCH p = ({name: 'A'})-[:KNOWS*..]->()\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (c {name: 'C'})
            CREATE (a)-[:KNOWS]->(b),
                   (b)-[:KNOWS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<({name: 'A'})-[:KNOWS]->({name: 'B'})>"},
                {"p": "<({name: 'A'})-[:KNOWS]->({name: 'B'})-[:KNOWS]->({name: 'C'})>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and named path returns are not supported",
        tags=("match", "path", "variable-length", "xfail"),
    ),

    Scenario(
        key="match6-21",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[21] Fail when a node has the same variable in a preceding MATCH",
        cypher="MATCH <pattern>\nMATCH p = ()-[]-()\nRETURN p",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for variable reuse in MATCH is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),

    Scenario(
        key="match6-22",
        feature_path="tck/features/clauses/match/Match6.feature",
        scenario="[22] Fail when a relationship has the same variable in a preceding MATCH",
        cypher="MATCH <pattern>\nMATCH p = ()-[]-()\nRETURN p",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for variable reuse in MATCH is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),
]
