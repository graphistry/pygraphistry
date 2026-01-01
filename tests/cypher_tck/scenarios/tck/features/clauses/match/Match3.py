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
        key="match3-1",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[1] Get neighbours",
        cypher="MATCH (n1)-[rel:KNOWS]->(n2)\nRETURN n1, n2",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {num: 1})-[:KNOWS]->(b:B {num: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"n1": "(:A {num: 1})", "n2": "(:B {num: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with multiple return columns are not supported",
        tags=("match", "relationship", "return", "xfail"),
    ),

    Scenario(
        key="match3-2",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[2] Directed match of a simple relationship",
        cypher="MATCH (a)-[r]->(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:LOOP]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:LOOP]", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with relationships are not supported",
        tags=("match", "relationship", "return", "xfail"),
    ),

    Scenario(
        key="match3-3",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[3] Undirected match on simple relationship graph",
        cypher="MATCH (a)-[r]-(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:LOOP]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:LOOP]", "b": "(:B)"},
                {"a": "(:B)", "r": "[:LOOP]", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with undirected relationship matches are not supported",
        tags=("match", "relationship", "undirected", "xfail"),
    ),

    Scenario(
        key="match3-5",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[5] Return two subgraphs with bound undirected relationship",
        cypher="MATCH (a)-[r {name: 'r'}]-(b)\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {num: 1})-[:REL {name: 'r'}]->(b:B {num: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {num: 2})", "b": "(:A {num: 1})"},
                {"a": "(:A {num: 1})", "b": "(:B {num: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with undirected relationship matches are not supported",
        tags=("match", "relationship", "undirected", "xfail"),
    ),

    Scenario(
        key="match3-7",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[7] Matching nodes with many labels",
        cypher="MATCH (n:A:B:C:D:E:F:G:H:I:J:K:L:M)-[:T]->(m:Z:Y:X:W:V:U)\nRETURN n, m",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A:B:C:D:E:F:G:H:I:J:K:L:M),
                   (b:U:V:W:X:Y:Z)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A:B:C:D:E:F:G:H:I:J:K:L:M)", "m": "(:Z:Y:X:W:V:U)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Label predicates on relationship endpoints and row projections are not supported",
        tags=("match", "relationship", "label", "xfail"),
    ),

    Scenario(
        key="match3-4",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[4] Get two related nodes",
        cypher="MATCH ()-[rel:KNOWS]->(x)\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {num: 1}),
              (a)-[:KNOWS]->(b:B {num: 2}),
              (a)-[:KNOWS]->(c:C {num: 3})
            """
        ),
        expected=Expected(
            node_ids=["b", "c"],
            rows=[
                {"x": "(:B {num: 2})"},
                {"x": "(:C {num: 3})"},
            ],
        ),
        gfql=[n(), e_forward({"type": "KNOWS"}), n(name="x")],
        return_alias="x",
        tags=("match", "relationship", "return"),
    ),

    Scenario(
        key="match3-6",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[6] Matching a relationship pattern using a label predicate",
        cypher="MATCH (a)-->(b:Foo)\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b1:Foo), (b2)
            CREATE (a)-[:T]->(b1),
                   (a)-[:T]->(b2)
            """
        ),
        expected=Expected(
            node_ids=["b1"],
            rows=[
                {"b": "(:Foo)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Label predicates on relationship endpoints are not supported in the harness",
        tags=("match", "relationship", "label", "xfail"),
    ),

    Scenario(
        key="match3-8",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[8] Matching using relationship predicate with multiples of the same type",
        cypher="MATCH (a)-[:T|:T]->(b)\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            node_ids=["b"],
            rows=[
                {"b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Multiple relationship types are not supported",
        tags=("match", "relationship", "multi-type", "xfail"),
    ),

    Scenario(
        key="match3-9",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[9] Get related to related to",
        cypher="MATCH (n)-->(a)-->(b)\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {num: 1})-[:KNOWS]->(b:B {num: 2})-[:FRIEND]->(c:C {num: 3})
            """
        ),
        expected=Expected(
            node_ids=["c"],
            rows=[
                {"b": "(:C {num: 3})"},
            ],
        ),
        gfql=[n(), e_forward(), n(), e_forward(), n(name="b")],
        return_alias="b",
        tags=("match", "relationship", "multi-hop", "return"),
    ),

    Scenario(
        key="match3-10",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[10] Matching using self-referencing pattern returns no result",
        cypher="MATCH (a)-->(b), (b)-->(b)\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (c)
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            node_ids=[],
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable reuse and row projections are not supported",
        tags=("match", "pattern-join", "self-reference", "xfail"),
    ),

    Scenario(
        key="match3-11",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[11] Undirected match in self-relationship graph",
        cypher="MATCH (a)-[r]-(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:LOOP]", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with undirected relationship matches are not supported",
        tags=("match", "relationship", "undirected", "xfail"),
    ),

    Scenario(
        key="match3-12",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[12] Undirected match of self-relationship in self-relationship graph",
        cypher="MATCH (n)-[r]-(n)\nRETURN n, r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A)", "r": "[:LOOP]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Repeated node variables in patterns and row projections are not supported",
        tags=("match", "relationship", "undirected", "self-reference", "xfail"),
    ),

    Scenario(
        key="match3-13",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[13] Directed match on self-relationship graph",
        cypher="MATCH (a)-[r]->(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:LOOP]", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Row projections with relationship matches are not supported",
        tags=("match", "relationship", "return", "xfail"),
    ),

    Scenario(
        key="match3-14",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[14] Directed match of self-relationship on self-relationship graph",
        cypher="MATCH (n)-[r]->(n)\nRETURN n, r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:LOOP]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A)", "r": "[:LOOP]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Repeated node variables in patterns and row projections are not supported",
        tags=("match", "relationship", "self-reference", "xfail"),
    ),

    Scenario(
        key="match3-15",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[15] Mixing directed and undirected pattern parts with self-relationship, simple",
        cypher="MATCH (x:A)-[r1]->(y)-[r2]-(z)\nRETURN x, r1, y, r2, z",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(l:Looper),
                   (l)-[:LOOP]->(l),
                   (l)-[:T2]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:A)", "r1": "[:T1]", "y": "(:Looper)", "r2": "[:LOOP]", "z": "(:Looper)"},
                {"x": "(:A)", "r1": "[:T1]", "y": "(:Looper)", "r2": "[:T2]", "z": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Mixed directed/undirected patterns and row projections are not supported",
        tags=("match", "relationship", "mixed-direction", "xfail"),
    ),

    Scenario(
        key="match3-16",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[16] Mixing directed and undirected pattern parts with self-relationship, undirected",
        cypher="MATCH (x)-[r1]-(y)-[r2]-(z)\nRETURN x, r1, y, r2, z",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(l:Looper),
                   (l)-[:LOOP]->(l),
                   (l)-[:T2]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:A)", "r1": "[:T1]", "y": "(:Looper)", "r2": "[:LOOP]", "z": "(:Looper)"},
                {"x": "(:A)", "r1": "[:T1]", "y": "(:Looper)", "r2": "[:T2]", "z": "(:B)"},
                {"x": "(:Looper)", "r1": "[:LOOP]", "y": "(:Looper)", "r2": "[:T1]", "z": "(:A)"},
                {"x": "(:Looper)", "r1": "[:LOOP]", "y": "(:Looper)", "r2": "[:T2]", "z": "(:B)"},
                {"x": "(:B)", "r1": "[:T2]", "y": "(:Looper)", "r2": "[:LOOP]", "z": "(:Looper)"},
                {"x": "(:B)", "r1": "[:T2]", "y": "(:Looper)", "r2": "[:T1]", "z": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Undirected multi-hop patterns and row projections are not supported",
        tags=("match", "relationship", "undirected", "xfail"),
    ),

    Scenario(
        key="match3-17",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[17] Handling cyclic patterns",
        cypher="MATCH (a)-[:A]->()-[:B]->(a)\nRETURN a.name",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})
            CREATE (a)-[:A]->(b),
                   (b)-[:B]->(a),
                   (b)-[:B]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"a.name": "'a'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable reuse and property projections are not supported",
        tags=("match", "pattern-join", "cycle", "xfail"),
    ),

    Scenario(
        key="match3-18",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[18] Handling cyclic patterns when separated into two parts",
        cypher="MATCH (a)-[:A]->(b), (b)-[:B]->(a)\nRETURN a.name",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})
            CREATE (a)-[:A]->(b),
                   (b)-[:B]->(a),
                   (b)-[:B]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"a.name": "'a'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable reuse across comma-separated patterns is not supported",
        tags=("match", "pattern-join", "cycle", "xfail"),
    ),

    Scenario(
        key="match3-19",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[19] Two bound nodes pointing to the same node",
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'})\nMATCH (a)-->(x)<-->(b)\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (x1 {name: 'x1'}), (x2 {name: 'x2'})
            CREATE (a)-[:KNOWS]->(x1),
                   (a)-[:KNOWS]->(x2),
                   (b)-[:KNOWS]->(x1),
                   (b)-[:KNOWS]->(x2)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "({name: 'x1'})"},
                {"x": "({name: 'x2'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable binding across multiple MATCH clauses is not supported",
        tags=("match", "pattern-join", "multi-match", "xfail"),
    ),

    Scenario(
        key="match3-20",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[20] Three bound nodes pointing to the same node",
        cypher="MATCH (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})\nMATCH (a)-->(x), (b)-->(x), (c)-->(x)\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'}),
                   (x1 {name: 'x1'}), (x2 {name: 'x2'})
            CREATE (a)-[:KNOWS]->(x1),
                   (a)-[:KNOWS]->(x2),
                   (b)-[:KNOWS]->(x1),
                   (b)-[:KNOWS]->(x2),
                   (c)-[:KNOWS]->(x1),
                   (c)-[:KNOWS]->(x2)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "({name: 'x1'})"},
                {"x": "({name: 'x2'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable binding across multiple MATCH clauses is not supported",
        tags=("match", "pattern-join", "multi-match", "xfail"),
    ),

    Scenario(
        key="match3-21",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[21] Three bound nodes pointing to the same node with extra connections",
        cypher="MATCH (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'})\nMATCH (a)-->(x), (b)-->(x), (c)-->(x)\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'a'}), (b {name: 'b'}), (c {name: 'c'}),
                   (d {name: 'd'}), (e {name: 'e'}), (f {name: 'f'}),
                   (g {name: 'g'}), (h {name: 'h'}), (i {name: 'i'}),
                   (j {name: 'j'}), (k {name: 'k'})
            CREATE (a)-[:KNOWS]->(d),
                   (a)-[:KNOWS]->(e),
                   (a)-[:KNOWS]->(f),
                   (a)-[:KNOWS]->(g),
                   (a)-[:KNOWS]->(i),
                   (b)-[:KNOWS]->(d),
                   (b)-[:KNOWS]->(e),
                   (b)-[:KNOWS]->(f),
                   (b)-[:KNOWS]->(h),
                   (b)-[:KNOWS]->(k),
                   (c)-[:KNOWS]->(d),
                   (c)-[:KNOWS]->(e),
                   (c)-[:KNOWS]->(h),
                   (c)-[:KNOWS]->(g),
                   (c)-[:KNOWS]->(j)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "({name: 'd'})"},
                {"x": "({name: 'e'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern-level variable binding across multiple MATCH clauses is not supported",
        tags=("match", "pattern-join", "multi-match", "xfail"),
    ),

    Scenario(
        key="match3-22",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[22] Returning bound nodes that are not part of the pattern",
        cypher="MATCH (a {name: 'A'}), (c {name: 'C'})\nMATCH (a)-->(b)\nRETURN a, b, c",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (c {name: 'C'})
            CREATE (a)-[:KNOWS]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'A'})", "b": "({name: 'B'})", "c": "({name: 'C'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Multiple MATCH clause bindings and row projections are not supported",
        tags=("match", "multi-match", "return", "xfail"),
    ),

    Scenario(
        key="match3-23",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[23] Matching disconnected patterns",
        cypher="MATCH (a)-->(b)\nMATCH (c)-->(d)\nRETURN a, b, c, d",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:T]->(b),
                   (a)-[:T]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)", "c": "(:A)", "d": "(:B)"},
                {"a": "(:A)", "b": "(:B)", "c": "(:A)", "d": "(:C)"},
                {"a": "(:A)", "b": "(:C)", "c": "(:A)", "d": "(:B)"},
                {"a": "(:A)", "b": "(:C)", "c": "(:A)", "d": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian products across multiple MATCH clauses and row projections are not supported",
        tags=("match", "cartesian", "return", "xfail"),
    ),

    Scenario(
        key="match3-24",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[24] Matching twice with duplicate relationship types on same relationship",
        cypher="MATCH (a1)-[r:T]->()\nWITH r, a1\nMATCH (a1)-[r:T]->(b2)\nRETURN a1, r, b2",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a1": "(:A)", "r": "[:T]", "b2": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping and row projections are not supported",
        tags=("match", "with", "pipeline", "xfail"),
    ),

    Scenario(
        key="match3-25",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[25] Matching twice with an additional node label",
        cypher="MATCH (a1)-[r]->()\nWITH r, a1\nMATCH (a1:X)-[r]->(b2)\nRETURN a1, r, b2",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping, label predicates, and row projections are not supported",
        tags=("match", "with", "label", "xfail"),
    ),

    Scenario(
        key="match3-26",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[26] Matching twice with a duplicate predicate",
        cypher="MATCH (a1:X:Y)-[r]->()\nWITH r, a1\nMATCH (a1:Y)-[r]->(b2)\nRETURN a1, r, b2",
        graph=graph_fixture_from_create(
            """
            CREATE (:X:Y)-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a1": "(:X:Y)", "r": "[:T]", "b2": "()"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping, label predicates, and row projections are not supported",
        tags=("match", "with", "label", "xfail"),
    ),

    Scenario(
        key="match3-27",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[27] Matching from null nodes should return no results owing to finding no matches",
        cypher="OPTIONAL MATCH (a)\nWITH a\nMATCH (a)-->(b)\nRETURN b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, WITH pipelines, and null handling are not supported",
        tags=("match", "optional-match", "with", "null", "xfail"),
    ),

    Scenario(
        key="match3-28",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[28] Matching from null nodes should return no results owing to matches being filtered out",
        cypher="OPTIONAL MATCH (a:TheLabel)\nWITH a\nMATCH (a)-->(b)\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, WITH pipelines, and null handling are not supported",
        tags=("match", "optional-match", "with", "null", "xfail"),
    ),

    Scenario(
        key="match3-29",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[29] Fail when re-using a relationship in the same pattern",
        cypher="MATCH (a)-[r]->()-[r]->(a)\nRETURN r",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for relationship uniqueness is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),

    Scenario(
        key="match3-30",
        feature_path="tck/features/clauses/match/Match3.feature",
        scenario="[30] Fail when using a list or nodes as a node",
        cypher="MATCH (n)\nWITH [n] AS users\nMATCH (users)-->(messages)\nRETURN messages",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for variable type conflicts is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),
]
