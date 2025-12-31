from graphistry.compute import e_forward, e_undirected, n

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.parse_cypher import graph_fixture_from_create


MATCH5_GRAPH = graph_fixture_from_create(
    """
    CREATE (n0:A {name: 'n0'}),
           (n00:B {name: 'n00'}),
           (n01:B {name: 'n01'}),
           (n000:C {name: 'n000'}),
           (n001:C {name: 'n001'}),
           (n010:C {name: 'n010'}),
           (n011:C {name: 'n011'}),
           (n0000:D {name: 'n0000'}),
           (n0001:D {name: 'n0001'}),
           (n0010:D {name: 'n0010'}),
           (n0011:D {name: 'n0011'}),
           (n0100:D {name: 'n0100'}),
           (n0101:D {name: 'n0101'}),
           (n0110:D {name: 'n0110'}),
           (n0111:D {name: 'n0111'})
    CREATE (n0)-[:LIKES]->(n00),
           (n0)-[:LIKES]->(n01),
           (n00)-[:LIKES]->(n000),
           (n00)-[:LIKES]->(n001),
           (n01)-[:LIKES]->(n010),
           (n01)-[:LIKES]->(n011),
           (n000)-[:LIKES]->(n0000),
           (n000)-[:LIKES]->(n0001),
           (n001)-[:LIKES]->(n0010),
           (n001)-[:LIKES]->(n0011),
           (n010)-[:LIKES]->(n0100),
           (n010)-[:LIKES]->(n0101),
           (n011)-[:LIKES]->(n0110),
           (n011)-[:LIKES]->(n0111)
    """
)

MATCH7_GRAPH_SINGLE = graph_fixture_from_create(
    """
    CREATE (s:Single), (a:A {num: 42}),
           (b:B {num: 46}), (c:C)
    CREATE (s)-[:REL]->(a),
           (s)-[:REL]->(b),
           (a)-[:REL]->(c),
           (b)-[:LOOP]->(b)
    """
)

MATCH7_GRAPH_AB = graph_fixture_from_create(
    """
    CREATE (:A)-[:T]->(:B)
    """
)

MATCH7_GRAPH_ABC = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
    CREATE (a)-[:KNOWS]->(b),
           (b)-[:KNOWS]->(c)
    """
)

MATCH7_GRAPH_REL = graph_fixture_from_create(
    """
    CREATE (a:A {num: 1})-[:REL {name: 'r1'}]->(b:B {num: 2})-[:REL {name: 'r2'}]->(c:C {num: 3})
    """
)

MATCH7_GRAPH_X = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
    CREATE (a)-[:X]->(b)
    """
)

MATCH7_GRAPH_AB_X = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'})
    CREATE (a)-[:X]->(b)
    """
)

MATCH7_GRAPH_LABELS = graph_fixture_from_create(
    """
    CREATE (:X), (x:X), (y1:Y), (y2:Y:Z)
    CREATE (x)-[:REL]->(y1),
           (x)-[:REL]->(y2)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_BOTH = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team)
    CREATE (a)-[:PLAYS_FOR]->(b),
           (a)-[:SUPPORTS]->(b)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_SINGLE = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team)
    CREATE (a)-[:PLAYS_FOR]->(b)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_DIFF = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team), (c:Team)
    CREATE (a)-[:PLAYS_FOR]->(b),
           (a)-[:SUPPORTS]->(c)
    """
)

WITH_ORDERBY4_GRAPH = graph_fixture_from_create(
    """
    CREATE (:A {num: 1, num2: 4}),
           (:A {num: 5, num2: 2}),
           (:A {num: 9, num2: 0}),
           (:A {num: 3, num2: 3}),
           (:A {num: 7, num2: 1})
    """
)


SCENARIOS = [
    Scenario(
        key="match1-1",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[1] Match non-existent nodes returns empty",
        cypher="MATCH (n)\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(node_ids=[]),
        gfql=[n()],
        tags=("match", "return", "empty-graph"),
    ),
    Scenario(
        key="match1-2",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[2] Matching all nodes",
        cypher="MATCH (n)\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B {name: 'b'}), ({name: 'c'})
            """
        ),
        expected=Expected(
            node_ids=["anon_1", "anon_2", "anon_3"],
            rows=[
                {"n": "(:A)"},
                {"n": "(:B {name: 'b'})"},
                {"n": "({name: 'c'})"},
            ],
        ),
        gfql=[n()],
        tags=("match", "return", "parser-graph"),
    ),
    Scenario(
        key="match1-3",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[3] Matching nodes using multiple labels",
        cypher="MATCH (a:A:B)\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A:B:C), (:A:B), (:A:C), (:B:C),
                   (:A), (:B), (:C),
                   ({name: ':A:B:C'}), ({abc: 'abc'}), ()
            """
        ),
        expected=Expected(node_ids=["anon_1", "anon_2"]),
        gfql=[n({"label__A": True, "label__B": True})],
        tags=("match", "labels", "parser-graph"),
    ),
    Scenario(
        key="match1-4",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[4] Simple node inline property predicate",
        cypher="MATCH (n {name: 'bar'})\nRETURN n",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "name": "bar"},
                {"id": "n2", "name": "monkey"},
                {"id": "n3", "firstname": "bar"},
            ],
            edges=[],
        ),
        expected=Expected(
            node_ids=["n1"],
            rows=[{"n": "({name: 'bar'})"}],
        ),
        gfql=[n({"name": "bar"})],
        tags=("match", "property", "inline-predicate"),
    ),
    Scenario(
        key="match1-5",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[5] Use multiple MATCH clauses to do a Cartesian product",
        cypher="MATCH (n), (m)\nRETURN n.num AS n, m.num AS m",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1}), ({num: 2}), ({num: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"n": 1, "m": 1},
                {"n": 1, "m": 2},
                {"n": 1, "m": 3},
                {"n": 2, "m": 1},
                {"n": 2, "m": 2},
                {"n": 2, "m": 3},
                {"n": 3, "m": 3},
                {"n": 3, "m": 1},
                {"n": 3, "m": 2},
            ]
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian product + projection results not supported in current GFQL harness",
        tags=("match", "cartesian", "return", "xfail"),
    ),
    Scenario(
        key="match-where1-3",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[3] Filter node with property predicate on a single variable with multiple bindings",
        cypher="MATCH (n)\nWHERE n.name = 'Bar'\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (), ({name: 'Bar'}), (:Bar)
            """
        ),
        expected=Expected(node_ids=["anon_2"]),
        gfql=[n({"name": "Bar"})],
        tags=("match-where", "property"),
    ),
    Scenario(
        key="match-where1-4",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[4] Filter start node of relationship with property predicate on multi variables with multiple bindings",
        cypher="MATCH (n:Person)-->()\nWHERE n.name = 'Bob'\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Person {name: 'Alice'}), (b:Person {name: 'Bob'}),
                   (c), (d)
            CREATE (a)-[:T]->(c),
                   (b)-[:T]->(d)
            """
        ),
        expected=Expected(node_ids=["b"]),
        gfql=[n({"label__Person": True, "name": "Bob"}, name="n"), e_forward(), n()],
        return_alias="n",
        tags=("match-where", "relationship", "alias"),
    ),
    Scenario(
        key="match-where1-5",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[5] Filter end node of relationship with property predicate on multi variables with multiple bindings",
        cypher="MATCH ()-[rel:X]-(a)\nWHERE a.name = 'Andres'\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'Someone'})<-[:X]-()-[:X]->({name: 'Andres'})
            """
        ),
        expected=Expected(node_ids=["anon_3"]),
        gfql=[n(), e_undirected({"type": "X"}), n({"name": "Andres"}, name="a")],
        return_alias="a",
        tags=("match-where", "relationship", "type"),
    ),
    Scenario(
        key="match-where1-6",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[6] Filter node with a parameter in a property predicate on multi variables with one binding",
        cypher="MATCH (a)-[r]->(b)\nWHERE b.name = $param\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T {name: 'bar'}]->(:B {name: 'me'})
            """
        ),
        expected=Expected(
            rows=[{"r": "[:T {name: 'bar'}]"}],
        ),
        gfql=None,
        status="xfail",
        reason="Parameter binding and edge-return validation not supported in harness",
        tags=("match-where", "params", "edge-return", "xfail"),
    ),
    Scenario(
        key="match-where1-7",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[7] Filter relationship with relationship type predicate on multi variables with multiple bindings",
        cypher="MATCH (n {name: 'A'})-[r]->(x)\nWHERE type(r) = 'KNOWS'\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'}),
              (b:B {name: 'B'}),
              (c:C {name: 'C'}),
              (a)-[:KNOWS]->(b),
              (a)-[:HATES]->(c)
            """
        ),
        expected=Expected(node_ids=["b"]),
        gfql=[n({"name": "A"}, name="n"), e_forward({"type": "KNOWS"}), n(name="x")],
        return_alias="x",
        tags=("match-where", "relationship", "type"),
    ),
    Scenario(
        key="match-where1-8",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[8] Filter relationship with property predicate on multi variables with multiple bindings",
        cypher="MATCH (node)-[r:KNOWS]->(a)\nWHERE r.name = 'monkey'\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)<-[:KNOWS {name: 'monkey'}]-()-[:KNOWS {name: 'woot'}]->(:B)
            """
        ),
        expected=Expected(node_ids=["anon_1"]),
        gfql=[n(), e_forward({"type": "KNOWS", "name": "monkey"}, name="r"), n(name="a")],
        return_alias="a",
        tags=("match-where", "relationship", "property"),
    ),
    Scenario(
        key="match-where1-9",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[9] Filter relationship with a parameter in a property predicate on multi variables with one binding",
        cypher="MATCH (a)-[r]->(b)\nWHERE r.name = $param\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T {name: 'bar'}]->(:B {name: 'me'})
            """
        ),
        expected=Expected(
            node_ids=["anon_2"],
            rows=[{"b": "(:B {name: 'me'})"}],
        ),
        gfql=None,
        status="xfail",
        reason="Parameter binding and edge-return validation not supported in harness",
        tags=("match-where", "params", "edge-return", "xfail"),
    ),
    Scenario(
        key="match-where1-10",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[10] Filter node with disjunctive property predicate on single variables with multiple bindings",
        cypher="MATCH (n)\nWHERE n.p1 = 12 OR n.p2 = 13\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {p1: 12}),
              (b:B {p2: 13}),
              (c:C)
            """
        ),
        expected=Expected(node_ids=["a", "b"]),
        gfql=None,
        status="xfail",
        reason="Disjunctive WHERE predicates are not supported in harness",
        tags=("match-where", "or", "xfail"),
    ),
    Scenario(
        key="match-where1-11",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[11] Filter relationship with disjunctive relationship type predicate on multi variables with multiple bindings",
        cypher="MATCH (n)-[r]->(x)\nWHERE type(r) = 'KNOWS' OR type(r) = 'HATES'\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}),
              (b {name: 'B'}),
              (c {name: 'C'}),
              (a)-[:KNOWS]->(b),
              (a)-[:HATES]->(c),
              (a)-[:WONDERS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[:KNOWS]"},
                {"r": "[:HATES]"},
            ]
        ),
        gfql=None,
        status="xfail",
        reason="Disjunctive WHERE predicates and edge-return validation are not supported",
        tags=("match-where", "or", "edge-return", "xfail"),
    ),
    Scenario(
        key="match-where1-12",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[12] Filter path with path length predicate on multi variables with one binding",
        cypher="MATCH p = (n)-->(x)\nWHERE length(p) = 1\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})
            """
        ),
        expected=Expected(node_ids=["b"]),
        gfql=None,
        status="xfail",
        reason="Path variables and length() predicates are not supported in harness",
        tags=("match-where", "path-length", "xfail"),
    ),
    Scenario(
        key="match-where1-13",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[13] Filter path with false path length predicate on multi variables with one binding",
        cypher="MATCH p = (n)-->(x)\nWHERE length(p) = 10\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})
            """
        ),
        expected=Expected(node_ids=[]),
        gfql=None,
        status="xfail",
        reason="Path variables and length() predicates are not supported in harness",
        tags=("match-where", "path-length", "xfail"),
    ),
    Scenario(
        key="match-where1-14",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[14] Fail when filtering path with property predicate",
        cypher="MATCH (n)\nMATCH r = (n)-[*]->()\nWHERE r.name = 'apa'\nRETURN r",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Syntax error validation for invalid path property predicates not enforced",
        tags=("match-where", "syntax-error", "xfail"),
    ),
    Scenario(
        key="match-where1-15",
        feature_path="tck/features/clauses/match-where/MatchWhere1.feature",
        scenario="[15] Fail on aggregation in WHERE",
        cypher="MATCH (a)\nWHERE count(a) > 10\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Syntax error validation for aggregations in WHERE not enforced",
        tags=("match-where", "syntax-error", "xfail"),
    ),
    Scenario(
        key="match-where2-1",
        feature_path="tck/features/clauses/match-where/MatchWhere2.feature",
        scenario="[1] Filter nodes with conjunctive two-part property predicate on multi variables with multiple bindings",
        cypher="MATCH (a)--(b)--(c)--(d)--(a), (b)--(d)\nWHERE a.id = 1\n  AND c.id = 2\nRETURN d",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (c:C {id: 2}), (d:D)
            CREATE (a)-[:T]->(b),
                   (a)-[:T]->(c),
                   (a)-[:T]->(d),
                   (b)-[:T]->(c),
                   (b)-[:T]->(d),
                   (c)-[:T]->(d)
            """
        ),
        expected=Expected(
            node_ids=["a", "d"],
            rows=[
                {"d": "(:A)"},
                {"d": "(:D)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Multi-variable WHERE predicates and complex pattern matching not supported",
        return_alias="d",
        tags=("match-where", "and", "multi-var", "xfail"),
    ),
    Scenario(
        key="match-where2-2",
        feature_path="tck/features/clauses/match-where/MatchWhere2.feature",
        scenario="[2] Filter node with conjunctive multi-part property predicates on multi variables with multiple bindings",
        cypher="MATCH (advertiser)-[:ADV_HAS_PRODUCT]->(out)-[:AP_HAS_VALUE]->(red)<-[:AA_HAS_VALUE]-(a)\nWHERE advertiser.id = $1\n  AND a.id = $2\n  AND red.name = 'red'\n  AND out.name = 'product1'\nRETURN out.name",
        graph=graph_fixture_from_create(
            """
            CREATE (advertiser {name: 'advertiser1', id: 0}),
                   (thing {name: 'Color', id: 1}),
                   (red {name: 'red'}),
                   (p1 {name: 'product1'}),
                   (p2 {name: 'product4'})
            CREATE (advertiser)-[:ADV_HAS_PRODUCT]->(p1),
                   (advertiser)-[:ADV_HAS_PRODUCT]->(p2),
                   (thing)-[:AA_HAS_VALUE]->(red),
                   (p1)-[:AP_HAS_VALUE]->(red),
                   (p2)-[:AP_HAS_VALUE]->(red)
            """
        ),
        expected=Expected(
            rows=[{"out.name": "'product1'"}],
        ),
        gfql=None,
        status="xfail",
        reason="Parameter binding, multi-variable WHERE, and projection validation not supported",
        tags=("match-where", "params", "and", "multi-var", "xfail"),
    ),
    Scenario(
        key="match-where3-1",
        feature_path="tck/features/clauses/match-where/MatchWhere3.feature",
        scenario="[1] Join between node identities",
        cypher="MATCH (a), (b)\nWHERE a = b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:A)"},
                {"a": "(:B)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian products, variable equality joins, and row projections are not supported",
        tags=("match-where", "join", "identity", "xfail"),
    ),
    Scenario(
        key="match-where3-2",
        feature_path="tck/features/clauses/match-where/MatchWhere3.feature",
        scenario="[2] Join between node properties of disconnected nodes",
        cypher="MATCH (a:A), (b:B)\nWHERE a.id = b.id\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {id: 1}),
                   (:A {id: 2}),
                   (:B {id: 2}),
                   (:B {id: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {id: 2})", "b": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian products, cross-variable equality predicates, and row projections are not supported",
        tags=("match-where", "join", "property", "xfail"),
    ),
    Scenario(
        key="match-where3-3",
        feature_path="tck/features/clauses/match-where/MatchWhere3.feature",
        scenario="[3] Join between node properties of adjacent nodes",
        cypher="MATCH (n)-[rel]->(x)\nWHERE n.animal = x.animal\nRETURN n, x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {animal: 'monkey'}),
              (b:B {animal: 'cow'}),
              (c:C {animal: 'monkey'}),
              (d:D {animal: 'cow'}),
              (a)-[:KNOWS]->(b),
              (a)-[:KNOWS]->(c),
              (d)-[:KNOWS]->(b),
              (d)-[:KNOWS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A {animal: 'monkey'})", "x": "(:C {animal: 'monkey'})"},
                {"n": "(:D {animal: 'cow'})", "x": "(:B {animal: 'cow'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cross-variable equality predicates and row projections are not supported",
        tags=("match-where", "join", "property", "xfail"),
    ),
    Scenario(
        key="match-where4-1",
        feature_path="tck/features/clauses/match-where/MatchWhere4.feature",
        scenario="[1] Join nodes on inequality",
        cypher="MATCH (a), (b)\nWHERE a <> b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)"},
                {"a": "(:B)", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian products, variable inequality joins, and row projections are not supported",
        tags=("match-where", "join", "inequality", "xfail"),
    ),
    Scenario(
        key="match-where4-2",
        feature_path="tck/features/clauses/match-where/MatchWhere4.feature",
        scenario="[2] Join with disjunctive multi-part predicates including patterns",
        cypher="MATCH (a), (b)\nWHERE a.id = 0\n  AND (a)-[:T]->(b:TheLabel)\n  OR (a)-[:T*]->(b:MissingLabel)\nRETURN DISTINCT b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:TheLabel {id: 0}), (b:TheLabel {id: 1}), (c:TheLabel {id: 2})
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            node_ids=["b"],
            rows=[
                {"b": "(:TheLabel {id: 1})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern predicates, variable-length relationships, OR, DISTINCT, and row projections not supported",
        tags=("match-where", "or", "pattern-predicate", "variable-length", "distinct", "xfail"),
    ),
    Scenario(
        key="match-where5-1",
        feature_path="tck/features/clauses/match-where/MatchWhere5.feature",
        scenario="[1] Filter out on null",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWHERE i.var > 'te'\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            node_ids=["child1"],
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Comparison predicates and null semantics are not supported in the harness",
        tags=("match-where", "null", "comparison", "xfail"),
    ),
    Scenario(
        key="match-where5-2",
        feature_path="tck/features/clauses/match-where/MatchWhere5.feature",
        scenario="[2] Filter out on null if the AND'd predicate evaluates to false",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWHERE i.var > 'te' AND i:TextNode\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            node_ids=["child1"],
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Comparison predicates, label predicates in WHERE, and null semantics are not supported",
        tags=("match-where", "null", "comparison", "label-predicate", "xfail"),
    ),
    Scenario(
        key="match-where5-3",
        feature_path="tck/features/clauses/match-where/MatchWhere5.feature",
        scenario="[3] Filter out on null if the AND'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWHERE i.var > 'te' AND i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            node_ids=["child1"],
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Comparison predicates, IS NOT NULL, and null semantics are not supported",
        tags=("match-where", "null", "comparison", "is-not-null", "xfail"),
    ),
    Scenario(
        key="match-where5-4",
        feature_path="tck/features/clauses/match-where/MatchWhere5.feature",
        scenario="[4] Do not filter out on null if the OR'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i)\nWHERE i.var > 'te' OR i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            node_ids=["child1", "child2"],
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
                {"i": "(:IntNode {var: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OR predicates, comparison predicates, IS NOT NULL, and null semantics are not supported",
        tags=("match-where", "null", "or", "comparison", "is-not-null", "xfail"),
    ),
    Scenario(
        key="match-where6-1",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[1] Filter node with node label predicate on multi variables with multiple bindings after MATCH and OPTIONAL MATCH",
        cypher="MATCH (a)-->(b)\nWHERE b:B\nOPTIONAL MATCH (a)-->(c)\nWHERE c:C\nRETURN a.name",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b:B {name: 'B'}), (c:C {name: 'C'}), (d:D {name: 'C'})
            CREATE (a)-[:T]->(b),
                   (a)-[:T]->(c),
                   (a)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"a.name": "'A'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, label predicates in WHERE, and projection comparisons are not supported",
        tags=("match-where", "optional-match", "label-predicate", "xfail"),
    ),
    Scenario(
        key="match-where6-2",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[2] Filter node with false node label predicate after OPTIONAL MATCH",
        cypher="MATCH (n:Single)\nOPTIONAL MATCH (n)-[r]-(m)\nWHERE m:NonExistent\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Single), (a:A {num: 42}),
                   (b:B {num: 46}), (c:C)
            CREATE (s)-[:REL]->(a),
                   (s)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (b)-[:LOOP]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, label predicates in WHERE, and null row comparisons are not supported",
        tags=("match-where", "optional-match", "label-predicate", "null", "xfail"),
    ),
    Scenario(
        key="match-where6-3",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[3] Filter node with property predicate on multi variables with multiple bindings after OPTIONAL MATCH",
        cypher="MATCH (n:Single)\nOPTIONAL MATCH (n)-[r]-(m)\nWHERE m.num = 42\nRETURN m",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Single), (a:A {num: 42}),
                   (b:B {num: 46}), (c:C)
            CREATE (s)-[:REL]->(a),
                   (s)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (b)-[:LOOP]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"m": "(:A {num: 42})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match-where", "optional-match", "property", "xfail"),
    ),
    Scenario(
        key="match-where6-4",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[4] Do not fail when predicates on optionally matched and missed nodes are invalid",
        cypher="MATCH (n)-->(x0)\nOPTIONAL MATCH (x0)-->(x1)\nWHERE x1.name = 'bar'\nRETURN x0.name",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b {name: 'Mark'})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"x0.name": "'Mark'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and null predicate handling are not supported",
        tags=("match-where", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="match-where6-5",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[5] Matching and optionally matching with unbound nodes and equality predicate in reverse direction",
        cypher="MATCH (a1)-[r]->()\nWITH r, a1\n  LIMIT 1\nOPTIONAL MATCH (a2)<-[r]-(b2)\nWHERE a1 = a2\nRETURN a1, r, b2, a2",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a1": "(:A)", "r": "[:T]", "b2": "null", "a2": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping, OPTIONAL MATCH semantics, and variable equality joins are not supported",
        tags=("match-where", "optional-match", "with", "limit", "join", "xfail"),
    ),
    Scenario(
        key="match-where6-6",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[6] Join nodes on non-equality of properties – OPTIONAL MATCH and WHERE",
        cypher="MATCH (x:X)\nOPTIONAL MATCH (x)-[:E1]->(y:Y)\nWHERE x.val < y.val\nRETURN x, y",
        graph=graph_fixture_from_create(
            """
            CREATE
              (:X {val: 1})-[:E1]->(:Y {val: 2})-[:E2]->(:Z {val: 3}),
              (:X {val: 4})-[:E1]->(:Y {val: 5}),
              (:X {val: 6})
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:X {val: 1})", "y": "(:Y {val: 2})"},
                {"x": "(:X {val: 4})", "y": "(:Y {val: 5})"},
                {"x": "(:X {val: 6})", "y": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable comparisons, and row projections are not supported",
        tags=("match-where", "optional-match", "comparison", "join", "xfail"),
    ),
    Scenario(
        key="match-where6-7",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[7] Join nodes on non-equality of properties – OPTIONAL MATCH on two relationships and WHERE",
        cypher="MATCH (x:X)\nOPTIONAL MATCH (x)-[:E1]->(y:Y)-[:E2]->(z:Z)\nWHERE x.val < z.val\nRETURN x, y, z",
        graph=graph_fixture_from_create(
            """
            CREATE
              (:X {val: 1})-[:E1]->(:Y {val: 2})-[:E2]->(:Z {val: 3}),
              (:X {val: 4})-[:E1]->(:Y {val: 5}),
              (:X {val: 6})
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:X {val: 1})", "y": "(:Y {val: 2})", "z": "(:Z {val: 3})"},
                {"x": "(:X {val: 4})", "y": "null", "z": "null"},
                {"x": "(:X {val: 6})", "y": "null", "z": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable comparisons, and row projections are not supported",
        tags=("match-where", "optional-match", "comparison", "join", "xfail"),
    ),
    Scenario(
        key="match-where6-8",
        feature_path="tck/features/clauses/match-where/MatchWhere6.feature",
        scenario="[8] Join nodes on non-equality of properties – Two OPTIONAL MATCH clauses and WHERE",
        cypher="MATCH (x:X)\nOPTIONAL MATCH (x)-[:E1]->(y:Y)\nOPTIONAL MATCH (y)-[:E2]->(z:Z)\nWHERE x.val < z.val\nRETURN x, y, z",
        graph=graph_fixture_from_create(
            """
            CREATE
              (:X {val: 1})-[:E1]->(:Y {val: 2})-[:E2]->(:Z {val: 3}),
              (:X {val: 4})-[:E1]->(:Y {val: 5}),
              (:X {val: 6})
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:X {val: 1})", "y": "(:Y {val: 2})", "z": "(:Z {val: 3})"},
                {"x": "(:X {val: 4})", "y": "(:Y {val: 5})", "z": "null"},
                {"x": "(:X {val: 6})", "y": "null", "z": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable comparisons, and row projections are not supported",
        tags=("match-where", "optional-match", "comparison", "join", "xfail"),
    ),
    Scenario(
        key="match2-1",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[1] Match non-existent relationships returns empty",
        cypher="MATCH ()-[r]->()\nRETURN r",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(edge_ids=[]),
        gfql=[n(), e_forward(), n()],
        tags=("match", "relationship", "empty-graph"),
    ),
    Scenario(
        key="match2-2",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[2] Matching a relationship pattern using a label predicate on both sides",
        cypher="MATCH (:A)-[r]->(:B)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(:B),
                   (:B)-[:T2]->(:A),
                   (:B)-[:T3]->(:B),
                   (:A)-[:T4]->(:A)
            """
        ),
        expected=Expected(
            edge_ids=["rel_1"],
            rows=[
                {"r": "[:T1]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Label predicates on both sides of relationship matches are not supported in the harness",
        tags=("match", "relationship", "label", "xfail"),
    ),
    Scenario(
        key="match2-3",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[3] Matching a self-loop with an undirected relationship pattern",
        cypher="MATCH ()-[r]-()\nRETURN type(r) AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a)
            CREATE (a)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="type(r) projection and row comparisons are not supported",
        tags=("match", "relationship", "type-projection", "xfail"),
    ),
    Scenario(
        key="match2-4",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[4] Matching a self-loop with a directed relationship pattern",
        cypher="MATCH ()-[r]->()\nRETURN type(r) AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a)
            CREATE (a)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="type(r) projection and row comparisons are not supported",
        tags=("match", "relationship", "type-projection", "xfail"),
    ),
    Scenario(
        key="match2-5",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[5] Match relationship with inline property value",
        cypher="MATCH (node)-[r:KNOWS {name: 'monkey'}]->(a)\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)<-[:KNOWS {name: 'monkey'}]-()-[:KNOWS {name: 'woot'}]->(:B)
            """
        ),
        expected=Expected(
            node_ids=["anon_1"],
            rows=[
                {"a": "(:A)"},
            ],
        ),
        gfql=[n(), e_forward({"type": "KNOWS", "name": "monkey"}), n(name="a")],
        return_alias="a",
        tags=("match", "relationship", "property"),
    ),
    Scenario(
        key="match2-6",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[6] Match relationships with multiple types",
        cypher="MATCH (n)-[r:KNOWS|HATES]->(x)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}),
              (b {name: 'B'}),
              (c {name: 'C'}),
              (a)-[:KNOWS]->(b),
              (a)-[:HATES]->(c),
              (a)-[:WONDERS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[:KNOWS]"},
                {"r": "[:HATES]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Multiple relationship types and row projections are not supported",
        tags=("match", "relationship", "multi-type", "xfail"),
    ),
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
    Scenario(
        key="match4-1",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[1] Handling fixed-length variable length pattern",
        cypher="MATCH (a)-[r*1..1]->(b)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:T]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-2",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[2] Simple variable length pattern",
        cypher="MATCH (a {name: 'A'})-[*]->(x)\nRETURN x",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (c {name: 'C'}), (d {name: 'D'})
            CREATE (a)-[:CONTAINS]->(b),
                   (b)-[:CONTAINS]->(c),
                   (c)-[:CONTAINS]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "({name: 'B'})"},
                {"x": "({name: 'C'})"},
                {"x": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching is not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-3",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[3] Zero-length variable length pattern in the middle of the pattern",
        cypher="MATCH (a {name: 'A'})-[:CONTAINS*0..1]->(b)-[:FRIEND*0..1]->(c)\nRETURN a, b, c",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}),
                   (c {name: 'C'}), ({name: 'D'}),
                   ({name: 'E'})
            CREATE (a)-[:CONTAINS]->(b),
                   (b)-[:FRIEND]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'A'})", "b": "({name: 'A'})", "c": "({name: 'A'})"},
                {"a": "({name: 'A'})", "b": "({name: 'B'})", "c": "({name: 'B'})"},
                {"a": "({name: 'A'})", "b": "({name: 'B'})", "c": "({name: 'C'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-4",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[4] Matching longer variable length paths",
        cypher="MATCH (n {var: 'start'})-[:T*]->(m {var: 'end'})\nRETURN m",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"m": "({var: 'end'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and UNWIND-based setup are not supported in the harness",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-5",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[5] Matching variable length pattern with property predicate",
        cypher="MATCH (a:Artist)-[:WORKED_WITH* {year: 1988}]->(b:Artist)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Artist:A), (b:Artist:B), (c:Artist:C)
            CREATE (a)-[:WORKED_WITH {year: 1987}]->(b),
                   (b)-[:WORKED_WITH {year: 1988}]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:Artist:B)", "b": "(:Artist:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-6",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[6] Matching variable length patterns from a bound node",
        cypher="MATCH (a:A)\nMATCH (a)-[r*2]->()\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b), (c)
            CREATE (a)-[:X]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:X], [:Y]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and list projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match4-7",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[7] Matching variable length patterns including a bound relationship",
        cypher="MATCH ()-[r:EDGE]-()\nMATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m)\nRETURN count(p) AS c",
        graph=graph_fixture_from_create(
            """
            CREATE (n0:Node),
                   (n1:Node),
                   (n2:Node),
                   (n3:Node),
                   (n0)-[:EDGE]->(n1),
                   (n1)-[:EDGE]->(n2),
                   (n2)-[:EDGE]->(n3)
            """
        ),
        expected=Expected(
            rows=[
                {"c": 32},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and aggregations are not supported",
        tags=("match", "variable-length", "aggregation", "xfail"),
    ),
    Scenario(
        key="match4-8",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[8] Matching relationships into a list and matching variable length using the list",
        cypher="MATCH ()-[r1]->()-[r2]->()\nWITH [r1, r2] AS rs\n  LIMIT 1\nMATCH (first)-[rs*]->(second)\nRETURN first, second",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:Y]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"first": "(:A)", "second": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list relationship variables, and variable-length matching are not supported",
        tags=("match", "variable-length", "with", "xfail"),
    ),
    Scenario(
        key="match4-9",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[9] Fail when asterisk operator is missing",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES..]->(c)\nRETURN c.name",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for invalid relationship patterns is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),
    Scenario(
        key="match4-10",
        feature_path="tck/features/clauses/match/Match4.feature",
        scenario="[10] Fail on negative bound",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*-2]->(c)\nRETURN c.name",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for invalid relationship patterns is not enforced",
        tags=("match", "syntax-error", "xfail"),
    ),
    Scenario(
        key="match5-1",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[1] Handling unbounded variable length match",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-2",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[2] Handling explicitly unbounded variable length match",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*..]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-3",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[3] Handling single bounded variable length match 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*0]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-4",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[4] Handling single bounded variable length match 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-5",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[5] Handling single bounded variable length match 3",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-6",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[6] Handling upper and lower bounded variable length match 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*0..2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0'"},
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-7",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[7] Handling upper and lower bounded variable length match 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1..2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-8",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[8] Handling symmetrically bounded variable length match, bounds are zero",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*0..0]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-9",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[9] Handling symmetrically bounded variable length match, bounds are one",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1..1]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-10",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[10] Handling symmetrically bounded variable length match, bounds are two",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*2..2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-11",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[11] Handling upper and lower bounded variable length match, empty interval 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*2..1]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-12",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[12] Handling upper and lower bounded variable length match, empty interval 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1..0]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-13",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[13] Handling upper bounded variable length match, empty interval",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*..0]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-14",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[14] Handling upper bounded variable length match 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*..1]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-15",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[15] Handling upper bounded variable length match 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*..2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-16",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[16] Handling lower bounded variable length match 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*0..]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0'"},
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-17",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[17] Handling lower bounded variable length match 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1..]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-18",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[18] Handling lower bounded variable length match 3",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*2..]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-19",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[19] Handling a variable length relationship and a standard relationship in chain, zero length 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*0]->()-[:LIKES]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-20",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[20] Handling a variable length relationship and a standard relationship in chain, zero length 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES]->()-[:LIKES*0]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00'"},
                {"c.name": "'n01'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-21",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[21] Handling a variable length relationship and a standard relationship in chain, single length 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*1]->()-[:LIKES]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-22",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[22] Handling a variable length relationship and a standard relationship in chain, single length 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES]->()-[:LIKES*1]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n000'"},
                {"c.name": "'n001'"},
                {"c.name": "'n010'"},
                {"c.name": "'n011'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-23",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[23] Handling a variable length relationship and a standard relationship in chain, longer 1",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES*2]->()-[:LIKES]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-24",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[24] Handling a variable length relationship and a standard relationship in chain, longer 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES]->()-[:LIKES*2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n0000'"},
                {"c.name": "'n0001'"},
                {"c.name": "'n0010'"},
                {"c.name": "'n0011'"},
                {"c.name": "'n0100'"},
                {"c.name": "'n0101'"},
                {"c.name": "'n0110'"},
                {"c.name": "'n0111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and row projections are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-25",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[25] Handling a variable length relationship and a standard relationship in chain, longer 3",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES]->()-[:LIKES*3]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00000'"},
                {"c.name": "'n00001'"},
                {"c.name": "'n00010'"},
                {"c.name": "'n00011'"},
                {"c.name": "'n00100'"},
                {"c.name": "'n00101'"},
                {"c.name": "'n00110'"},
                {"c.name": "'n00111'"},
                {"c.name": "'n01000'"},
                {"c.name": "'n01001'"},
                {"c.name": "'n01010'"},
                {"c.name": "'n01011'"},
                {"c.name": "'n01100'"},
                {"c.name": "'n01101'"},
                {"c.name": "'n01110'"},
                {"c.name": "'n01111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and MATCH/CREATE setup expressions are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-26",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[26] Handling mixed relationship patterns and directions 1",
        cypher="MATCH (a:A)\nMATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00000'"},
                {"c.name": "'n00001'"},
                {"c.name": "'n00010'"},
                {"c.name": "'n00011'"},
                {"c.name": "'n00100'"},
                {"c.name": "'n00101'"},
                {"c.name": "'n00110'"},
                {"c.name": "'n00111'"},
                {"c.name": "'n01000'"},
                {"c.name": "'n01001'"},
                {"c.name": "'n01010'"},
                {"c.name": "'n01011'"},
                {"c.name": "'n01100'"},
                {"c.name": "'n01101'"},
                {"c.name": "'n01110'"},
                {"c.name": "'n01111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching, direction mixing, and MATCH/DELETE/CREATE setup are not supported",
        tags=("match", "variable-length", "direction", "xfail"),
    ),
    Scenario(
        key="match5-27",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[27] Handling mixed relationship patterns and directions 2",
        cypher="MATCH (a:A)\nMATCH (a)-[:LIKES]->()<-[:LIKES*3]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00000'"},
                {"c.name": "'n00001'"},
                {"c.name": "'n00010'"},
                {"c.name": "'n00011'"},
                {"c.name": "'n00100'"},
                {"c.name": "'n00101'"},
                {"c.name": "'n00110'"},
                {"c.name": "'n00111'"},
                {"c.name": "'n01000'"},
                {"c.name": "'n01001'"},
                {"c.name": "'n01010'"},
                {"c.name": "'n01011'"},
                {"c.name": "'n01100'"},
                {"c.name": "'n01101'"},
                {"c.name": "'n01110'"},
                {"c.name": "'n01111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching, direction mixing, and MATCH/DELETE/CREATE setup are not supported",
        tags=("match", "variable-length", "direction", "xfail"),
    ),
    Scenario(
        key="match5-28",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[28] Handling mixed relationship patterns 1",
        cypher="MATCH (a:A)\nMATCH (p)-[:LIKES*1]->()-[:LIKES]->()-[r:LIKES*2]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00000'"},
                {"c.name": "'n00001'"},
                {"c.name": "'n00010'"},
                {"c.name": "'n00011'"},
                {"c.name": "'n00100'"},
                {"c.name": "'n00101'"},
                {"c.name": "'n00110'"},
                {"c.name": "'n00111'"},
                {"c.name": "'n01000'"},
                {"c.name": "'n01001'"},
                {"c.name": "'n01010'"},
                {"c.name": "'n01011'"},
                {"c.name": "'n01100'"},
                {"c.name": "'n01101'"},
                {"c.name": "'n01110'"},
                {"c.name": "'n01111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and MATCH/CREATE setup expressions are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match5-29",
        feature_path="tck/features/clauses/match/Match5.feature",
        scenario="[29] Handling mixed relationship patterns 2",
        cypher="MATCH (a:A)\nMATCH (p)-[:LIKES]->()-[:LIKES*2]->()-[r:LIKES]->(c)\nRETURN c.name",
        graph=MATCH5_GRAPH,
        expected=Expected(
            rows=[
                {"c.name": "'n00000'"},
                {"c.name": "'n00001'"},
                {"c.name": "'n00010'"},
                {"c.name": "'n00011'"},
                {"c.name": "'n00100'"},
                {"c.name": "'n00101'"},
                {"c.name": "'n00110'"},
                {"c.name": "'n00111'"},
                {"c.name": "'n01000'"},
                {"c.name": "'n01001'"},
                {"c.name": "'n01010'"},
                {"c.name": "'n01011'"},
                {"c.name": "'n01100'"},
                {"c.name": "'n01101'"},
                {"c.name": "'n01110'"},
                {"c.name": "'n01111'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship matching and MATCH/CREATE setup expressions are not supported",
        tags=("match", "variable-length", "xfail"),
    ),
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
    Scenario(
        key="match7-1",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[1] Simple OPTIONAL MATCH on empty graph",
        cypher="OPTIONAL MATCH (n)\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"n": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and null row preservation are not supported",
        tags=("match", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="match7-2",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[2] OPTIONAL MATCH with previously bound nodes",
        cypher="MATCH (n)\nOPTIONAL MATCH (n)-[:NOT_EXIST]->(x)\nRETURN n, x",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"n": "()", "x": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="match7-3",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[3] OPTIONAL MATCH and bound nodes",
        cypher="MATCH (a:A), (b:C)\nOPTIONAL MATCH (x)-->(b)\nRETURN x",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"x": "(:A {num: 42})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "xfail"),
    ),
    Scenario(
        key="match7-4",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[4] Optionally matching relationship with bound nodes in reverse direction",
        cypher="MATCH (a1)-[r]->()\nWITH r, a1\n  LIMIT 1\nOPTIONAL MATCH (a1)<-[r]-(b2)\nRETURN a1, r, b2",
        graph=MATCH7_GRAPH_AB,
        expected=Expected(
            rows=[
                {"a1": "(:A)", "r": "[:T]", "b2": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping and OPTIONAL MATCH semantics are not supported",
        tags=("match", "optional-match", "with", "limit", "xfail"),
    ),
    Scenario(
        key="match7-5",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[5] Optionally matching relationship with a relationship that is already bound",
        cypher="MATCH ()-[r]->()\nWITH r\n  LIMIT 1\nOPTIONAL MATCH (a2)-[r]->(b2)\nRETURN a2, r, b2",
        graph=MATCH7_GRAPH_AB,
        expected=Expected(
            rows=[
                {"a2": "(:A)", "r": "[:T]", "b2": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping and OPTIONAL MATCH semantics are not supported",
        tags=("match", "optional-match", "with", "limit", "xfail"),
    ),
    Scenario(
        key="match7-6",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[6] Optionally matching relationship with a relationship and node that are both already bound",
        cypher="MATCH (a1)-[r]->()\nWITH r, a1\n  LIMIT 1\nOPTIONAL MATCH (a1)-[r]->(b2)\nRETURN a1, r, b2",
        graph=MATCH7_GRAPH_AB,
        expected=Expected(
            rows=[
                {"a1": "(:A)", "r": "[:T]", "b2": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT scoping and OPTIONAL MATCH semantics are not supported",
        tags=("match", "optional-match", "with", "limit", "xfail"),
    ),
    Scenario(
        key="match7-7",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[7] MATCH with OPTIONAL MATCH in longer pattern",
        cypher="MATCH (a {name: 'A'})\nOPTIONAL MATCH (a)-[:KNOWS]->()-[:KNOWS]->(foo)\nRETURN foo",
        graph=MATCH7_GRAPH_ABC,
        expected=Expected(
            rows=[
                {"foo": "({name: 'C'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "xfail"),
    ),
    Scenario(
        key="match7-8",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[8] Longer pattern with bound nodes without matches",
        cypher="MATCH (a:A), (c:C)\nOPTIONAL MATCH (a)-->(b)-->(c)\nRETURN b",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"b": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and null propagation are not supported",
        tags=("match", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="match7-9",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[9] Longer pattern with bound nodes",
        cypher="MATCH (a:Single), (c:C)\nOPTIONAL MATCH (a)-->(b)-->(c)\nRETURN b",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"b": "(:A {num: 42})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "xfail"),
    ),
    Scenario(
        key="match7-10",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[10] Optionally matching from null nodes should return null",
        cypher="OPTIONAL MATCH (a)\nWITH a\nOPTIONAL MATCH (a)-->(b)\nRETURN b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"b": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("match", "optional-match", "with", "null", "xfail"),
    ),
    Scenario(
        key="match7-11",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[11] Return two subgraphs with bound undirected relationship and optional relationship",
        cypher="MATCH (a)-[r {name: 'r1'}]-(b)\nOPTIONAL MATCH (b)-[r2]-(c)\nWHERE r <> r2\nRETURN a, b, c",
        graph=MATCH7_GRAPH_REL,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1})", "b": "(:B {num: 2})", "c": "(:C {num: 3})"},
                {"a": "(:B {num: 2})", "b": "(:A {num: 1})", "c": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable comparison joins, and row projections are not supported",
        tags=("match", "optional-match", "join", "xfail"),
    ),
    Scenario(
        key="match7-12",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[12] Variable length optional relationships",
        cypher="MATCH (a:Single)\nOPTIONAL MATCH (a)-[*]->(b)\nRETURN b",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"b": "(:A {num: 42})"},
                {"b": "(:B {num: 46})"},
                {"b": "(:B {num: 46})"},
                {"b": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length relationships, and row projections are not supported",
        tags=("match", "optional-match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match7-13",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[13] Variable length optional relationships with bound nodes",
        cypher="MATCH (a:Single), (x:C)\nOPTIONAL MATCH (a)-[*]->(x)\nRETURN x",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"x": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and variable-length relationships are not supported",
        tags=("match", "optional-match", "variable-length", "xfail"),
    ),
    Scenario(
        key="match7-14",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[14] Variable length optional relationships with length predicates",
        cypher="MATCH (a:Single)\nOPTIONAL MATCH (a)-[*3..]-(b)\nRETURN b",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"b": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and variable-length relationships are not supported",
        tags=("match", "optional-match", "variable-length", "null", "xfail"),
    ),
    Scenario(
        key="match7-15",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[15] Variable length patterns and nulls",
        cypher="MATCH (a:A)\nOPTIONAL MATCH (a)-[:FOO]->(b:B)\nOPTIONAL MATCH (b)<-[:BAR*]-(c:B)\nRETURN a, b, c",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "null", "c": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and variable-length relationships are not supported",
        tags=("match", "optional-match", "variable-length", "null", "xfail"),
    ),
    Scenario(
        key="match7-16",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[16] Optionally matching named paths - null result",
        cypher="MATCH (a:A)\nOPTIONAL MATCH p = (a)-[:X]->(b)\nRETURN p",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and named path returns are not supported",
        tags=("match", "optional-match", "path", "null", "xfail"),
    ),
    Scenario(
        key="match7-17",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[17] Optionally matching named paths - existing result",
        cypher="MATCH (a {name: 'A'}), (x)\nWHERE x.name IN ['B', 'C']\nOPTIONAL MATCH p = (a)-->(x)\nRETURN x, p",
        graph=MATCH7_GRAPH_X,
        expected=Expected(
            rows=[
                {"x": "({name: 'B'})", "p": "<({name: 'A'})-[:X]->({name: 'B'})>"},
                {"x": "({name: 'C'})", "p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, IN predicates, and named path returns are not supported",
        tags=("match", "optional-match", "path", "in-predicate", "xfail"),
    ),
    Scenario(
        key="match7-18",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[18] Named paths inside optional matches with node predicates",
        cypher="MATCH (a:A), (b:B)\nOPTIONAL MATCH p = (a)-[:X]->(b)\nRETURN p",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and named path returns are not supported",
        tags=("match", "optional-match", "path", "xfail"),
    ),
    Scenario(
        key="match7-19",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[19] Optionally matching named paths with single and variable length patterns",
        cypher="MATCH (a {name: 'A'})\nOPTIONAL MATCH p = (a)-->(b)-[*]->(c)\nRETURN p",
        graph=MATCH7_GRAPH_AB_X,
        expected=Expected(
            rows=[
                {"p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length relationships, and named path returns are not supported",
        tags=("match", "optional-match", "path", "variable-length", "xfail"),
    ),
    Scenario(
        key="match7-20",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[20] Variable length optional relationships with bound nodes, no matches",
        cypher="MATCH (a:A), (b:B)\nOPTIONAL MATCH p = (a)-[*]->(b)\nRETURN p",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length relationships, and named path returns are not supported",
        tags=("match", "optional-match", "path", "variable-length", "null", "xfail"),
    ),
    Scenario(
        key="match7-21",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[21] Handling optional matches between nulls",
        cypher="OPTIONAL MATCH (a:NotThere)\nOPTIONAL MATCH (b:NotThere)\nWITH a, b\nOPTIONAL MATCH (b)-[r:NOR_THIS]->(a)\nRETURN a, b, r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"a": "null", "b": "null", "r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("match", "optional-match", "with", "null", "xfail"),
    ),
    Scenario(
        key="match7-22",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[22] MATCH after OPTIONAL MATCH",
        cypher="MATCH (a:Single)\nOPTIONAL MATCH (a)-->(b:NonExistent)\nOPTIONAL MATCH (a)-->(c:NonExistent)\nWITH coalesce(b, c) AS x\nMATCH (x)-->(d)\nRETURN d",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and row projections are not supported",
        tags=("match", "optional-match", "with", "xfail"),
    ),
    Scenario(
        key="match7-23",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[23] OPTIONAL MATCH with labels on the optional end node",
        cypher="MATCH (a:X)\nOPTIONAL MATCH (a)-->(b:Y)\nRETURN b",
        graph=MATCH7_GRAPH_LABELS,
        expected=Expected(
            rows=[
                {"b": "null"},
                {"b": "(:Y)"},
                {"b": "(:Y:Z)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and label predicates on relationship endpoints are not supported",
        tags=("match", "optional-match", "label", "xfail"),
    ),
    Scenario(
        key="match7-24",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[24] Optionally matching self-loops",
        cypher="MATCH (a:B)\nOPTIONAL MATCH (a)-[r]-(a)\nRETURN r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"r": "[:LOOP]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "relationship", "xfail"),
    ),
    Scenario(
        key="match7-25",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[25] Optionally matching self-loops without matches",
        cypher="MATCH (a)\nWHERE NOT (a:B)\nOPTIONAL MATCH (a)-[r]->(a)\nRETURN r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"r": "null"},
                {"r": "null"},
                {"r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, label predicates in WHERE, and null handling are not supported",
        tags=("match", "optional-match", "label-predicate", "null", "xfail"),
    ),
    Scenario(
        key="match7-26",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[26] Handling correlated optional matches; first does not match implies second does not match",
        cypher="MATCH (a:A), (b:B)\nOPTIONAL MATCH (a)-->(x)\nOPTIONAL MATCH (x)-[r]->(b)\nRETURN x, r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"x": "(:C)", "r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and row projections are not supported",
        tags=("match", "optional-match", "xfail"),
    ),
    Scenario(
        key="match7-27",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[27] Handling optional matches between optionally matched entities",
        cypher="OPTIONAL MATCH (a:NotThere)\nWITH a\nMATCH (b:B)\nWITH a, b\nOPTIONAL MATCH (b)-[r:NOR_THIS]->(a)\nRETURN a, b, r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"a": "null", "b": "(:B {num: 46})", "r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and row projections are not supported",
        tags=("match", "optional-match", "with", "null", "xfail"),
    ),
    Scenario(
        key="match7-28",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[28] Handling optional matches with inline label predicate",
        cypher="MATCH (n:Single)\nOPTIONAL MATCH (n)-[r]-(m:NonExistent)\nRETURN r",
        graph=MATCH7_GRAPH_SINGLE,
        expected=Expected(
            rows=[
                {"r": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and label predicates on relationship endpoints are not supported",
        tags=("match", "optional-match", "label", "xfail"),
    ),
    Scenario(
        key="match7-29",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[29] Satisfies the open world assumption, relationships between same nodes",
        cypher="MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)\nOPTIONAL MATCH (p)-[s:SUPPORTS]->(team)\nRETURN count(*) AS matches, s IS NULL AS optMatch",
        graph=MATCH7_GRAPH_PLAYER_TEAM_BOTH,
        expected=Expected(
            rows=[
                {"matches": 1, "optMatch": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, aggregations, and IS NULL projections are not supported",
        tags=("match", "optional-match", "aggregation", "is-null", "xfail"),
    ),
    Scenario(
        key="match7-30",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[30] Satisfies the open world assumption, single relationship",
        cypher="MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)\nOPTIONAL MATCH (p)-[s:SUPPORTS]->(team)\nRETURN count(*) AS matches, s IS NULL AS optMatch",
        graph=MATCH7_GRAPH_PLAYER_TEAM_SINGLE,
        expected=Expected(
            rows=[
                {"matches": 1, "optMatch": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, aggregations, and IS NULL projections are not supported",
        tags=("match", "optional-match", "aggregation", "is-null", "xfail"),
    ),
    Scenario(
        key="match7-31",
        feature_path="tck/features/clauses/match/Match7.feature",
        scenario="[31] Satisfies the open world assumption, relationships between different nodes",
        cypher="MATCH (p:Player)-[:PLAYS_FOR]->(team:Team)\nOPTIONAL MATCH (p)-[s:SUPPORTS]->(team)\nRETURN count(*) AS matches, s IS NULL AS optMatch",
        graph=MATCH7_GRAPH_PLAYER_TEAM_DIFF,
        expected=Expected(
            rows=[
                {"matches": 1, "optMatch": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, aggregations, and IS NULL projections are not supported",
        tags=("match", "optional-match", "aggregation", "is-null", "xfail"),
    ),
    Scenario(
        key="match8-1",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[1] Pattern independent of bound variables results in cross product",
        cypher="MATCH (a)\nWITH a\nMATCH (b)\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:A)"},
                {"a": "(:A)", "b": "(:B)"},
                {"a": "(:B)", "b": "(:A)"},
                {"a": "(:B)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, cartesian products, and row projections are not supported",
        tags=("match", "with", "cartesian", "xfail"),
    ),
    Scenario(
        key="match8-2",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[2] Counting rows after MATCH, MERGE, OPTIONAL MATCH",
        cypher="MATCH (a)\nMERGE (b)\nWITH *\nOPTIONAL MATCH (a)--(b)\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T1]->(b),
                   (b)-[:T2]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="MERGE, OPTIONAL MATCH semantics, aggregations, and row projections are not supported",
        tags=("match", "merge", "optional-match", "aggregation", "xfail"),
    ),
    Scenario(
        key="match8-3",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[3] Matching and disregarding output, then matching again",
        cypher="MATCH ()-->()\nWITH 1 AS x\nMATCH ()-[r1]->()<--()\nRETURN sum(r1.times)",
        graph=graph_fixture_from_create(
            """
            CREATE (andres {name: 'Andres'}),
                   (michael {name: 'Michael'}),
                   (peter {name: 'Peter'}),
                   (bread {type: 'Bread'}),
                   (veggies {type: 'Veggies'}),
                   (meat {type: 'Meat'})
            CREATE (andres)-[:ATE {times: 10}]->(bread),
                   (andres)-[:ATE {times: 8}]->(veggies),
                   (michael)-[:ATE {times: 4}]->(veggies),
                   (michael)-[:ATE {times: 6}]->(bread),
                   (michael)-[:ATE {times: 9}]->(meat),
                   (peter)-[:ATE {times: 7}]->(veggies),
                   (peter)-[:ATE {times: 7}]->(bread),
                   (peter)-[:ATE {times: 4}]->(meat)
            """
        ),
        expected=Expected(
            rows=[
                {"sum(r1.times)": 776},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and row projections are not supported",
        tags=("match", "with", "aggregation", "xfail"),
    ),
    Scenario(
        key="match9-1",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[1] Variable length relationship variables are lists of relationships",
        cypher="MATCH ()-[r*0..1]-()\nRETURN last(r) AS l",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (c)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"l": "[:T]"},
                {"l": "[:T]"},
                {"l": "null"},
                {"l": "null"},
                {"l": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists, list functions, and row projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-2",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[2] Return relationships by collecting them as a list - directed, one way",
        cypher="MATCH (a)-[r:REL*2..2]->(b:End)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(e:End)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num: 1}], [:REL {num: 2}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-3",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[3] Return relationships by collecting them as a list - undirected, starting from two extremes",
        cypher="MATCH (a)-[r:REL*2..2]-(b:End)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:End)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num:1}], [:REL {num:2}]]"},
                {"r": "[[:REL {num:2}], [:REL {num:1}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-4",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[4] Return relationships by collecting them as a list - undirected, starting from one extreme",
        cypher="MATCH (a:Start)-[r:REL*2..2]-(b)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Start)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:C)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num: 1}], [:REL {num: 2}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-5",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[5] Variable length pattern with label predicate on both sides",
        cypher="MATCH (a:Blue)-[r*]->(b:Green)\nRETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Blue), (b:Red), (c:Green), (d:Yellow)
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c),
                   (b)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"count(r)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, label predicates, and aggregations are not supported",
        tags=("match", "variable-length", "label", "aggregation", "xfail"),
    ),
    Scenario(
        key="match9-6",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[6] Matching relationships into a list and matching variable length using the list, with bound nodes",
        cypher="MATCH (a)-[r1]->()-[r2]->(b)\nWITH [r1, r2] AS rs, a AS first, b AS second\n  LIMIT 1\nMATCH (first)-[rs*]->(second)\nRETURN first, second",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:Y]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"first": "(:A)", "second": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT pipelines, variable-length patterns, and relationship list matching are not supported",
        tags=("match", "with", "limit", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-7",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[7] Matching relationships into a list and matching variable length using the list, with bound nodes, wrong direction",
        cypher="MATCH (a)-[r1]->()-[r2]->(b)\nWITH [r1, r2] AS rs, a AS second, b AS first\n  LIMIT 1\nMATCH (first)-[rs*]->(second)\nRETURN first, second",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:Y]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT pipelines, variable-length patterns, and relationship list matching are not supported",
        tags=("match", "with", "limit", "variable-length", "list", "xfail"),
    ),
    Scenario(
        key="match9-8",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[8] Variable length relationship in OPTIONAL MATCH",
        cypher="MATCH (a:A), (b:B)\nOPTIONAL MATCH (a)-[r*]-(b)\nWHERE r IS NULL\n  AND a <> b\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length patterns, null predicates, and variable comparisons are not supported",
        tags=("match", "optional-match", "variable-length", "null", "comparison", "xfail"),
    ),
    Scenario(
        key="match9-9",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[9] Optionally matching named paths with variable length patterns",
        cypher="MATCH (a {name: 'A'}), (x)\nWHERE x.name IN ['B', 'C']\nOPTIONAL MATCH p = (a)-[r*]->(x)\nRETURN r, x, p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
            CREATE (a)-[:X]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:X]]", "x": "({name: 'B'})", "p": "<({name: 'A'})-[:X]->({name: 'B'})>"},
                {"r": "null", "x": "({name: 'C'})", "p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length relationship lists, IN predicates, and named path returns are not supported",
        tags=("match", "optional-match", "variable-length", "list", "path", "in-predicate", "xfail"),
    ),
    Scenario(
        key="return1-1",
        feature_path="tck/features/clauses/return/Return1.feature",
        scenario="[1] Returning a list property",
        cypher="MATCH (n)\nRETURN n",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "numbers": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            node_ids=["n1"],
            rows=[
                {"n": "({numbers: [1, 2, 3]})"},
            ],
        ),
        gfql=[n()],
        tags=("return", "list-property"),
    ),
    Scenario(
        key="return1-2",
        feature_path="tck/features/clauses/return/Return1.feature",
        scenario="[2] Fail when returning an undefined variable",
        cypher="MATCH ()\nRETURN foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for undefined variables is not enforced",
        tags=("return", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return2-1",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[1] Arithmetic expressions should propagate null values",
        cypher="RETURN 1 + (2 - (3 * (4 / (5 ^ (6 % null))))) AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expression evaluation and null arithmetic semantics are not supported",
        tags=("return", "expression", "null", "xfail"),
    ),
    Scenario(
        key="return2-2",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[2] Returning a node property value",
        cypher="MATCH (a)\nRETURN a.num",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a.num": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "xfail"),
    ),
    Scenario(
        key="return2-3",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[3] Missing node property should become null",
        cypher="MATCH (a)\nRETURN a.name",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a.name": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections and null semantics are not supported",
        tags=("return", "property", "null", "xfail"),
    ),
    Scenario(
        key="return2-4",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[4] Returning a relationship property value",
        cypher="MATCH ()-[r]->()\nRETURN r.num",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {num: 1}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"r.num": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN relationship property projections are not supported",
        tags=("return", "relationship", "property", "xfail"),
    ),
    Scenario(
        key="return2-5",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[5] Missing relationship property should become null",
        cypher="MATCH ()-[r]->()\nRETURN r.name2",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {name: 1}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"r.name2": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN relationship property projections and null semantics are not supported",
        tags=("return", "relationship", "property", "null", "xfail"),
    ),
    Scenario(
        key="return2-6",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[6] Adding a property and a literal in projection",
        cypher="MATCH (a)\nRETURN a.num + 1 AS foo",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"foo": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN arithmetic expression evaluation is not supported",
        tags=("return", "expression", "arithmetic", "xfail"),
    ),
    Scenario(
        key="return2-7",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[7] Adding list properties in projection",
        cypher="MATCH (a)\nRETURN a.list2 + a.list1 AS foo",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list1": [1, 2, 3], "list2": [4, 5]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"foo": "[4, 5, 1, 2, 3]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN list expressions and list property projections are not supported",
        tags=("return", "list", "expression", "xfail"),
    ),
    Scenario(
        key="return2-8",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[8] Returning label predicate expression",
        cypher="MATCH (n)\nRETURN (n:Foo)",
        graph=graph_fixture_from_create(
            """
            CREATE (), (:Foo)
            """
        ),
        expected=Expected(
            rows=[
                {"(n:Foo)": "true"},
                {"(n:Foo)": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN label predicate expressions are not supported",
        tags=("return", "label-predicate", "xfail"),
    ),
    Scenario(
        key="return2-9",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[9] Returning a projected map",
        cypher="RETURN {a: 1, b: 'foo'}",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "numbers": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"{a: 1, b: 'foo'}": "{a: 1, b: 'foo'}"},
                {"{a: 1, b: 'foo'}": "{a: 1, b: 'foo'}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN map projections are not supported",
        tags=("return", "map", "xfail"),
    ),
    Scenario(
        key="return2-10",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[10] Return count aggregation over an empty graph",
        cypher="MATCH (a)\nRETURN count(a) > 0",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"count(a) > 0": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and boolean projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return2-11",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[11] RETURN does not lose precision on large integers",
        cypher="MATCH (p:TheLabel)\nRETURN p.id",
        graph=graph_fixture_from_create(
            """
            CREATE (:TheLabel {id: 4611686018427387905})
            """
        ),
        expected=Expected(
            rows=[
                {"p.id": 4611686018427387905},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "big-int", "xfail"),
    ),
    Scenario(
        key="return2-12",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[12] Projecting a list of nodes and relationships",
        cypher="MATCH (n)-[r]->(m)\nRETURN [n, r, m] AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[(:A), [:T], (:B)]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN list projections of nodes and relationships are not supported",
        tags=("return", "list", "projection", "xfail"),
    ),
    Scenario(
        key="return2-13",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[13] Projecting a map of nodes and relationships",
        cypher="MATCH (n)-[r]->(m)\nRETURN {node1: n, rel: r, node2: m} AS m",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"m": "{node1: (:A), rel: [:T], node2: (:B)}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN map projections of nodes and relationships are not supported",
        tags=("return", "map", "projection", "xfail"),
    ),
    Scenario(
        key="return2-14",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[14] Do not fail when returning type of deleted relationships",
        cypher="MATCH ()-[r]->()\nDELETE r\nRETURN type(r)",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"type(r)": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and return expressions on deleted relationships are not supported",
        tags=("return", "delete", "type", "xfail"),
    ),
    Scenario(
        key="return2-15",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[15] Fail when returning properties of deleted nodes",
        cypher="MATCH (n)\nDELETE n\nRETURN n.num",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return2-16",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[16] Fail when returning labels of deleted nodes",
        cypher="MATCH (n)\nDELETE n\nRETURN labels(n)",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return2-17",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[17] Fail when returning properties of deleted relationships",
        cypher="MATCH ()-[r]->()\nDELETE r\nRETURN r.num",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T {num: 0}]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="DELETE semantics and deleted-entity runtime errors are not supported",
        tags=("return", "delete", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return2-18",
        feature_path="tck/features/clauses/return/Return2.feature",
        scenario="[18] Fail on projecting a non-existent function",
        cypher="MATCH (a)\nRETURN foo(a)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for unknown functions is not enforced",
        tags=("return", "syntax-error", "function", "xfail"),
    ),
    Scenario(
        key="return3-1",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[1] Returning multiple expressions",
        cypher="MATCH (a)\nRETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b",
        graph=GraphFixture(
            nodes=[
                {"node_id": "n1", "labels": []},
            ],
            edges=[],
            node_id="node_id",
            node_columns=("node_id", "labels"),
        ),
        expected=Expected(
            rows=[
                {"a": "false", "b": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expression projections and null predicates are not supported",
        tags=("return", "expression", "null", "xfail"),
    ),
    Scenario(
        key="return3-2",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[2] Returning multiple node property values",
        cypher="MATCH (a)\nRETURN a.name, a.age, a.seasons",
        graph=GraphFixture(
            nodes=[
                {
                    "id": "n1",
                    "labels": [],
                    "name": "Philip J. Fry",
                    "age": 2046,
                    "seasons": [1, 2, 3, 4, 5, 6, 7],
                }
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {
                    "a.name": "'Philip J. Fry'",
                    "a.age": 2046,
                    "a.seasons": "[1, 2, 3, 4, 5, 6, 7]",
                },
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN property projections are not supported",
        tags=("return", "property", "xfail"),
    ),
    Scenario(
        key="return3-3",
        feature_path="tck/features/clauses/return/Return3.feature",
        scenario="[3] Projecting nodes and relationships",
        cypher="MATCH (a)-[r]->()\nRETURN a AS foo, r AS bar",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"foo": "(:A)", "bar": "[:T]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN projections of nodes and relationships are not supported",
        tags=("return", "projection", "xfail"),
    ),
    Scenario(
        key="return4-1",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[1] Honour the column name for RETURN items",
        cypher="MATCH (a)\nWITH a.name AS a\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'Someone'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "'Someone'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and RETURN property projections are not supported",
        tags=("return", "with", "projection", "xfail"),
    ),
    Scenario(
        key="return4-2",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[2] Support column renaming",
        cypher="MATCH (a)\nRETURN a AS ColumnName",
        graph=graph_fixture_from_create(
            """
            CREATE (:Singleton)
            """
        ),
        expected=Expected(
            rows=[
                {"ColumnName": "(:Singleton)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN column renaming and projections are not supported",
        tags=("return", "alias", "projection", "xfail"),
    ),
    Scenario(
        key="return4-3",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[3] Aliasing expressions",
        cypher="MATCH (a)\nRETURN a.id AS a, a.id",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"a": 42, "a.id": 42},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expression projections are not supported",
        tags=("return", "expression", "projection", "xfail"),
    ),
    Scenario(
        key="return4-4",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[4] Keeping used expression 1",
        cypher="MATCH (n)\nRETURN cOuNt( * )",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"cOuNt( * )": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return4-5",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[5] Keeping used expression 2",
        cypher="MATCH p = (n)-->(b)\nRETURN nOdEs( p )",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Path expressions and RETURN projections are not supported",
        tags=("return", "path", "projection", "xfail"),
    ),
    Scenario(
        key="return4-6",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[6] Keeping used expression 3",
        cypher="MATCH p = (n)-->(b)\nRETURN coUnt( dIstInct p )",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"coUnt( dIstInct p )": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations, DISTINCT, and path expressions are not supported",
        tags=("return", "aggregation", "distinct", "path", "xfail"),
    ),
    Scenario(
        key="return4-7",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[7] Keeping used expression 4",
        cypher="MATCH p = (n)-->(b)\nRETURN aVg(    n.aGe     )",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"aVg(    n.aGe     )": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "projection", "xfail"),
    ),
    Scenario(
        key="return4-8",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[8] Support column renaming for aggregations",
        cypher="MATCH ()\nRETURN count(*) AS columnName",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(11)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"columnName": 11},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "alias", "xfail"),
    ),
    Scenario(
        key="return4-9",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[9] Handle subexpression in aggregation also occurring as standalone expression with nested aggregation in a literal map",
        cypher="MATCH (a:A), (b:B)\nRETURN coalesce(a.num, b.num) AS foo,\n  b.num AS bar,\n  {name: count(b)} AS baz",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B {num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"foo": 42, "bar": 42, "baz": "{name: 1}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN expressions, aggregations, and map projections are not supported",
        tags=("return", "aggregation", "expression", "map", "xfail"),
    ),
    Scenario(
        key="return4-10",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[10] Fail when returning multiple columns with same name",
        cypher="RETURN 1 AS a, 2 AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for duplicate column names is not enforced",
        tags=("return", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return4-11",
        feature_path="tck/features/clauses/return/Return4.feature",
        scenario="[11] Reusing variable names in RETURN",
        cypher="MATCH (person:Person)<--(message)<-[like]-(:Person)\nWITH like.creationDate AS likeTime, person AS person\n  ORDER BY likeTime, message.id\nWITH head(collect({likeTime: likeTime})) AS latestLike, person AS person\nRETURN latestLike.likeTime AS likeTime\n  ORDER BY likeTime",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Person), (b:Person), (m:Message {id: 10})
            CREATE (a)-[:LIKE {creationDate: 20160614}]->(m)-[:POSTED_BY]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"likeTime": 20160614},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, aggregations, and list/map expressions are not supported",
        tags=("return", "with", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return5-1",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[1] DISTINCT inside aggregation should work with lists in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: n.list}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and map/list projections are not supported",
        tags=("return", "distinct", "aggregation", "xfail"),
    ),
    Scenario(
        key="return5-2",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[2] DISTINCT on nullable values",
        cypher="MATCH (n)\nRETURN DISTINCT n.name",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "name": "Florescu"},
                {"id": "n2", "labels": []},
                {"id": "n3", "labels": []},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"n.name": "'Florescu'"},
                {"n.name": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT projections and null semantics are not supported",
        tags=("return", "distinct", "null", "xfail"),
    ),
    Scenario(
        key="return5-3",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[3] DISTINCT inside aggregation should work with nested lists in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: [[n.list, n.list], [n.list, n.list]]}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and nested list projections are not supported",
        tags=("return", "distinct", "aggregation", "list", "xfail"),
    ),
    Scenario(
        key="return5-4",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[4] DISTINCT inside aggregation should work with nested lists of maps in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: [{name2: n.list}, {baz: {apa: n.list}}]}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and nested map/list projections are not supported",
        tags=("return", "distinct", "aggregation", "map", "xfail"),
    ),
    Scenario(
        key="return5-5",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[5] Aggregate on list values",
        cypher="MATCH (a)\nRETURN DISTINCT a.color, count(*)",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "color": ["red"]},
                {"id": "n2", "labels": [], "color": ["blue"]},
                {"id": "n3", "labels": [], "color": ["red"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"a.color": "['red']", "count(*)": 2},
                {"a.color": "['blue']", "count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT projections, aggregations, and list values are not supported",
        tags=("return", "distinct", "aggregation", "list", "xfail"),
    ),
    Scenario(
        key="return6-1",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[1] Return count aggregation over nodes",
        cypher="MATCH (n)\nRETURN n.num AS n, count(n) AS count",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"n": 42, "count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-2",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[2] Projecting an arithmetic expression with aggregation",
        cypher="MATCH (a)\nRETURN a, count(a) + 3",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({id: 42})", "count(a) + 3": 4},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations, arithmetic expressions, and RETURN projections are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="return6-3",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[3] Aggregating by a list property has a correct definition of equality",
        cypher="MATCH (a)\nWITH a.num AS a, count(*) AS count\nRETURN count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "a": [1, 2, 3]},
                {"id": "n2", "labels": [], "a": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and list property projections are not supported",
        tags=("return", "with", "aggregation", "list", "xfail"),
    ),
    Scenario(
        key="return6-4",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[4] Support multiple divisions in aggregate function",
        cypher="MATCH (n)\nRETURN count(n) / 60 / 60 AS count",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(7251)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and arithmetic expressions are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="return6-5",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[5] Aggregates inside normal functions",
        cypher="MATCH (a)\nRETURN size(collect(a))",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(11)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"size(collect(a))": 11},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations, list functions, and RETURN projections are not supported",
        tags=("return", "aggregation", "list", "xfail"),
    ),
    Scenario(
        key="return6-6",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[6] Handle aggregates inside non-aggregate expressions",
        cypher="MATCH (a {name: 'Andres'})<-[:FATHER]-(child)\nRETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations inside map projections are not supported",
        tags=("return", "aggregation", "map", "xfail"),
    ),
    Scenario(
        key="return6-7",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[7] Aggregate on property",
        cypher="MATCH (n)\nRETURN n.num, count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 33})
            CREATE ({num: 33})
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"n.num": 42, "count(*)": 1},
                {"n.num": 33, "count(*)": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-8",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[8] Handle aggregation on functions",
        cypher="MATCH p=(a:L)-[*]->(b)\nRETURN b, avg(length(p))",
        graph=graph_fixture_from_create(
            """
            CREATE (a:L), (b1), (b2)
            CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "()", "avg(length(p))": 1.0},
                {"b": "()", "avg(length(p))": 1.0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, path length functions, and aggregations are not supported",
        tags=("return", "variable-length", "path", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-9",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[9] Aggregates with arithmetics",
        cypher="MATCH ()\nRETURN count(*) * 10 AS c",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"c": 10},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and arithmetic expressions are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="return6-10",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[10] Multiple aggregates on same variable",
        cypher="MATCH (n)\nRETURN count(n), collect(n)",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"count(n)": 1, "collect(n)": "[()]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and list projections are not supported",
        tags=("return", "aggregation", "list", "xfail"),
    ),
    Scenario(
        key="return6-11",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[11] Counting matches",
        cypher="MATCH ()\nRETURN count(*)",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(100)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count(*)": 100},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-12",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[12] Counting matches per group",
        cypher="MATCH (a:L)-[rel]->(b)\nRETURN a, count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:L), (b1), (b2)
            CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:L)", "count(*)": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-13",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[13] Returning the minimum length of paths",
        cypher="MATCH p = (a:T {name: 'a'})-[:R*]->(other:T)\nWHERE other <> a\nWITH a, other, min(length(p)) AS len\nRETURN a.name AS name, collect(other.name) AS others, len",
        graph=graph_fixture_from_create(
            """
            CREATE (a:T {name: 'a'}), (b:T {name: 'b'}), (c:T {name: 'c'})
            CREATE (a)-[:R]->(b)
            CREATE (a)-[:R]->(c)
            CREATE (c)-[:R]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'a'", "others": "['c', 'b']", "len": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, path length functions, WITH pipelines, and aggregations are not supported",
        tags=("return", "variable-length", "path", "with", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-14",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[14] Aggregates in aggregates",
        cypher="RETURN count(count(*))",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for nested aggregations is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-15",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[15] Using `rand()` in aggregations",
        cypher="RETURN count(rand())",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for non-constant aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-16",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[16] Aggregation on complex expressions",
        cypher="MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)\nWHERE me.name = 'Michael'\nWITH me, count(DISTINCT r1) AS H1, count(DISTINCT r2) AS H2, you\nMATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)\nRETURN me, you, sum((1 - abs(r1.times / H1 - r2.times / H2)) * (r1.times + r2.times) / (H1 + H2)) AS sum",
        graph=graph_fixture_from_create(
            """
            CREATE (andres {name: 'Andres'}),
                   (michael {name: 'Michael'}),
                   (peter {name: 'Peter'}),
                   (bread {type: 'Bread'}),
                   (veggies {type: 'Veggies'}),
                   (meat {type: 'Meat'})
            CREATE (andres)-[:ATE {times: 10}]->(bread),
                   (andres)-[:ATE {times: 8}]->(veggies),
                   (michael)-[:ATE {times: 4}]->(veggies),
                   (michael)-[:ATE {times: 6}]->(bread),
                   (michael)-[:ATE {times: 9}]->(meat),
                   (peter)-[:ATE {times: 7}]->(veggies),
                   (peter)-[:ATE {times: 7}]->(bread),
                   (peter)-[:ATE {times: 4}]->(meat)
            """
        ),
        expected=Expected(
            rows=[
                {"me": "({name: 'Michael'})", "you": "({name: 'Andres'})", "sum": -7},
                {"me": "({name: 'Michael'})", "you": "({name: 'Peter'})", "sum": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, DISTINCT aggregations, and complex expressions are not supported",
        tags=("return", "with", "distinct", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-17",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[17] Handle constants and parameters inside an expression which contains an aggregation expression",
        cypher="MATCH (person)\nRETURN $age + avg(person.age) - 1000",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"$age + avg(person.age) - 1000": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Parameters, aggregations, and RETURN expressions are not supported",
        tags=("return", "params", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-18",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[18] Handle returned variables inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age AS age, you\nRETURN age, age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and RETURN expressions are not supported",
        tags=("return", "with", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-19",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[19] Handle returned property accesses inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age, me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN expressions are not supported",
        tags=("return", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-20",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[20] Fail if not returned variables are used inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="return6-21",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[21] Fail if more complex expressions, even if returned, are used inside expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age + you.age, me.age + you.age + count(*)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="return7-1",
        feature_path="tck/features/clauses/return/Return7.feature",
        scenario="[1] Return all variables",
        cypher="MATCH p = (a:Start)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:Start)-[:T]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:Start)", "b": "()", "p": "<(:Start)-[:T]->()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN * projections and named path returns are not supported",
        tags=("return", "projection", "path", "xfail"),
    ),
    Scenario(
        key="return7-2",
        feature_path="tck/features/clauses/return/Return7.feature",
        scenario="[2] Fail when using RETURN * without variables in scope",
        cypher="MATCH ()\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for RETURN * without scoped variables is not enforced",
        tags=("return", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return8-1",
        feature_path="tck/features/clauses/return/Return8.feature",
        scenario="[1] Return aggregation after With filtering",
        cypher="MATCH (n)\nWITH n\nWHERE n.num = 42\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 43}), ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, WHERE filtering, and aggregations are not supported",
        tags=("return", "with", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby1-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[1] ORDER BY should order booleans in the expected order",
        cypher="UNWIND [true, false] AS bools\nRETURN bools\nORDER BY bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "false"},
                {"bools": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[2] ORDER BY DESC should order booleans in the expected order",
        cypher="UNWIND [true, false] AS bools\nRETURN bools\nORDER BY bools DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "true"},
                {"bools": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-3",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[3] ORDER BY should order strings in the expected order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nRETURN strings\nORDER BY strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "''"},
                {"strings": "' '"},
                {"strings": "'.*'"},
                {"strings": "'one'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-4",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[4] ORDER BY DESC should order strings in the expected order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nRETURN strings\nORDER BY strings DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "'one'"},
                {"strings": "'.*'"},
                {"strings": "' '"},
                {"strings": "''"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-5",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[5] ORDER BY should order ints in the expected order",
        cypher="UNWIND [1, 3, 2] AS ints\nRETURN ints\nORDER BY ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 1},
                {"ints": 2},
                {"ints": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-6",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[6] ORDER BY DESC should order ints in the expected order",
        cypher="UNWIND [1, 3, 2] AS ints\nRETURN ints\nORDER BY ints DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 3},
                {"ints": 2},
                {"ints": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-7",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[7] ORDER BY should order floats in the expected order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nRETURN floats\nORDER BY floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 1.3},
                {"floats": 1.5},
                {"floats": 999.99},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-8",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[8] ORDER BY DESC should order floats in the expected order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nRETURN floats\nORDER BY floats DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 999.99},
                {"floats": 1.5},
                {"floats": 1.3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-9",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[9] ORDER BY should order lists in the expected order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nRETURN lists\nORDER BY lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[]"},
                {"lists": "['a']"},
                {"lists": "['a', 1]"},
                {"lists": "[1]"},
                {"lists": "[1, 'a']"},
                {"lists": "[1, null]"},
                {"lists": "[null, 1]"},
                {"lists": "[null, 2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-10",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[10] ORDER BY DESC should order lists in the expected order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nRETURN lists\nORDER BY lists DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[null, 2]"},
                {"lists": "[null, 1]"},
                {"lists": "[1, null]"},
                {"lists": "[1, 'a']"},
                {"lists": "[1]"},
                {"lists": "['a', 1]"},
                {"lists": "['a']"},
                {"lists": "[]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-11",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[11] ORDER BY should order distinct types in the expected order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nRETURN types\nORDER BY types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "{a: 'map'}"},
                {"types": "(:N)"},
                {"types": "[:REL]"},
                {"types": "['list']"},
                {"types": "<(:N)-[:REL]->()>"},
                {"types": "'text'"},
                {"types": "false"},
                {"types": 1.5},
                {"types": "NaN"},
                {"types": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby1-12",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[12] ORDER BY DESC should order distinct types in the expected order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nRETURN types\nORDER BY types DESC",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "null"},
                {"types": "NaN"},
                {"types": 1.5},
                {"types": "false"},
                {"types": "'text'"},
                {"types": "<(:N)-[:REL]->()>"},
                {"types": "['list']"},
                {"types": "[:REL]"},
                {"types": "(:N)"},
                {"types": "{a: 'map'}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby2-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[1] ORDER BY should return results in ascending order",
        cypher="MATCH (n)\nRETURN n.num AS prop\nORDER BY n.num",
        graph=graph_fixture_from_create(
            """
            CREATE (n1 {num: 1}),
              (n2 {num: 3}),
              (n3 {num: -5})
            """
        ),
        expected=Expected(
            rows=[
                {"prop": -5},
                {"prop": 1},
                {"prop": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and RETURN projections are not supported",
        tags=("return", "orderby", "projection", "xfail"),
    ),
    Scenario(
        key="return-orderby2-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[2] ORDER BY DESC should return results in descending order",
        cypher="MATCH (n)\nRETURN n.num AS prop\nORDER BY n.num DESC",
        graph=graph_fixture_from_create(
            """
            CREATE (n1 {num: 1}),
              (n2 {num: 3}),
              (n3 {num: -5})
            """
        ),
        expected=Expected(
            rows=[
                {"prop": 3},
                {"prop": 1},
                {"prop": -5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and RETURN projections are not supported",
        tags=("return", "orderby", "projection", "xfail"),
    ),
    Scenario(
        key="return-orderby2-3",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[3] Sort on aggregated function",
        cypher="MATCH (n)\nRETURN n.division, max(n.age)\nORDER BY max(n.age)",
        graph=graph_fixture_from_create(
            """
            CREATE ({division: 'A', age: 22}),
              ({division: 'B', age: 33}),
              ({division: 'B', age: 44}),
              ({division: 'C', age: 55})
            """
        ),
        expected=Expected(
            rows=[
                {"n.division": "'A'", "max(n.age)": 22},
                {"n.division": "'B'", "max(n.age)": 44},
                {"n.division": "'C'", "max(n.age)": 55},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and aggregations are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby2-4",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[4] Support sort and distinct",
        cypher="MATCH (a)\nRETURN DISTINCT a\nORDER BY a.name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'A'})"},
                {"a": "({name: 'B'})"},
                {"a": "({name: 'C'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and DISTINCT projections are not supported",
        tags=("return", "orderby", "distinct", "xfail"),
    ),
    Scenario(
        key="return-orderby2-5",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[5] Support ordering by a property after being distinct-ified",
        cypher="MATCH (a)-->(b)\nRETURN DISTINCT b\nORDER BY b.name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and DISTINCT projections are not supported",
        tags=("return", "orderby", "distinct", "xfail"),
    ),
    Scenario(
        key="return-orderby2-6",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[6] Count star should count everything in scope",
        cypher="MATCH (a)\nRETURN a, count(*)\nORDER BY count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (:L1), (:L2), (:L3)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:L1)", "count(*)": 1},
                {"a": "(:L2)", "count(*)": 1},
                {"a": "(:L3)", "count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and aggregations are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby2-7",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[7] Ordering with aggregation",
        cypher="MATCH (n)\nRETURN n.name, count(*) AS foo\nORDER BY n.name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'nisse'})
            """
        ),
        expected=Expected(
            rows=[
                {"n.name": "'nisse'", "foo": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and aggregations are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby2-8",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[8] Returning all variables with ordering",
        cypher="MATCH (n)\nRETURN *\nORDER BY n.id",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1}), ({id: 10})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({id: 1})"},
                {"n": "({id: 10})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="RETURN * projections and ORDER BY are not supported",
        tags=("return", "orderby", "return-star", "xfail"),
    ),
    Scenario(
        key="return-orderby2-9",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[9] Using aliased DISTINCT expression in ORDER BY",
        cypher="MATCH (n)\nRETURN DISTINCT n.id AS id\nORDER BY id DESC",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1}), ({id: 10})
            """
        ),
        expected=Expected(
            rows=[
                {"id": 10},
                {"id": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and DISTINCT projections are not supported",
        tags=("return", "orderby", "distinct", "xfail"),
    ),
    Scenario(
        key="return-orderby2-10",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[10] Returned columns do not change from using ORDER BY",
        cypher="MATCH (n)\nRETURN DISTINCT n\nORDER BY n.id",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1}), ({id: 10})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({id: 1})"},
                {"n": "({id: 10})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and DISTINCT projections are not supported",
        tags=("return", "orderby", "distinct", "xfail"),
    ),
    Scenario(
        key="return-orderby2-11",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[11] Aggregates ordered by arithmetics",
        cypher="MATCH (a:A), (b:X)\nRETURN count(a) * 10 + count(b) * 5 AS x\nORDER BY x",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:X), (:X)
            """
        ),
        expected=Expected(
            rows=[
                {"x": 30},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and aggregation expressions are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby2-12",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[12] Aggregation of named paths",
        cypher="MATCH p = (a)-[*]->(b)\nRETURN collect(nodes(p)) AS paths, length(p) AS l\nORDER BY l",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C), (d:D), (e:E), (f:F)
            CREATE (a)-[:R]->(b)
            CREATE (c)-[:R]->(d)
            CREATE (d)-[:R]->(e)
            CREATE (e)-[:R]->(f)
            """
        ),
        expected=Expected(
            rows=[
                {"paths": "[[(:A), (:B)], [(:C), (:D)], [(:D), (:E)], [(:E), (:F)]]", "l": 1},
                {"paths": "[[(:C), (:D), (:E)], [(:D), (:E), (:F)]]", "l": 2},
                {"paths": "[[(:C), (:D), (:E), (:F)]]", "l": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, path functions, aggregations, and ORDER BY are not supported",
        tags=("return", "orderby", "path", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby2-13",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[13] Fail when sorting on variable removed by DISTINCT",
        cypher="MATCH (a)\nRETURN DISTINCT a.name\nORDER BY a.age",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A', age: 13}), ({name: 'B', age: 12}), ({name: 'C', age: 11})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("return", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-orderby2-14",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy2.feature",
        scenario="[14] Fail on aggregation in ORDER BY after RETURN",
        cypher="MATCH (n)\nRETURN n.num1\nORDER BY max(n.num2)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation expressions is not enforced",
        tags=("return", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-orderby3-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy3.feature",
        scenario="[1] Sort on aggregate function and normal property",
        cypher="MATCH (n)\nRETURN n.division, count(*)\nORDER BY count(*) DESC, n.division ASC",
        graph=graph_fixture_from_create(
            """
            CREATE ({division: 'Sweden'})
            CREATE ({division: 'Germany'})
            CREATE ({division: 'England'})
            CREATE ({division: 'Sweden'})
            """
        ),
        expected=Expected(
            rows=[
                {"n.division": "'Sweden'", "count(*)": 2},
                {"n.division": "'England'", "count(*)": 1},
                {"n.division": "'Germany'", "count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY and aggregations are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby4-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy4.feature",
        scenario="[1] ORDER BY of a column introduced in RETURN should return salient results in ascending order",
        cypher="WITH [0, 1] AS prows, [[2], [3, 4]] AS qrows\nUNWIND prows AS p\nUNWIND qrows[p] AS q\nWITH p, count(q) AS rng\nRETURN p\nORDER BY rng",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"p": 0},
                {"p": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and ORDER BY are not supported",
        tags=("return", "orderby", "with", "unwind", "xfail"),
    ),
    Scenario(
        key="return-orderby4-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy4.feature",
        scenario="[2] Handle projections with ORDER BY",
        cypher="MATCH (c:Crew {name: 'Neo'})\nWITH c, 0 AS relevance\nRETURN c.rank AS rank\nORDER BY relevance, c.rank",
        graph=graph_fixture_from_create(
            """
            CREATE (c1:Crew {name: 'Neo', rank: 1}),
              (c2:Crew {name: 'Neo', rank: 2}),
              (c3:Crew {name: 'Neo', rank: 3}),
              (c4:Crew {name: 'Neo', rank: 4}),
              (c5:Crew {name: 'Neo', rank: 5})
            """
        ),
        expected=Expected(
            rows=[
                {"rank": 1},
                {"rank": 2},
                {"rank": 3},
                {"rank": 4},
                {"rank": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and ORDER BY are not supported",
        tags=("return", "orderby", "with", "xfail"),
    ),
    Scenario(
        key="return-orderby5-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy5.feature",
        scenario="[1] Renaming columns before ORDER BY should return results in ascending order",
        cypher="MATCH (n)\nRETURN n.num AS n\nORDER BY n + 2",
        graph=graph_fixture_from_create(
            """
            CREATE (n1 {num: 1}),
              (n2 {num: 3}),
              (n3 {num: -5})
            """
        ),
        expected=Expected(
            rows=[
                {"n": -5},
                {"n": 1},
                {"n": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="ORDER BY expression evaluation is not supported",
        tags=("return", "orderby", "expression", "xfail"),
    ),
    Scenario(
        key="return-orderby6-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario="[1] Handle constants and parameters inside an order by item which contains an aggregation expression",
        cypher="MATCH (person)\nRETURN avg(person.age) AS avgAge\nORDER BY $age + avg(person.age) - 1000",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"avgAge": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Parameters, aggregations, and ORDER BY are not supported",
        tags=("return", "orderby", "aggregation", "params", "xfail"),
    ),
    Scenario(
        key="return-orderby6-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario="[2] Handle returned aliases inside an order by item which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age AS age, count(you.age) AS cnt\nORDER BY age, age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations and ORDER BY are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby6-3",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario="[3] Handle returned property accesses inside an order by item which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age AS age, count(you.age) AS cnt\nORDER BY me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations and ORDER BY are not supported",
        tags=("return", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-orderby6-4",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario="[4] Fail if not returned variables are used inside an order by item which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN count(you.age) AS agg\nORDER BY me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation expressions is not enforced",
        tags=("return", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-orderby6-5",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy6.feature",
        scenario="[5] Fail if more complex expressions, even if returned, are used inside an order by item which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age + you.age, count(*) AS cnt\nORDER BY me.age + you.age + count(*)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation expressions is not enforced",
        tags=("return", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-1",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[1] Start the result from the second row",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP 2",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
                {"n": "({name: 'E'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP and ORDER BY are not supported",
        tags=("return", "skip", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-2",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[2] Start the result from the second row by param",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP $skipAmount",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
                {"n": "({name: 'E'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP, ORDER BY, and parameter binding are not supported",
        tags=("return", "skip", "orderby", "params", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-3",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[3] SKIP with an expression that does not depend on variables",
        cypher="MATCH (n)\nWITH n SKIP toInteger(rand()*9)\nWITH count(*) AS count\nRETURN count > 0 AS nonEmpty",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "nr": i} for i in range(1, 11)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"nonEmpty": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, SKIP, and functions are not supported",
        tags=("return", "skip", "with", "function", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-4",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[4] Accept skip zero",
        cypher="MATCH (n)\nWHERE 1 = 0\nRETURN n\nSKIP 0",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="SKIP is not supported",
        tags=("return", "skip", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-5",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[5] SKIP with an expression that depends on variables should fail",
        cypher="MATCH (n)\nRETURN n\nSKIP n.count",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for SKIP expressions is not enforced",
        tags=("return", "skip", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-6",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[6] Negative parameter for SKIP should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nSKIP $_skip",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for SKIP are not supported",
        tags=("return", "skip", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-7",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[7] Negative SKIP should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nSKIP -1",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for SKIP arguments is not enforced",
        tags=("return", "skip", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-8",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[8] Floating point parameter for SKIP should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nSKIP $_limit",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for SKIP are not supported",
        tags=("return", "skip", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-9",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[9] Floating point SKIP should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nSKIP 1.5",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for SKIP arguments is not enforced",
        tags=("return", "skip", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-10",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[10] Fail when using non-constants in SKIP",
        cypher="MATCH (n)\nRETURN n\nSKIP n.count",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for SKIP expressions is not enforced",
        tags=("return", "skip", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit1-11",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit1.feature",
        scenario="[11] Fail when using negative value in SKIP",
        cypher="MATCH (n)\nRETURN n\nSKIP -1",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for SKIP arguments is not enforced",
        tags=("return", "skip", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-1",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[1] Limit to two hits",
        cypher="UNWIND [1, 1, 1, 1, 1] AS i\nRETURN i\nLIMIT 2",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"i": 1},
                {"i": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and LIMIT are not supported",
        tags=("return", "limit", "unwind", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-2",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[2] Limit to two hits with explicit order",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nLIMIT 2",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'A'})"},
                {"n": "({name: 'B'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="LIMIT and ORDER BY are not supported",
        tags=("return", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-3",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[3] LIMIT 0 should return an empty result",
        cypher="MATCH (n)\nRETURN n\nLIMIT 0",
        graph=graph_fixture_from_create(
            """
            CREATE (), (), ()
            """
        ),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="LIMIT is not supported",
        tags=("return", "limit", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-4",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[4] Handle ORDER BY with LIMIT 1",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nORDER BY p.name\nLIMIT 1",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
              (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'Craig'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="LIMIT and ORDER BY are not supported",
        tags=("return", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-5",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[5] ORDER BY with LIMIT 0 should not generate errors",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nORDER BY p.name\nLIMIT 0",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="LIMIT and ORDER BY are not supported",
        tags=("return", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-6",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[6] LIMIT with an expression that does not depend on variables",
        cypher="MATCH (n)\nWITH n LIMIT toInteger(ceil(1.7))\nRETURN count(*) AS count",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "nr": i} for i in range(1, 4)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and functions are not supported",
        tags=("return", "limit", "with", "function", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-7",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[7] Limit to more rows than actual results 1",
        cypher="MATCH (foo)\nRETURN foo.num AS x\nORDER BY x DESC\nLIMIT 4",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1}), ({num: 3}), ({num: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"x": 3},
                {"x": 2},
                {"x": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="LIMIT and ORDER BY are not supported",
        tags=("return", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-8",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[8] Limit to more rows than actual results 2",
        cypher="MATCH (a:A)-->(n)-->(m)\nRETURN n.num, count(*)\nORDER BY n.num\nLIMIT 1000",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (n1 {num: 1}), (n2 {num: 2}),
                   (m1), (m2)
            CREATE (a)-[:T]->(n1),
                   (n1)-[:T]->(m1),
                   (a)-[:T]->(n2),
                   (n2)-[:T]->(m2)
            """
        ),
        expected=Expected(
            rows=[
                {"n.num": 1, "count(*)": 1},
                {"n.num": 2, "count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="LIMIT, ORDER BY, and aggregations are not supported",
        tags=("return", "limit", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-9",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[9] Fail when using non-constants in LIMIT",
        cypher="MATCH (n)\nRETURN n\nLIMIT n.count",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for LIMIT expressions is not enforced",
        tags=("return", "limit", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-10",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[10] Negative parameter for LIMIT should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nLIMIT $_limit",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for LIMIT are not supported",
        tags=("return", "limit", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-11",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[11] Negative parameter for LIMIT with ORDER BY should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nORDER BY name\nLIMIT $_limit",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for LIMIT are not supported",
        tags=("return", "limit", "orderby", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-12",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[12] Fail when using negative value in LIMIT 1",
        cypher="MATCH (n)\nRETURN n\nLIMIT -1",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for LIMIT arguments is not enforced",
        tags=("return", "limit", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-13",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[13] Fail when using negative value in LIMIT 2",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nLIMIT -1",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for LIMIT arguments is not enforced",
        tags=("return", "limit", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-14",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[14] Floating point parameter for LIMIT should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nLIMIT $_limit",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for LIMIT are not supported",
        tags=("return", "limit", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-15",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[15] Floating point parameter for LIMIT with ORDER BY should fail",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nORDER BY name\nLIMIT $_limit",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Parameter binding and runtime validation for LIMIT are not supported",
        tags=("return", "limit", "orderby", "params", "runtime-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-16",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[16] Fail when using floating point in LIMIT 1",
        cypher="MATCH (n)\nRETURN n\nLIMIT 1.7",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for LIMIT arguments is not enforced",
        tags=("return", "limit", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit2-17",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit2.feature",
        scenario="[17] Fail when using floating point in LIMIT 2",
        cypher="MATCH (p:Person)\nRETURN p.name AS name\nLIMIT 1.5",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Person {name: 'Steven'}),
                   (c:Person {name: 'Craig'})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for LIMIT arguments is not enforced",
        tags=("return", "limit", "syntax-error", "xfail"),
    ),
    Scenario(
        key="return-skip-limit3-1",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[1] Get rows in the middle",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP 2\nLIMIT 2",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP/LIMIT and ORDER BY are not supported",
        tags=("return", "skip", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="return-skip-limit3-2",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[2] Get rows in the middle by param",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP/LIMIT, ORDER BY, and parameter binding are not supported",
        tags=("return", "skip", "limit", "orderby", "params", "xfail"),
    ),
    Scenario(
        key="return-skip-limit3-3",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[3] Limiting amount of rows when there are fewer left than the LIMIT argument",
        cypher="MATCH (a)\nRETURN a.count\nORDER BY a.count\nSKIP 10\nLIMIT 10",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "count": i} for i in range(16)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"a.count": 10},
                {"a.count": 11},
                {"a.count": 12},
                {"a.count": 13},
                {"a.count": 14},
                {"a.count": 15},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("return", "skip", "limit", "orderby", "unwind", "xfail"),
    ),
    Scenario(
        key="with1-1",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[1] Forwarind a node variable 1",
        cypher="MATCH (a:A)\nWITH a\nMATCH (a)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and RETURN * projections are not supported",
        tags=("with", "pipeline", "return-star", "xfail"),
    ),
    Scenario(
        key="with1-2",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[2] Forwarind a node variable 2",
        cypher="MATCH (a:A)\nWITH a\nMATCH (x:X), (a)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            CREATE (:X)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)", "x": "(:X)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, cartesian products, and RETURN * projections are not supported",
        tags=("with", "pipeline", "cartesian", "return-star", "xfail"),
    ),
    Scenario(
        key="with1-3",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[3] Forwarding a relationship variable",
        cypher="MATCH ()-[r1]->(:X)\nWITH r1 AS r2\nMATCH ()-[r2]->()\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(:X),
                   ()-[:T2]->(:X),
                   ()-[:T3]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and relationship variable aliasing are not supported",
        tags=("with", "relationship", "alias", "xfail"),
    ),
    Scenario(
        key="with1-4",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[4] Forwarding a path variable",
        cypher="MATCH p = (a)\nWITH p\nRETURN p",
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
        reason="WITH pipelines and path variables are not supported",
        tags=("with", "path", "xfail"),
    ),
    Scenario(
        key="with1-5",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[5] Forwarding null",
        cypher="OPTIONAL MATCH (a:Start)\nWITH a\nMATCH (a)-->(b)\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and WITH pipelines are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="with1-6",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[6] Forwarding a node variable possibly null",
        cypher="OPTIONAL MATCH (a:A)\nWITH a AS a\nMATCH (b:B)\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Single), (a:A {num: 42}),
                   (b:B {num: 46}), (c:C)
            CREATE (s)-[:REL]->(a),
                   (s)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (b)-[:LOOP]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 42})", "b": "(:B {num: 46})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and WITH pipelines are not supported",
        tags=("with", "optional-match", "xfail"),
    ),
    Scenario(
        key="with2-1",
        feature_path="tck/features/clauses/with/With2.feature",
        scenario="[1] Forwarding a property to express a join",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\nMATCH (b)\nWHERE b.id = property\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End {num: 42, id: 0}),
                   (:End {num: 3}),
                   (:Begin {num: a.id})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),
    Scenario(
        key="with2-2",
        feature_path="tck/features/clauses/with/With2.feature",
        scenario="[2] Forwarding a nested map literal",
        cypher="WITH {name: {name2: 'baz'}} AS nestedMap\nRETURN nestedMap.name.name2",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"nestedMap.name.name2": "'baz'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and map projections are not supported",
        tags=("with", "map", "projection", "xfail"),
    ),
    Scenario(
        key="with3-1",
        feature_path="tck/features/clauses/with/With3.feature",
        scenario="[1] Forwarding multiple node and relationship variables",
        cypher="MATCH (a)-[r]->(b:X)\nWITH a, r, b\nMATCH (a)-[r]->(b)\nRETURN r AS rel\n  ORDER BY rel.id",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {id: 0}]->(:X),
                   ()-[:T2 {id: 1}]->(:X),
                   ()-[:T2 {id: 2}]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1 {id: 0}]"},
                {"rel": "[:T2 {id: 1}]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, and relationship projections are not supported",
        tags=("with", "orderby", "relationship", "xfail"),
    ),
    Scenario(
        key="with4-1",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[1] Aliasing relationship variable",
        cypher="MATCH ()-[r1]->()\nWITH r1 AS r2\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(),
                   ()-[:T2]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and relationship aliasing are not supported",
        tags=("with", "alias", "relationship", "xfail"),
    ),
    Scenario(
        key="with4-2",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[2] Aliasing expression to new variable name",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\nMATCH (b:End)\nWHERE property = b.num\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (:Begin {num: 42}),
                   (:End {num: 42}),
                   (:End {num: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, expression aliasing, and row projections are not supported",
        tags=("with", "alias", "projection", "xfail"),
    ),
    Scenario(
        key="with4-3",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[3] Aliasing expression to existing variable name",
        cypher="MATCH (n)\nWITH n.name AS n\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1, name: 'King Kong'}),
              ({num: 2, name: 'Ann Darrow'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "'Ann Darrow'"},
                {"n": "'King Kong'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and expression projections are not supported",
        tags=("with", "alias", "projection", "xfail"),
    ),
    Scenario(
        key="with4-4",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[4] Fail when forwarding multiple aliases with the same name",
        cypher="WITH 1 AS a, 2 AS a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for duplicate WITH aliases is not enforced",
        tags=("with", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with4-5",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[5] Fail when not aliasing expressions in WITH",
        cypher="MATCH (a)\nWITH a, count(*)\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for WITH expression aliasing is not enforced",
        tags=("with", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with4-6",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[6] Reusing variable names in WITH",
        cypher="MATCH (person:Person)<--(message)<-[like]-(:Person)\nWITH like.creationDate AS likeTime, person AS person\n  ORDER BY likeTime, message.id\nWITH head(collect({likeTime: likeTime})) AS latestLike, person AS person\nWITH latestLike.likeTime AS likeTime\n  ORDER BY likeTime\nRETURN likeTime",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Person), (b:Person), (m:Message {id: 10})
            CREATE (a)-[:LIKE {creationDate: 20160614}]->(m)-[:POSTED_BY]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"likeTime": 20160614},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, aggregations, and list/map expressions are not supported",
        tags=("with", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="with4-7",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[7] Multiple aliasing and backreferencing",
        cypher="CREATE (m {id: 0})\nWITH {first: m.id} AS m\nWITH {second: m.first} AS m\nRETURN m.second",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"m.second": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, map projections, and side effect validation are not supported",
        tags=("with", "map", "projection", "xfail"),
    ),
    Scenario(
        key="with5-1",
        feature_path="tck/features/clauses/with/With5.feature",
        scenario="[1] DISTINCT on an expression",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections are not supported",
        tags=("with", "distinct", "xfail"),
    ),
    Scenario(
        key="with5-2",
        feature_path="tck/features/clauses/with/With5.feature",
        scenario="[2] Handling DISTINCT with lists in maps",
        cypher="MATCH (n)\nWITH DISTINCT {name: n.list} AS map\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({list: ['A', 'B']}), ({list: ['A', 'B']})
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections and aggregations are not supported",
        tags=("with", "distinct", "aggregation", "xfail"),
    ),
    Scenario(
        key="with6-1",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[1] Implicit grouping with single expression as grouping key and single aggregation",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS relCount\nRETURN name, relCount",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'", "relCount": 2},
                {"name": "'B'", "relCount": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH aggregations and row projections are not supported",
        tags=("with", "aggregation", "xfail"),
    ),
    Scenario(
        key="with6-2",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[2] Implicit grouping with single relationship variable as grouping key and single aggregation",
        cypher="MATCH ()-[r1]->(:X)\nWITH r1 AS r2, count(*) AS c\nMATCH ()-[r2]->()\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(:X),
                   ()-[:T2]->(:X),
                   ()-[:T3]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH aggregations and relationship projections are not supported",
        tags=("with", "aggregation", "relationship", "xfail"),
    ),
    Scenario(
        key="with6-3",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[3] Implicit grouping with multiple node and relationship variables as grouping key and single aggregation",
        cypher="MATCH (a)-[r1]->(b:X)\nWITH a, r1 AS r2, b, count(*) AS c\nMATCH (a)-[r2]->(b)\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(:X),
                   ()-[:T2]->(:X),
                   ()-[:T3]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH aggregations and relationship projections are not supported",
        tags=("with", "aggregation", "relationship", "xfail"),
    ),
    Scenario(
        key="with6-4",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[4] Implicit grouping with single path variable as grouping key and single aggregation",
        cypher="MATCH p = ()-[*]->()\nWITH count(*) AS count, p AS p\nRETURN nodes(p) AS nodes",
        graph=graph_fixture_from_create(
            """
            CREATE (n1 {num: 1}), (n2 {num: 2}),
                   (n3 {num: 3}), (n4 {num: 4})
            CREATE (n1)-[:T]->(n2),
                   (n3)-[:T]->(n4)
            """
        ),
        expected=Expected(
            rows=[
                {"nodes": "[({num: 1}), ({num: 2})]"},
                {"nodes": "[({num: 3}), ({num: 4})]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable-length patterns, path functions, and aggregations are not supported",
        tags=("with", "aggregation", "path", "variable-length", "xfail"),
    ),
    Scenario(
        key="with6-5",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[5] Handle constants and parameters inside an expression which contains an aggregation expression",
        cypher="MATCH (person)\nWITH $age + avg(person.age) - 1000 AS agg\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"agg": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Parameters, aggregations, and WITH pipelines are not supported",
        tags=("with", "aggregation", "params", "xfail"),
    ),
    Scenario(
        key="with6-6",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[6] Handle projected variables inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age AS age, you\nWITH age, age + count(you.age) AS agg\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and aggregations are not supported",
        tags=("with", "aggregation", "xfail"),
    ),
    Scenario(
        key="with6-7",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[7] Handle projected property accesses inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age AS age, me.age + count(you.age) AS agg\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and aggregations are not supported",
        tags=("with", "aggregation", "xfail"),
    ),
    Scenario(
        key="with6-8",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[8] Fail if not projected variables are used inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age + count(you.age) AS agg\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("with", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="with6-9",
        feature_path="tck/features/clauses/with/With6.feature",
        scenario="[9] Fail if more complex expression, even if projected, are used inside expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age + you.age AS grp, me.age + you.age + count(*) AS agg\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("with", "syntax-error", "aggregation", "xfail"),
    ),
    Scenario(
        key="with7-1",
        feature_path="tck/features/clauses/with/With7.feature",
        scenario="[1] A simple pattern with one bound endpoint",
        cypher="MATCH (a:A)-[r:REL]->(b:B)\nWITH a AS b, b AS tmp, r AS r\nWITH b AS a, r\nLIMIT 1\nMATCH (a)-[r]->(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:REL]", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and row projections are not supported",
        tags=("with", "limit", "xfail"),
    ),
    Scenario(
        key="with7-2",
        feature_path="tck/features/clauses/with/With7.feature",
        scenario="[2] Multiple WITHs using a predicate and aggregation",
        cypher="MATCH (david {name: 'David'})--(otherPerson)-->()\nWITH otherPerson, count(*) AS foaf\nWHERE foaf > 1\nWITH otherPerson\nWHERE otherPerson.name <> 'NotOther'\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'David'}),
                   (b {name: 'Other'}),
                   (c {name: 'NotOther'}),
                   (d {name: 'NotOther2'}),
                   (a)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (a)-[:REL]->(d),
                   (b)-[:REL]->(),
                   (b)-[:REL]->(),
                   (c)-[:REL]->(),
                   (c)-[:REL]->(),
                   (d)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, WHERE filtering, and aggregations are not supported",
        tags=("with", "where", "aggregation", "xfail"),
    ),
    Scenario(
        key="with-where1-1",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[1] Filter node with property predicate on a single variable with multiple bindings",
        cypher="MATCH (a)\nWITH a\nWHERE a.name = 'B'\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'B'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and WHERE filtering are not supported",
        tags=("with", "where", "xfail"),
    ),
    Scenario(
        key="with-where1-2",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[2] Filter node with property predicate on a single variable with multiple distinct bindings",
        cypher="MATCH (a)\nWITH DISTINCT a.name2 AS name\nWHERE a.name2 = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'A'}),
                   ({name2: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections and WHERE filtering are not supported",
        tags=("with", "distinct", "where", "xfail"),
    ),
    Scenario(
        key="with-where1-3",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[3] Filter for an unbound relationship variable",
        cypher="MATCH (a:A), (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE r IS NULL\nRETURN other",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"other": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="with-where1-4",
        feature_path="tck/features/clauses/with-where/WithWhere1.feature",
        scenario="[4] Filter for an unbound node variable",
        cypher="MATCH (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE a IS NULL\nRETURN other",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (:B {id: 2})
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"other": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OPTIONAL MATCH semantics, and null handling are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),
    Scenario(
        key="with-where2-1",
        feature_path="tck/features/clauses/with-where/WithWhere2.feature",
        scenario="[1] Filter nodes with conjunctive two-part property predicate on multi variables with multiple bindings",
        cypher="MATCH (a)--(b)--(c)--(d)--(a), (b)--(d)\nWITH a, c, d\nWHERE a.id = 1\n  AND c.id = 2\nRETURN d",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (c:C {id: 2}), (d:D)
            CREATE (a)-[:T]->(b),
                   (a)-[:T]->(c),
                   (a)-[:T]->(d),
                   (b)-[:T]->(c),
                   (b)-[:T]->(d),
                   (c)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"d": "(:A)"},
                {"d": "(:D)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and multi-variable WHERE predicates are not supported",
        tags=("with", "where", "multi-var", "xfail"),
    ),
    Scenario(
        key="with-where2-2",
        feature_path="tck/features/clauses/with-where/WithWhere2.feature",
        scenario="[2] Filter node with conjunctive multi-part property predicates on multi variables with multiple bindings",
        cypher="MATCH (advertiser)-[:ADV_HAS_PRODUCT]->(out)-[:AP_HAS_VALUE]->(red)<-[:AA_HAS_VALUE]-(a)\nWITH a, advertiser, red, out\nWHERE advertiser.id = $1\n  AND a.id = $2\n  AND red.name = 'red'\n  AND out.name = 'product1'\nRETURN out.name",
        graph=graph_fixture_from_create(
            """
            CREATE (advertiser {name: 'advertiser1', id: 0}),
                   (thing {name: 'Color', id: 1}),
                   (red {name: 'red'}),
                   (p1 {name: 'product1'}),
                   (p2 {name: 'product4'})
            CREATE (advertiser)-[:ADV_HAS_PRODUCT]->(p1),
                   (advertiser)-[:ADV_HAS_PRODUCT]->(p2),
                   (thing)-[:AA_HAS_VALUE]->(red),
                   (p1)-[:AP_HAS_VALUE]->(red),
                   (p2)-[:AP_HAS_VALUE]->(red)
            """
        ),
        expected=Expected(
            rows=[
                {"out.name": "'product1'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, parameter binding, and multi-variable WHERE predicates are not supported",
        tags=("with", "where", "params", "multi-var", "xfail"),
    ),
    Scenario(
        key="with-where3-1",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[1] Join between node identities",
        cypher="MATCH (a), (b)\nWITH a, b\nWHERE a = b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:A)"},
                {"a": "(:B)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable equality joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),
    Scenario(
        key="with-where3-2",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[2] Join between node properties of disconnected nodes",
        cypher="MATCH (a:A), (b:B)\nWITH a, b\nWHERE a.id = b.id\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {id: 1}),
                   (:A {id: 2}),
                   (:B {id: 2}),
                   (:B {id: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {id: 2})", "b": "(:B {id: 2})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable comparison joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),
    Scenario(
        key="with-where3-3",
        feature_path="tck/features/clauses/with-where/WithWhere3.feature",
        scenario="[3] Join between node properties of adjacent nodes",
        cypher="MATCH (n)-[rel]->(x)\nWITH n, x\nWHERE n.animal = x.animal\nRETURN n, x",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A {animal: 'monkey'}),
              (b:B {animal: 'cow'}),
              (c:C {animal: 'monkey'}),
              (d:D {animal: 'cow'}),
              (a)-[:KNOWS]->(b),
              (a)-[:KNOWS]->(c),
              (d)-[:KNOWS]->(b),
              (d)-[:KNOWS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"n": "(:A {animal: 'monkey'})", "x": "(:C {animal: 'monkey'})"},
                {"n": "(:D {animal: 'cow'})", "x": "(:B {animal: 'cow'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable comparison joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),
    Scenario(
        key="with-where4-1",
        feature_path="tck/features/clauses/with-where/WithWhere4.feature",
        scenario="[1] Join nodes on inequality",
        cypher="MATCH (a), (b)\nWITH a, b\nWHERE a <> b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)"},
                {"a": "(:B)", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable inequality joins, and row projections are not supported",
        tags=("with", "join", "inequality", "xfail"),
    ),
    Scenario(
        key="with-where4-2",
        feature_path="tck/features/clauses/with-where/WithWhere4.feature",
        scenario="[2] Join with disjunctive multi-part predicates including patterns",
        cypher="MATCH (a), (b)\nWITH a, b\nWHERE a.id = 0\n  AND (a)-[:T]->(b:TheLabel)\n  OR (a)-[:T*]->(b:MissingLabel)\nRETURN DISTINCT b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:TheLabel {id: 0}), (b:TheLabel {id: 1}), (c:TheLabel {id: 2})
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:TheLabel {id: 1})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OR predicates, pattern predicates, variable-length patterns, and DISTINCT are not supported",
        tags=("with", "or", "pattern-predicate", "variable-length", "distinct", "xfail"),
    ),
    Scenario(
        key="with-where5-1",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[1] Filter out on null",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te'\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, and null semantics are not supported",
        tags=("with", "null", "comparison", "xfail"),
    ),
    Scenario(
        key="with-where5-2",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[2] Filter out on null if the AND'd predicate evaluates to false",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te' AND i:TextNode\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, label predicates, and null semantics are not supported",
        tags=("with", "null", "comparison", "label-predicate", "xfail"),
    ),
    Scenario(
        key="with-where5-3",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[3] Filter out on null if the AND'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te' AND i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, IS NOT NULL, and null semantics are not supported",
        tags=("with", "null", "comparison", "is-not-null", "xfail"),
    ),
    Scenario(
        key="with-where5-4",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[4] Do not filter out on null if the OR'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i)\nWITH i\nWHERE i.var > 'te' OR i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
                {"i": "(:IntNode {var: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OR predicates, comparison predicates, and null semantics are not supported",
        tags=("with", "null", "or", "comparison", "xfail"),
    ),
    Scenario(
        key="with-where6-1",
        feature_path="tck/features/clauses/with-where/WithWhere6.feature",
        scenario="[1] Filter a single aggregate",
        cypher="MATCH (a)-->()\nWITH a, count(*) AS relCount\nWHERE relCount > 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}),
                   (b {name: 'B'})
            CREATE (a)-[:REL]->(),
                   (a)-[:REL]->(),
                   (a)-[:REL]->(),
                   (b)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({name: 'A'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and WHERE filtering are not supported",
        tags=("with", "aggregation", "where", "xfail"),
    ),
    Scenario(
        key="with-where7-1",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[1] WHERE sees a variable bound before but not after WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE a.name2 = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and variable scoping rules are not supported",
        tags=("with", "where", "xfail"),
    ),
    Scenario(
        key="with-where7-2",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[2] WHERE sees a variable bound after but not before WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE name = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and variable scoping rules are not supported",
        tags=("with", "where", "xfail"),
    ),
    Scenario(
        key="with-where7-3",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[3] WHERE sees both, variable bound before but not after WITH and variable bound after but not before WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE name = 'B' OR a.name2 = 'C'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable scoping rules, and OR predicates are not supported",
        tags=("with", "where", "or", "xfail"),
    ),
    Scenario(
        key="with-skip-limit1-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit1.feature",
        scenario="[1] Handle dependencies across WITH with SKIP",
        cypher="MATCH (a)\nWITH a.name AS property, a.num AS idToUse\n  ORDER BY property\n  SKIP 1\nMATCH (b)\nWHERE b.id = idToUse\nRETURN DISTINCT b",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A', num: 0, id: 0}),
                   ({name: 'B', num: a.id, id: 1}),
                   ({name: 'C', num: 0, id: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "({name: 'A', num: 0, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, SKIP, and DISTINCT are not supported",
        tags=("with", "skip", "orderby", "distinct", "xfail"),
    ),
    Scenario(
        key="with-skip-limit1-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit1.feature",
        scenario="[2] Ordering and skipping on aggregate",
        cypher="MATCH ()-[r1]->(x)\nWITH x, sum(r1.num) AS c\n  ORDER BY c SKIP 1\nRETURN x, c",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {num: 3}]->(x:X),
                   ()-[:T2 {num: 2}]->(x),
                   ()-[:T3 {num: 1}]->(:Y)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:X)", "c": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, ORDER BY, and SKIP are not supported",
        tags=("with", "skip", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="with-skip-limit2-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[1] ORDER BY and LIMIT can be used",
        cypher="MATCH (a:A)\nWITH a\nORDER BY a.name\nLIMIT 1\nMATCH (a)-->(b)\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (), (), (),
                   (a)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, and LIMIT are not supported",
        tags=("with", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="with-skip-limit2-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[2] Handle dependencies across WITH with LIMIT",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\n  LIMIT 1\nMATCH (b)\nWHERE b.id = property\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End {num: 42, id: 0}),
                   (:End {num: 3}),
                   (:Begin {num: a.id})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and joins are not supported",
        tags=("with", "limit", "join", "xfail"),
    ),
    Scenario(
        key="with-skip-limit2-3",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[3] Connected components succeeding WITH with LIMIT",
        cypher="MATCH (n:A)\nWITH n\nLIMIT 1\nMATCH (m:B), (n)-->(x:X)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:X)
            CREATE (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"m": "(:B)", "n": "(:A)", "x": "(:X)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and RETURN * projections are not supported",
        tags=("with", "limit", "return-star", "xfail"),
    ),
    Scenario(
        key="with-skip-limit2-4",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit2.feature",
        scenario="[4] Ordering and limiting on aggregate",
        cypher="MATCH ()-[r1]->(x)\nWITH x, sum(r1.num) AS c\n  ORDER BY c LIMIT 1\nRETURN x, c",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1 {num: 3}]->(x:X),
                   ()-[:T2 {num: 2}]->(x),
                   ()-[:T3 {num: 1}]->(:Y)
            """
        ),
        expected=Expected(
            rows=[
                {"x": "(:Y)", "c": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, ORDER BY, and LIMIT are not supported",
        tags=("with", "limit", "orderby", "aggregation", "xfail"),
    ),
    Scenario(
        key="with-skip-limit3-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[1] Get rows in the middle",
        cypher="MATCH (n)\nWITH n\nORDER BY n.name ASC\nSKIP 2\nLIMIT 2\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("with", "skip", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="with-skip-limit3-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[2] Get rows in the middle by param",
        cypher="MATCH (n)\nWITH n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, SKIP/LIMIT, ORDER BY, and parameter binding are not supported",
        tags=("with", "skip", "limit", "orderby", "params", "xfail"),
    ),
    Scenario(
        key="with-skip-limit3-3",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[3] Limiting amount of rows when there are fewer left than the LIMIT argument",
        cypher="MATCH (a)\nWITH a.count AS count\n  ORDER BY a.count\n  SKIP 10\n  LIMIT 10\nRETURN count",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "count": i} for i in range(16)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 10},
                {"count": 11},
                {"count": 12},
                {"count": 13},
                {"count": 14},
                {"count": 15},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("with", "skip", "limit", "orderby", "xfail"),
    ),
    Scenario(
        key="with-orderby1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[1] Sort booleans in ascending order",
        cypher="UNWIND [true, false] AS bools\nWITH bools\n  ORDER BY bools\n  LIMIT 1\nRETURN bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[2] Sort booleans in descending order",
        cypher="UNWIND [true, false] AS bools\nWITH bools\n  ORDER BY bools DESC\n  LIMIT 1\nRETURN bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[3] Sort integers in ascending order",
        cypher="UNWIND [1, 3, 2] AS ints\nWITH ints\n  ORDER BY ints\n  LIMIT 2\nRETURN ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 1},
                {"ints": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[4] Sort integers in descending order",
        cypher="UNWIND [1, 3, 2] AS ints\nWITH ints\n  ORDER BY ints DESC\n  LIMIT 2\nRETURN ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 3},
                {"ints": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[5] Sort floats in ascending order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nWITH floats\n  ORDER BY floats\n  LIMIT 2\nRETURN floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 1.3},
                {"floats": 1.5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[6] Sort floats in descending order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nWITH floats\n  ORDER BY floats DESC\n  LIMIT 2\nRETURN floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 999.99},
                {"floats": 1.5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[7] Sort strings in ascending order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nWITH strings\n  ORDER BY strings\n  LIMIT 2\nRETURN strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "''"},
                {"strings": "' '"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[8] Sort strings in descending order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nWITH strings\n  ORDER BY strings DESC\n  LIMIT 2\nRETURN strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "'one'"},
                {"strings": "'.*'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[9] Sort lists in ascending order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nWITH lists\n  ORDER BY lists\n  LIMIT 4\nRETURN lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[]"},
                {"lists": "['a']"},
                {"lists": "['a', 1]"},
                {"lists": "[1]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[10] Sort lists in descending order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nWITH lists\n  ORDER BY lists DESC\n  LIMIT 4\nRETURN lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[null, 2]"},
                {"lists": "[null, 1]"},
                {"lists": "[1, null]"},
                {"lists": "[1, 'a']"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[11] Sort dates in ascending order",
        cypher="UNWIND [date({year: 1910, month: 5, day: 6}),\n        date({year: 1980, month: 12, day: 24}),\n        date({year: 1984, month: 10, day: 12}),\n        date({year: 1985, month: 5, day: 6}),\n        date({year: 1980, month: 10, day: 24}),\n        date({year: 1984, month: 10, day: 11})] AS dates\nWITH dates\n  ORDER BY dates\n  LIMIT 2\nRETURN dates",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"dates": "'1910-05-06'"},
                {"dates": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[12] Sort dates in descending order",
        cypher="UNWIND [date({year: 1910, month: 5, day: 6}),\n        date({year: 1980, month: 12, day: 24}),\n        date({year: 1984, month: 10, day: 12}),\n        date({year: 1985, month: 5, day: 6}),\n        date({year: 1980, month: 10, day: 24}),\n        date({year: 1984, month: 10, day: 11})] AS dates\nWITH dates\n  ORDER BY dates DESC\n  LIMIT 2\nRETURN dates",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"dates": "'1985-05-06'"},
                {"dates": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[13] Sort local times in ascending order",
        cypher="UNWIND [localtime({hour: 10, minute: 35}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n        localtime({hour: 12, minute: 35, second: 13}),\n        localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\nWITH localtimes\n  ORDER BY localtimes\n  LIMIT 3\nRETURN localtimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localtimes": "'10:35'"},
                {"localtimes": "'12:30:14.645876123'"},
                {"localtimes": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[14] Sort local times in descending order",
        cypher="UNWIND [localtime({hour: 10, minute: 35}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n        localtime({hour: 12, minute: 35, second: 13}),\n        localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\nWITH localtimes\n  ORDER BY localtimes DESC\n  LIMIT 3\nRETURN localtimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localtimes": "'12:35:13'"},
                {"localtimes": "'12:31:14.645876124'"},
                {"localtimes": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[15] Sort times in ascending order",
        cypher="UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n        time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n        time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\nWITH times\n  ORDER BY times\n  LIMIT 3\nRETURN times",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"times": "'12:35:15+05:00'"},
                {"times": "'12:30:14.645876123+01:01'"},
                {"times": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[16] Sort times in descending order",
        cypher="UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n        time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n        time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\nWITH times\n  ORDER BY times DESC\n  LIMIT 3\nRETURN times",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"times": "'10:35-08:00'"},
                {"times": "'12:31:14.645876124+01:00'"},
                {"times": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[17] Sort local date times in ascending order",
        cypher="UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n        localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n        localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n        localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\nWITH localdatetimes\n  ORDER BY localdatetimes\n  LIMIT 3\nRETURN localdatetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localdatetimes": "'0001-01-01T01:01:01.000000001'"},
                {"localdatetimes": "'1980-12-11T12:31:14'"},
                {"localdatetimes": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[18] Sort local date times in descending order",
        cypher="UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n        localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n        localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n        localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\nWITH localdatetimes\n  ORDER BY localdatetimes DESC\n  LIMIT 3\nRETURN localdatetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localdatetimes": "'9999-09-09T09:59:59.999999999'"},
                {"localdatetimes": "'1984-10-11T12:31:14.645876123'"},
                {"localdatetimes": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[19] Sort date times in ascending order",
        cypher="UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n        datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n        datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n        datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n        datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\nWITH datetimes\n  ORDER BY datetimes\n  LIMIT 3\nRETURN datetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"datetimes": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"datetimes": "'1980-12-11T12:31:14-11:59'"},
                {"datetimes": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[20] Sort date times in descending order",
        cypher="UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n        datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n        datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n        datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n        datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\nWITH datetimes\n  ORDER BY datetimes DESC\n  LIMIT 3\nRETURN datetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"datetimes": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"datetimes": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"datetimes": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[21] Sort distinct types in ascending order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nWITH types\n  ORDER BY types\n  LIMIT 5\nRETURN types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "{a: 'map'}"},
                {"types": "(:N)"},
                {"types": "[:REL]"},
                {"types": "['list']"},
                {"types": "<(:N)-[:REL]->()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[22] Sort distinct types in descending order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nWITH types\n  ORDER BY types DESC\n  LIMIT 5\nRETURN types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "null"},
                {"types": "NaN"},
                {"types": 1.5},
                {"types": "false"},
                {"types": "'text'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-23-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-23-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool ASC)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool ASC\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-23-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool ASCENDING\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-24-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[24] Sort by a boolean variable projected from a node property in descending order (sort=bool DESC)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool DESC\n  LIMIT 2\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true})", "bool": "true"},
                {"a": "(:D {bool: true})", "bool": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-24-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[24] Sort by a boolean variable projected from a node property in descending order (sort=bool DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool DESCENDING\n  LIMIT 2\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true})", "bool": "true"},
                {"a": "(:D {bool: true})", "bool": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-25-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-25-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num ASC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-25-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-26-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[26] Sort by an integer variable projected from a node property in descending order (sort=num DESC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054})", "num": 7054},
                {"a": "(:C {num: 30})", "num": 30},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-26-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[26] Sort by an integer variable projected from a node property in descending order (sort=num DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054})", "num": 7054},
                {"a": "(:C {num: 30})", "num": 30},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-27-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-27-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num ASC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-27-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-28-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[28] Sort by a float variable projected from a node property in descending order (sort=num DESC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008})", "num": 7054.008},
                {"a": "(:B {num: 30.94857})", "num": 30.94857},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-28-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[28] Sort by a float variable projected from a node property in descending order (sort=num DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008})", "num": 7054.008},
                {"a": "(:B {num: 30.94857})", "num": 30.94857},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-29-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-29-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name ASC)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name ASC\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-29-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name ASCENDING\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-30-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[30] Sort by a string variable projected from a node property in descending order (sort=name DESC)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name DESC\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {name: 'sit'})", "name": "'sit'"},
                {"a": "(:A {name: 'lorem'})", "name": "'lorem'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-30-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[30] Sort by a string variable projected from a node property in descending order (sort=name DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name DESCENDING\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {name: 'sit'})", "name": "'sit'"},
                {"a": "(:A {name: 'lorem'})", "name": "'lorem'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby1-31-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-31-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list ASC)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list ASC\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-31-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list ASCENDING\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-32-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[32] Sort by a list variable projected from a node property in descending order (sort=list DESC)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list DESC\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0]})", "list": "[300, 0]"},
                {"a": "(:E {list: [2, -2, 100]})", "list": "[2, -2, 100]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-32-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[32] Sort by a list variable projected from a node property in descending order (sort=list DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list DESCENDING\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0]})", "list": "[300, 0]"},
                {"a": "(:E {list: [2, -2, 100]})", "list": "[2, -2, 100]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-33-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date\n  LIMIT 2\nRETURN a, date",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-33-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date ASC)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date ASC\n  LIMIT 2\nRETURN a, date",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-33-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date ASCENDING\n  LIMIT 2\nRETURN a, date",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-34-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[34] Sort by a date variable projected from a node property in descending order (sort=date DESC)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date DESC\n  LIMIT 2\nRETURN a, date",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})", "date": "'1985-05-06'"},
                {"a": "(:C {date: '1984-10-12'})", "date": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-34-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[34] Sort by a date variable projected from a node property in descending order (sort=date DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date DESCENDING\n  LIMIT 2\nRETURN a, date",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})", "date": "'1985-05-06'"},
                {"a": "(:C {date: '1984-10-12'})", "date": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-35-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-35-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time ASC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASC\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-35-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASCENDING\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-36-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[36] Sort by a local time variable projected from a node property in descending order (sort=time DESC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESC\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})", "time": "'12:31:15'"},
                {"a": "(:C {time: '12:31:14.645876124'})", "time": "'12:31:14.645876124'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-36-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[36] Sort by a local time variable projected from a node property in descending order (sort=time DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESCENDING\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})", "time": "'12:31:15'"},
                {"a": "(:C {time: '12:31:14.645876124'})", "time": "'12:31:14.645876124'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-37-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-37-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time ASC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASC\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-37-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASCENDING\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-38-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[38] Sort by a time variable projected from a node property in descending order (sort=time DESC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESC\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})", "time": "'10:35-08:00'"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})", "time": "'12:31:14.645876124+01:00'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-38-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[38] Sort by a time variable projected from a node property in descending order (sort=time DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESCENDING\n  LIMIT 3\nRETURN a, time",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})", "time": "'10:35-08:00'"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})", "time": "'12:31:14.645876124+01:00'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-39-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-39-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime ASC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASC\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-39-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASCENDING\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-40-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[40] Sort by a local date time variable projected from a node property in descending order (sort=datetime DESC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESC\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})", "datetime": "'9999-09-09T09:59:59.999999999'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})", "datetime": "'1984-10-11T12:31:14.645876123'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-40-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[40] Sort by a local date time variable projected from a node property in descending order (sort=datetime DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESCENDING\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})", "datetime": "'9999-09-09T09:59:59.999999999'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})", "datetime": "'1984-10-11T12:31:14.645876123'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-41-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-41-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime ASC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASC\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-41-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASCENDING\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-42-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[42] Sort by a date time variable projected from a node property in descending order (sort=datetime DESC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESC\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})", "datetime": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})", "datetime": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-42-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[42] Sort by a date time variable projected from a node property in descending order (sort=datetime DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESCENDING\n  LIMIT 3\nRETURN a, datetime",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})", "datetime": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})", "datetime": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-43-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[43] Sort by a variable that is only partially orderable on a non-distinct binding table (dir=ASC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH x\n  ORDER BY x ASC\n  LIMIT 2\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 0},
                {"x": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-43-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[43] Sort by a variable that is only partially orderable on a non-distinct binding table (dir=DESC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH x\n  ORDER BY x DESC\n  LIMIT 2\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby1-44-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[44] Sort by a variable that is only partially orderable on a non-distinct binding table, but made distinct (dir=ASC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH DISTINCT x\n  ORDER BY x ASC\n  LIMIT 1\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and DISTINCT are not supported",
        tags=("with", "orderby", "unwind", "limit", "distinct", "xfail"),
    ),
    Scenario(
        key="with-orderby1-44-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[44] Sort by a variable that is only partially orderable on a non-distinct binding table, but made distinct (dir=DESC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH DISTINCT x\n  ORDER BY x DESC\n  LIMIT 1\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and DISTINCT are not supported",
        tags=("with", "orderby", "unwind", "limit", "distinct", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=booleans)",
        cypher="WITH [true, false] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=integers)",
        cypher="WITH [351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=floats)",
        cypher="WITH [351.5, -3974856.01, -3.203957, 123.0002, 123.0001, 123.00013, 123.00011, 0.0100000, 0.0999999, 0.00000001, 3.0, 209345.87, -10.654] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=string)",
        cypher="WITH ['Sort', 'order', ' ', 'should', 'be', '', 'consistent', 'with', 'comparisons', ', ', 'where', 'comparisons are', 'defined', '!'] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=lists)",
        cypher="WITH [[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=dates)",
        cypher="WITH [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 12, day: 24}), date({year: 1984, month: 10, day: 12}), date({year: 1985, month: 5, day: 6}), date({year: 1980, month: 10, day: 24}), date({year: 1984, month: 10, day: 11})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=localtimes)",
        cypher="WITH [localtime({hour: 10, minute: 35}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}), localtime({hour: 12, minute: 35, second: 13}), localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 15})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=times)",
        cypher="WITH [time({hour: 10, minute: 35, timezone: '-08:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=localdatetimes)",
        cypher="WITH [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}), localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}), localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}), localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-45-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=datetimes)",
        cypher="WITH [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c ASC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c ASC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c ASCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c ASCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c DESC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c DESC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c DESCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c DESCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d ASC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d ASC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d ASCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d ASCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d DESC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d DESC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby1-46-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d DESCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d DESCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[1] Sort by a projected expression",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\n  ORDER BY a.num + a.num2\n  LIMIT 3\nRETURN a, sum",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "sum": 5},
                {"a": "(:A {num: 3, num2: 3})", "sum": 6},
                {"a": "(:A {num: 5, num2: 2})", "sum": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[2] Sort by an alias of a projected expression",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\n  ORDER BY sum\n  LIMIT 3\nRETURN a, sum",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "sum": 5},
                {"a": "(:A {num: 3, num2: 3})", "sum": 6},
                {"a": "(:A {num: 5, num2: 2})", "sum": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[3] Sort by two projected expressions with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY a.num2 % 3, a.num + a.num2\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[4] Sort by one projected expression and one alias of a projected expression with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY a.num2 % 3, sum\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[5] Sort by one alias of a projected expression and one projected expression with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY mod, a.num + a.num2\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[6] Sort by aliases of two projected expressions with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY mod, sum\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[7] Sort by an alias of a projected expression where the alias shadows an existing variable",
        cypher="MATCH (a:A)\nWITH a, a.num2 % 3 AS x\nWITH a, a.num + a.num2 AS x\n  ORDER BY x\n  LIMIT 3\nRETURN a, x",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "x": 5},
                {"a": "(:A {num: 3, num2: 3})", "x": 6},
                {"a": "(:A {num: 5, num2: 2})", "x": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[8] Sort by non-projected existing variable",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\nWITH a, a.num2 % 3 AS mod\n  ORDER BY sum\n  LIMIT 3\nRETURN a, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "mod": 1},
                {"a": "(:A {num: 3, num2: 3})", "mod": 0},
                {"a": "(:A {num: 5, num2: 2})", "mod": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby4-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[9] Sort by an alias of a projected expression containing the variable shadowed by the alias",
        cypher="MATCH (a:A)\nWITH a.num2 AS x\nWITH x % 3 AS x\n  ORDER BY x\n  LIMIT 3\nRETURN x",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"x": 0},
                {"x": 0},
                {"x": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2)\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) ASC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) ASCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-2-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[2] Sort by a boolean expression in descending order (sort=NOT (a.bool AND a.bool2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false, bool2: false})"},
                {"a": "(:C {bool: false, bool2: true})"},
                {"a": "(:E {bool: true, bool2: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-2-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[2] Sort by a boolean expression in descending order (sort=NOT (a.bool AND a.bool2) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false, bool2: false})"},
                {"a": "(:C {bool: false, bool2: true})"},
                {"a": "(:E {bool: true, bool2: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-3-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-3-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-3-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[4] Sort by an integer expression in descending order (sort=(a.num2 + (a.num * 2)) * -1 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11, num2: 2})"},
                {"a": "(:B {num: 5, num2: 4})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[4] Sort by an integer expression in descending order (sort=(a.num2 + (a.num * 2)) * -1 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11, num2: 2})"},
                {"a": "(:B {num: 5, num2: 4})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-5-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-5-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-5-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-6-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[6] Sort by a float expression in descending order (sort=(a.num + a.num2 * 2) * -1.01 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943, num2: -8.5007})"},
                {"a": "(:A {num: 5.025648, num2: 1.96357})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-6-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[6] Sort by a float expression in descending order (sort=(a.num + a.num2 * 2) * -1.01 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943, num2: -8.5007})"},
                {"a": "(:A {num: 5.025648, num2: 1.96357})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-7-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-7-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-7-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-8-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[8] Sort by a string expression in descending order (sort=a.title + ' ' + a.name DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {name: 'dolor', title: 'prof.'})"},
                {"a": "(:E {name: 'amet', title: 'prof.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-8-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[8] Sort by a string expression in descending order (sort=a.title + ' ' + a.name DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {name: 'dolor', title: 'prof.'})"},
                {"a": "(:E {name: 'amet', title: 'prof.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-9-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby2-9-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby2-9-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby2-10-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[10] Sort by a list expression in descending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {list: [2, -2, 100], list2: [5, -2]})"},
                {"a": "(:D {list: [1, -20], list2: [4, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby2-10-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[10] Sort by a list expression in descending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {list: [2, -2, 100], list2: [5, -2]})"},
                {"a": "(:D {list: [1, -20], list2: [4, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),
    Scenario(
        key="with-orderby2-11-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2})\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-11-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) ASC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-11-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) ASCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-12-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[12] Sort by a date expression in descending order (sort=a.date + duration({months: 1, days: 2}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) DESC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})"},
                {"a": "(:C {date: '1984-10-12'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-12-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[12] Sort by a date expression in descending order (sort=a.date + duration({months: 1, days: 2}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) DESCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})"},
                {"a": "(:C {date: '1984-10-12'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-13-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-13-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-13-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-14-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[14] Sort by a local time expression in descending order (sort=a.time + duration({minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})"},
                {"a": "(:C {time: '12:31:14.645876124'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-14-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[14] Sort by a local time expression in descending order (sort=a.time + duration({minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})"},
                {"a": "(:C {time: '12:31:14.645876124'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-15-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-15-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-15-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-16-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[16] Sort by a time expression in descending order (sort=a.time + duration({minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-16-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[16] Sort by a time expression in descending order (sort=a.time + duration({minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-17-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-17-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-17-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-18-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[18] Sort by a local date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-18-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[18] Sort by a local date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-19-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-19-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-19-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-20-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[20] Sort by a date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-20-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[20] Sort by a date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),
    Scenario(
        key="with-orderby2-21-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[21] Sort by an expression that is only partially orderable on a non-distinct binding table (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name\n  ORDER BY a.name + 'C' ASC\n  LIMIT 2\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
                {"name": "'A'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-21-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[21] Sort by an expression that is only partially orderable on a non-distinct binding table (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name\n  ORDER BY a.name + 'C' DESC\n  LIMIT 2\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'"},
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-22-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[22] Sort by an expression that is only partially orderable on a non-distinct binding table, but used as a grouping key (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name ASC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and aggregation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "xfail"),
    ),
    Scenario(
        key="with-orderby2-22-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[22] Sort by an expression that is only partially orderable on a non-distinct binding table, but used as a grouping key (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name DESC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and aggregation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "xfail"),
    ),
    Scenario(
        key="with-orderby2-23-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[23] Sort by an expression that is only partially orderable on a non-distinct binding table, but used in parts as a grouping key (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name + 'C' ASC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, aggregation, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-23-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[23] Sort by an expression that is only partially orderable on a non-distinct binding table, but used in parts as a grouping key (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name + 'C' DESC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, aggregation, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby2-24-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[24] Sort by an expression that is only partially orderable on a non-distinct binding table, but made distinct (dir=ASC)",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\n  ORDER BY a.name ASC\n  LIMIT 1\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "limit", "distinct", "xfail"),
    ),
    Scenario(
        key="with-orderby2-24-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[24] Sort by an expression that is only partially orderable on a non-distinct binding table, but made distinct (dir=DESC)",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\n  ORDER BY a.name DESC\n  LIMIT 1\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "limit", "distinct", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(n))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(n)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(n.num1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(n.num1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(1 + n.num1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(1 + n.num1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASCENDING)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASCENDING\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESCENDING)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESCENDING\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2), n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2), n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASC, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASC, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASCENDING, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASCENDING, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESC, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESC, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESCENDING, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESCENDING, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name ASC, max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name ASC, max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name ASC, max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name ASC, max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name DESC, max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name DESC, max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name DESC, max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name DESC, max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, max(n.num2), n.name2)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, max(n.num2), n.name2\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, n.name2, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, n.name2, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.num1, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.num1, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, max(n.num2), n.num1)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, max(n.num2), n.num1\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby2-25-25",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, count(n.num1), max(n.num2), n.num1)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, count(n.num1), max(n.num2), n.num1\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-1-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASC, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASC, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASCENDING, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-2-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASCENDING, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-3-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESC, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESC, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-4-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESCENDING, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-4-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESCENDING, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-5-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 ASC, a.num, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 ASC, a.num, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 2, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-5-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 ASC, a.num, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 ASC, a.num, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 2, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-5-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 DESC, a.num, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 DESC, a.num, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-5-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 DESC, a.num, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 DESC, a.num, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-6-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a ASC, a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a ASC, a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a + 2 ASC, a + 2 DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a + 2 ASC, a + 2 DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a * a ASC, a * a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a * a ASC, a * a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a ASC, -1 * a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a ASC, -1 * a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=-1 * a DESC, a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY -1 * a DESC, a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a DESC, a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a DESC, a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a + 2 DESC, a + 2 ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a + 2 DESC, a + 2 ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a * a DESC, a * a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a * a DESC, a * a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a DESC, -1 * a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a DESC, -1 * a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-7-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=-1 * a ASC, a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY -1 * a ASC, a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c ASC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c ASC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c DESC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c DESC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c ASC, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c ASC, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c DESC, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c DESC, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c ASC, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c ASC, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c DESC, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c DESC, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=b, c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, b, c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, b, c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, d, b, b, d, c, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, d, b, b, d, c, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e ASC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e ASC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e DESC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e DESC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e ASC, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e ASC, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e DESC, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e DESC, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e ASC, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e ASC, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e DESC, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e DESC, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=b, e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-25",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, b, e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, b, e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-26",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, f, b, b, f, e, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, f, b, b, f, e, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-27",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=a, c, e)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, e\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-28",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=a, c, e, b)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, e, b\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-29",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=b, c, a, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, c, a, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="with-orderby3-8-30",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=d, f, b, b, f, c, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY d, f, b, b, f, c, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
    Scenario(
        key="unwind1-1",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[1] Unwinding a list",
        cypher="UNWIND [1, 2, 3] AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),
    Scenario(
        key="unwind1-2",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[2] Unwinding a range",
        cypher="UNWIND range(1, 3) AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and range() expressions are not supported",
        tags=("unwind", "expression", "xfail"),
    ),
    Scenario(
        key="unwind1-3",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[3] Unwinding a concatenation of lists",
        cypher="WITH [1, 2, 3] AS first, [4, 5, 6] AS second\nUNWIND (first + second) AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
                {"x": 3},
                {"x": 4},
                {"x": 5},
                {"x": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and list concatenation are not supported",
        tags=("unwind", "with", "expression", "xfail"),
    ),
    Scenario(
        key="unwind1-4",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[4] Unwinding a collected unwound expression",
        cypher="UNWIND RANGE(1, 2) AS row\nWITH collect(row) AS rows\nUNWIND rows AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, aggregation, and range() expressions are not supported",
        tags=("unwind", "aggregation", "expression", "xfail"),
    ),
    Scenario(
        key="unwind1-5",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[5] Unwinding a collected expression",
        cypher="MATCH (row)\nWITH collect(row) AS rows\nUNWIND rows AS node\nRETURN node.id",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 1}), ({id: 2})
            """
        ),
        expected=Expected(
            rows=[
                {"node.id": 1},
                {"node.id": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, aggregation, and projection evaluation are not supported",
        tags=("unwind", "aggregation", "projection", "xfail"),
    ),
    Scenario(
        key="unwind1-6",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[6] Creating nodes from an unwound parameter list",
        cypher="UNWIND $events AS event\nMATCH (y:Year {year: event.year})\nMERGE (e:Event {id: event.id})\nMERGE (y)<-[:IN]-(e)\nRETURN e.id AS x\nORDER BY x",
        graph=graph_fixture_from_create(
            """
            CREATE (:Year {year: 2016})
            """
        ),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, parameters, MERGE, ORDER BY, and side-effect validation are not supported",
        tags=("unwind", "params", "merge", "orderby", "xfail"),
    ),
    Scenario(
        key="unwind1-7",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[7] Double unwinding a list of lists",
        cypher="WITH [[1, 2, 3], [4, 5, 6]] AS lol\nUNWIND lol AS x\nUNWIND x AS y\nRETURN y",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"y": 1},
                {"y": 2},
                {"y": 3},
                {"y": 4},
                {"y": 5},
                {"y": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and UNWIND are not supported",
        tags=("unwind", "with", "xfail"),
    ),
    Scenario(
        key="unwind1-8",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[8] Unwinding the empty list",
        cypher="UNWIND [] AS empty\nRETURN empty",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),
    Scenario(
        key="unwind1-9",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[9] Unwinding null",
        cypher="UNWIND null AS nil\nRETURN nil",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),
    Scenario(
        key="unwind1-10",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[10] Unwinding list with duplicates",
        cypher="UNWIND [1, 1, 2, 2, 3, 3, 4, 4, 5, 5] AS duplicate\nRETURN duplicate",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"duplicate": 1},
                {"duplicate": 1},
                {"duplicate": 2},
                {"duplicate": 2},
                {"duplicate": 3},
                {"duplicate": 3},
                {"duplicate": 4},
                {"duplicate": 4},
                {"duplicate": 5},
                {"duplicate": 5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND is not supported",
        tags=("unwind", "xfail"),
    ),
    Scenario(
        key="unwind1-11",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[11] Unwind does not prune context",
        cypher="WITH [1, 2, 3] AS list\nUNWIND list AS x\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"list": "[1, 2, 3]", "x": 1},
                {"list": "[1, 2, 3]", "x": 2},
                {"list": "[1, 2, 3]", "x": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, and projection expansion are not supported",
        tags=("unwind", "with", "projection", "xfail"),
    ),
    Scenario(
        key="unwind1-12",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[12] Unwind does not remove variables from scope",
        cypher="MATCH (a:S)-[:X]->(b1)\nWITH a, collect(b1) AS bees\nUNWIND bees AS b2\nMATCH (a)-[:Y]->(b2)\nRETURN a, b2",
        graph=graph_fixture_from_create(
            """
            CREATE (s:S),
              (n),
              (e:E),
              (s)-[:X]->(e),
              (s)-[:Y]->(e),
              (n)-[:Y]->(e)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:S)", "b2": "(:E)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, aggregation, and multi-step MATCH are not supported",
        tags=("unwind", "with", "aggregation", "match", "xfail"),
    ),
    Scenario(
        key="unwind1-13",
        feature_path="tck/features/clauses/unwind/Unwind1.feature",
        scenario="[13] Multiple unwinds after each other",
        cypher="WITH [1, 2] AS xs, [3, 4] AS ys, [5, 6] AS zs\nUNWIND xs AS x\nUNWIND ys AS y\nUNWIND zs AS z\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 1, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 3, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 5, "zs": "[5, 6]"},
                {"x": 2, "xs": "[1, 2]", "y": 4, "ys": "[3, 4]", "z": 6, "zs": "[5, 6]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and UNWIND are not supported",
        tags=("unwind", "with", "xfail"),
    ),
    Scenario(
        key="call1-1",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[1] Standalone call to procedure that takes no arguments and yields no results",
        cypher="CALL test.doNothing()",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-2",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[2] Standalone call to procedure that takes no arguments and yields no results, called with implicit arguments",
        cypher="CALL test.doNothing",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-3",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[3] In-query call to procedure that takes no arguments and yields no results",
        cypher="MATCH (n)\nCALL test.doNothing()\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-4",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[4] In-query call to procedure that takes no arguments and yields no results and consumes no rows",
        cypher="MATCH (n)\nCALL test.doNothing()\nRETURN n.name AS `name`",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'a'})
            CREATE (:B {name: 'b'})
            CREATE (:C {name: 'c'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'a'"},
                {"name": "'b'"},
                {"name": "'c'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-5",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[5] Standalone call to STRING procedure that takes no arguments",
        cypher="CALL test.labels()",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"label": "'A'"},
                {"label": "'B'"},
                {"label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-6",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[6] In-query call to STRING procedure that takes no arguments",
        cypher="CALL test.labels() YIELD label\nRETURN label",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"label": "'A'"},
                {"label": "'B'"},
                {"label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call1-7",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[7] Standalone call to procedure should fail if explicit argument is missing",
        cypher="CALL test.my.proc('Dobby')",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-8",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[8] In-query call to procedure should fail if explicit argument is missing",
        cypher="CALL test.my.proc('Dobby') YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-9",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[9] Standalone call to procedure should fail if too many explicit argument are given",
        cypher="CALL test.my.proc(1, 2, 3, 4)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-10",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[10] In-query call to procedure should fail if too many explicit argument are given",
        cypher="CALL test.my.proc(1, 2, 3, 4) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-11",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[11] Standalone call to procedure should fail if implicit argument is missing",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedures and parameter binding are not supported",
        tags=("call", "params", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-12",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[12] In-query call to procedure that has outputs fails if no outputs are yielded",
        cypher="CALL test.my.proc(1)\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call1-13",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[13] Standalone call to unknown procedure should fail",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure registry is not supported",
        tags=("call", "procedure-error", "xfail"),
    ),
    Scenario(
        key="call2-1",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[1] In-query call to procedure with explicit arguments",
        cypher="CALL test.my.proc('Stefan', 1) YIELD city, country_code\nRETURN city, country_code",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call2-2",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[2] Standalone call to procedure with explicit arguments",
        cypher="CALL test.my.proc('Stefan', 1)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call2-3",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[3] Standalone call to procedure with implicit arguments",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and parameter binding are not supported",
        tags=("call", "params", "xfail"),
    ),
    Scenario(
        key="call2-4",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[4] In-query call to procedure that takes arguments fails when trying to pass them implicitly",
        cypher="CALL test.my.proc YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call2-5",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[5] Standalone call to procedure should fail if input type is wrong",
        cypher="CALL test.my.proc(true)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call2-6",
        feature_path="tck/features/clauses/call/Call2.feature",
        scenario="[6] In-query call to procedure should fail if input type is wrong",
        cypher="CALL test.my.proc(true) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call3-1",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[1] Standalone call to procedure with argument of type NUMBER accepts value of type INTEGER",
        cypher="CALL test.my.proc(42)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'wisdom'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call3-2",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[2] In-query call to procedure with argument of type NUMBER accepts value of type INTEGER",
        cypher="CALL test.my.proc(42) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'wisdom'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call3-3",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[3] Standalone call to procedure with argument of type NUMBER accepts value of type FLOAT",
        cypher="CALL test.my.proc(42.3)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'about right'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call3-4",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[4] In-query call to procedure with argument of type NUMBER accepts value of type FLOAT",
        cypher="CALL test.my.proc(42.3) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'about right'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call3-5",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[5] Standalone call to procedure with argument of type FLOAT accepts value of type INTEGER",
        cypher="CALL test.my.proc(42)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'close enough'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call3-6",
        feature_path="tck/features/clauses/call/Call3.feature",
        scenario="[6] In-query call to procedure with argument of type FLOAT accepts value of type INTEGER",
        cypher="CALL test.my.proc(42) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'close enough'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call4-1",
        feature_path="tck/features/clauses/call/Call4.feature",
        scenario="[1] Standalone call to procedure with null argument",
        cypher="CALL test.my.proc(null)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call4-2",
        feature_path="tck/features/clauses/call/Call4.feature",
        scenario="[2] In-query call to procedure with null argument",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[1] Explicit procedure result projection",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[2] Explicit procedure result projection with RETURN *",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-3-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[3] The order of yield items is irrelevant (example=1, yield=a, b)",
        cypher="CALL test.my.proc(null) YIELD a, b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-3-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[3] The order of yield items is irrelevant (example=2, yield=b, a)",
        cypher="CALL test.my.proc(null) YIELD b, a\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=1, rename=a AS c, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS d\nRETURN c, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=2, rename=a AS b, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS b, b AS d\nRETURN b, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"b": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-3",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=3, rename=a AS c, b AS a)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS a\nRETURN c, a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "a": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-4",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=4, rename=a AS b, b AS a)",
        cypher="CALL test.my.proc(null) YIELD a AS b, b AS a\nRETURN b, a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"b": 1, "a": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-5",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=5, rename=a AS c, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS b\nRETURN c, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-6",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=6, rename=a AS c, b)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b\nRETURN c, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-7",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=7, rename=a AS a, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b AS d\nRETURN a, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-8",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=8, rename=a, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a, b AS d\nRETURN a, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-9",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=9, rename=a AS a, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b AS b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-10",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=10, rename=a AS a, b)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-4-11",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=11, rename=a, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a, b AS b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call5-5",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[5] Fail on renaming to an already bound variable name",
        cypher="CALL test.my.proc(null) YIELD a, b AS a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call5-6",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[6] Fail on renaming all outputs to the same variable name",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS c\nRETURN c",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call5-7",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[7] Fail on in-query call to procedure with YIELD *",
        cypher="CALL test.my.proc('Stefan', 1) YIELD *\nRETURN city, country_code",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),
    Scenario(
        key="call5-8",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[8] Allow standalone call to procedure with YIELD *",
        cypher="CALL test.my.proc('Stefan', 1) YIELD *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call6-1",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[1] Calling the same STRING procedure twice using the same outputs in each call",
        cypher="CALL test.labels() YIELD label\nWITH count(*) AS c\nCALL test.labels() YIELD label\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 3, "label": "'A'"},
                {"c": 3, "label": "'B'"},
                {"c": 3, "label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call6-2",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[2] Project procedure results between query scopes with WITH clause",
        cypher="CALL test.my.proc(null) YIELD out\nWITH out RETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="call6-3",
        feature_path="tck/features/clauses/call/Call6.feature",
        scenario="[3] Project procedure results between query scopes with WITH clause and rename the projection",
        cypher="CALL test.my.proc(null) YIELD out\nWITH out AS a RETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
    Scenario(
        key="union1-1",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[1] Two elements, both unique, distinct",
        cypher="RETURN 1 AS x\nUNION\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),
    Scenario(
        key="union1-2",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[2] Three elements, two unique, distinct",
        cypher="RETURN 2 AS x\nUNION\nRETURN 1 AS x\nUNION\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),
    Scenario(
        key="union1-3",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[3] Two single-column inputs, one with duplicates, distinct",
        cypher="UNWIND [2, 1, 2, 3] AS x\nRETURN x\nUNION\nUNWIND [3, 4] AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
                {"x": 3},
                {"x": 4},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION and UNWIND are not supported",
        tags=("union", "unwind", "xfail"),
    ),
    Scenario(
        key="union1-4",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[4] Should be able to create text output from union queries",
        cypher="MATCH (a:A)\nRETURN a AS a\nUNION\nMATCH (b:B)\nRETURN b AS a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)"},
                {"a": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION is not supported",
        tags=("union", "xfail"),
    ),
    Scenario(
        key="union1-5",
        feature_path="tck/features/clauses/union/Union1.feature",
        scenario="[5] Failing when UNION has different columns",
        cypher="RETURN 1 AS a\nUNION\nRETURN 2 AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),
    Scenario(
        key="union2-1",
        feature_path="tck/features/clauses/union/Union2.feature",
        scenario="[1] Two elements, both unique, not distinct",
        cypher="RETURN 1 AS x\nUNION ALL\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION ALL is not supported",
        tags=("union", "union-all", "xfail"),
    ),
    Scenario(
        key="union2-2",
        feature_path="tck/features/clauses/union/Union2.feature",
        scenario="[2] Three elements, two unique, not distinct",
        cypher="RETURN 2 AS x\nUNION ALL\nRETURN 1 AS x\nUNION ALL\nRETURN 2 AS x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION ALL is not supported",
        tags=("union", "union-all", "xfail"),
    ),
    Scenario(
        key="union2-3",
        feature_path="tck/features/clauses/union/Union2.feature",
        scenario="[3] Two single-column inputs, one with duplicates, not distinct",
        cypher="UNWIND [2, 1, 2, 3] AS x\nRETURN x\nUNION ALL\nUNWIND [3, 4] AS x\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 1},
                {"x": 2},
                {"x": 3},
                {"x": 3},
                {"x": 4},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION ALL and UNWIND are not supported",
        tags=("union", "union-all", "unwind", "xfail"),
    ),
    Scenario(
        key="union2-4",
        feature_path="tck/features/clauses/union/Union2.feature",
        scenario="[4] Should be able to create text output from union all queries",
        cypher="MATCH (a:A)\nRETURN a AS a\nUNION ALL\nMATCH (b:B)\nRETURN b AS a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)"},
                {"a": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNION ALL is not supported",
        tags=("union", "union-all", "xfail"),
    ),
    Scenario(
        key="union2-5",
        feature_path="tck/features/clauses/union/Union2.feature",
        scenario="[5] Failing when UNION ALL has different columns",
        cypher="RETURN 1 AS a\nUNION ALL\nRETURN 2 AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION ALL validation is not supported",
        tags=("union", "union-all", "syntax-error", "xfail"),
    ),
    Scenario(
        key="union3-1",
        feature_path="tck/features/clauses/union/Union3.feature",
        scenario="[1] Failing when mixing UNION and UNION ALL",
        cypher="RETURN 1 AS a\nUNION\nRETURN 2 AS a\nUNION ALL\nRETURN 3 AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION clause composition validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),
    Scenario(
        key="union3-2",
        feature_path="tck/features/clauses/union/Union3.feature",
        scenario="[2] Failing when mixing UNION ALL and UNION",
        cypher="RETURN 1 AS a\nUNION ALL\nRETURN 2 AS a\nUNION\nRETURN 3 AS a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="UNION clause composition validation is not supported",
        tags=("union", "syntax-error", "xfail"),
    ),
]
