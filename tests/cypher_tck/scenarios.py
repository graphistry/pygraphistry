from graphistry.compute import e_forward, e_undirected, n

from tests.cypher_tck.models import Expected, GraphFixture, Scenario
from tests.cypher_tck.parse_cypher import graph_fixture_from_create


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
]
