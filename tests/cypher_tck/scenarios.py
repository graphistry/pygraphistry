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
]
