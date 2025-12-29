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
]
