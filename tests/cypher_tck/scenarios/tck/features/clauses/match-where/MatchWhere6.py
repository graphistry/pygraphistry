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
]
