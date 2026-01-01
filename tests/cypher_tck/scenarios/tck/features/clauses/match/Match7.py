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
]
