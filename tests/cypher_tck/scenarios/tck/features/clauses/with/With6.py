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
]
