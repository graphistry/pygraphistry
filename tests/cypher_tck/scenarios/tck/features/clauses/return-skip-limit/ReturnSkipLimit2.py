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
]
