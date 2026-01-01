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
]
