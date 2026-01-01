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
]
