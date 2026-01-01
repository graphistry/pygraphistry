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
]
