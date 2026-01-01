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
        key="return6-1",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[1] Return count aggregation over nodes",
        cypher="MATCH (n)\nRETURN n.num AS n, count(n) AS count",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"n": 42, "count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-2",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[2] Projecting an arithmetic expression with aggregation",
        cypher="MATCH (a)\nRETURN a, count(a) + 3",
        graph=graph_fixture_from_create(
            """
            CREATE ({id: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({id: 42})", "count(a) + 3": 4},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations, arithmetic expressions, and RETURN projections are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="return6-3",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[3] Aggregating by a list property has a correct definition of equality",
        cypher="MATCH (a)\nWITH a.num AS a, count(*) AS count\nRETURN count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "a": [1, 2, 3]},
                {"id": "n2", "labels": [], "a": [1, 2, 3]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and list property projections are not supported",
        tags=("return", "with", "aggregation", "list", "xfail"),
    ),

    Scenario(
        key="return6-4",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[4] Support multiple divisions in aggregate function",
        cypher="MATCH (n)\nRETURN count(n) / 60 / 60 AS count",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(7251)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and arithmetic expressions are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="return6-5",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[5] Aggregates inside normal functions",
        cypher="MATCH (a)\nRETURN size(collect(a))",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(11)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"size(collect(a))": 11},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations, list functions, and RETURN projections are not supported",
        tags=("return", "aggregation", "list", "xfail"),
    ),

    Scenario(
        key="return6-6",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[6] Handle aggregates inside non-aggregate expressions",
        cypher="MATCH (a {name: 'Andres'})<-[:FATHER]-(child)\nRETURN a.name, {foo: a.name='Andres', kids: collect(child.name)}",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations inside map projections are not supported",
        tags=("return", "aggregation", "map", "xfail"),
    ),

    Scenario(
        key="return6-7",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[7] Aggregate on property",
        cypher="MATCH (n)\nRETURN n.num, count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 33})
            CREATE ({num: 33})
            CREATE ({num: 42})
            """
        ),
        expected=Expected(
            rows=[
                {"n.num": 42, "count(*)": 1},
                {"n.num": 33, "count(*)": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-8",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[8] Handle aggregation on functions",
        cypher="MATCH p=(a:L)-[*]->(b)\nRETURN b, avg(length(p))",
        graph=graph_fixture_from_create(
            """
            CREATE (a:L), (b1), (b2)
            CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "()", "avg(length(p))": 1.0},
                {"b": "()", "avg(length(p))": 1.0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, path length functions, and aggregations are not supported",
        tags=("return", "variable-length", "path", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-9",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[9] Aggregates with arithmetics",
        cypher="MATCH ()\nRETURN count(*) * 10 AS c",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"c": 10},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and arithmetic expressions are not supported",
        tags=("return", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="return6-10",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[10] Multiple aggregates on same variable",
        cypher="MATCH (n)\nRETURN count(n), collect(n)",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"count(n)": 1, "collect(n)": "[()]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and list projections are not supported",
        tags=("return", "aggregation", "list", "xfail"),
    ),

    Scenario(
        key="return6-11",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[11] Counting matches",
        cypher="MATCH ()\nRETURN count(*)",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": []} for i in range(100)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count(*)": 100},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-12",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[12] Counting matches per group",
        cypher="MATCH (a:L)-[rel]->(b)\nRETURN a, count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:L), (b1), (b2)
            CREATE (a)-[:A]->(b1), (a)-[:A]->(b2)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:L)", "count(*)": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN projections are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-13",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[13] Returning the minimum length of paths",
        cypher="MATCH p = (a:T {name: 'a'})-[:R*]->(other:T)\nWHERE other <> a\nWITH a, other, min(length(p)) AS len\nRETURN a.name AS name, collect(other.name) AS others, len",
        graph=graph_fixture_from_create(
            """
            CREATE (a:T {name: 'a'}), (b:T {name: 'b'}), (c:T {name: 'c'})
            CREATE (a)-[:R]->(b)
            CREATE (a)-[:R]->(c)
            CREATE (c)-[:R]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'a'", "others": "['c', 'b']", "len": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, path length functions, WITH pipelines, and aggregations are not supported",
        tags=("return", "variable-length", "path", "with", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-14",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[14] Aggregates in aggregates",
        cypher="RETURN count(count(*))",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for nested aggregations is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-15",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[15] Using `rand()` in aggregations",
        cypher="RETURN count(rand())",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for non-constant aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-16",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[16] Aggregation on complex expressions",
        cypher="MATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)\nWHERE me.name = 'Michael'\nWITH me, count(DISTINCT r1) AS H1, count(DISTINCT r2) AS H2, you\nMATCH (me)-[r1:ATE]->()<-[r2:ATE]-(you)\nRETURN me, you, sum((1 - abs(r1.times / H1 - r2.times / H2)) * (r1.times + r2.times) / (H1 + H2)) AS sum",
        graph=graph_fixture_from_create(
            """
            CREATE (andres {name: 'Andres'}),
                   (michael {name: 'Michael'}),
                   (peter {name: 'Peter'}),
                   (bread {type: 'Bread'}),
                   (veggies {type: 'Veggies'}),
                   (meat {type: 'Meat'})
            CREATE (andres)-[:ATE {times: 10}]->(bread),
                   (andres)-[:ATE {times: 8}]->(veggies),
                   (michael)-[:ATE {times: 4}]->(veggies),
                   (michael)-[:ATE {times: 6}]->(bread),
                   (michael)-[:ATE {times: 9}]->(meat),
                   (peter)-[:ATE {times: 7}]->(veggies),
                   (peter)-[:ATE {times: 7}]->(bread),
                   (peter)-[:ATE {times: 4}]->(meat)
            """
        ),
        expected=Expected(
            rows=[
                {"me": "({name: 'Michael'})", "you": "({name: 'Andres'})", "sum": -7},
                {"me": "({name: 'Michael'})", "you": "({name: 'Peter'})", "sum": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, DISTINCT aggregations, and complex expressions are not supported",
        tags=("return", "with", "distinct", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-17",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[17] Handle constants and parameters inside an expression which contains an aggregation expression",
        cypher="MATCH (person)\nRETURN $age + avg(person.age) - 1000",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"$age + avg(person.age) - 1000": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Parameters, aggregations, and RETURN expressions are not supported",
        tags=("return", "params", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-18",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[18] Handle returned variables inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nWITH me.age AS age, you\nRETURN age, age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and RETURN expressions are not supported",
        tags=("return", "with", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-19",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[19] Handle returned property accesses inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age, me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="Aggregations and RETURN expressions are not supported",
        tags=("return", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-20",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[20] Fail if not returned variables are used inside an expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age + count(you.age)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),

    Scenario(
        key="return6-21",
        feature_path="tck/features/clauses/return/Return6.feature",
        scenario="[21] Fail if more complex expressions, even if returned, are used inside expression which contains an aggregation expression",
        cypher="MATCH (me: Person)--(you: Person)\nRETURN me.age + you.age, me.age + you.age + count(*)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ambiguous aggregation expressions is not enforced",
        tags=("return", "syntax-error", "aggregation", "xfail"),
    ),
]
