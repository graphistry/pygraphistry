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
        key="with-orderby4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[1] Sort by a projected expression",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\n  ORDER BY a.num + a.num2\n  LIMIT 3\nRETURN a, sum",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "sum": 5},
                {"a": "(:A {num: 3, num2: 3})", "sum": 6},
                {"a": "(:A {num: 5, num2: 2})", "sum": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[2] Sort by an alias of a projected expression",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\n  ORDER BY sum\n  LIMIT 3\nRETURN a, sum",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "sum": 5},
                {"a": "(:A {num: 3, num2: 3})", "sum": 6},
                {"a": "(:A {num: 5, num2: 2})", "sum": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[3] Sort by two projected expressions with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY a.num2 % 3, a.num + a.num2\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[4] Sort by one projected expression and one alias of a projected expression with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY a.num2 % 3, sum\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[5] Sort by one alias of a projected expression and one projected expression with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY mod, a.num + a.num2\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[6] Sort by aliases of two projected expressions with order priority being different than projection order",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum, a.num2 % 3 AS mod\n  ORDER BY mod, sum\n  LIMIT 3\nRETURN a, sum, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 3, num2: 3})", "sum": 6, "mod": 0},
                {"a": "(:A {num: 9, num2: 0})", "sum": 9, "mod": 0},
                {"a": "(:A {num: 1, num2: 4})", "sum": 5, "mod": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[7] Sort by an alias of a projected expression where the alias shadows an existing variable",
        cypher="MATCH (a:A)\nWITH a, a.num2 % 3 AS x\nWITH a, a.num + a.num2 AS x\n  ORDER BY x\n  LIMIT 3\nRETURN a, x",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "x": 5},
                {"a": "(:A {num: 3, num2: 3})", "x": 6},
                {"a": "(:A {num: 5, num2: 2})", "x": 7},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[8] Sort by non-projected existing variable",
        cypher="MATCH (a:A)\nWITH a, a.num + a.num2 AS sum\nWITH a, a.num2 % 3 AS mod\n  ORDER BY sum\n  LIMIT 3\nRETURN a, mod",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"a": "(:A {num: 1, num2: 4})", "mod": 1},
                {"a": "(:A {num: 3, num2: 3})", "mod": 0},
                {"a": "(:A {num: 5, num2: 2})", "mod": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby4-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy4.feature",
        scenario="[9] Sort by an alias of a projected expression containing the variable shadowed by the alias",
        cypher="MATCH (a:A)\nWITH a.num2 AS x\nWITH x % 3 AS x\n  ORDER BY x\n  LIMIT 3\nRETURN x",
        graph=WITH_ORDERBY4_GRAPH,
        expected=Expected(
            rows=[
                {"x": 0},
                {"x": 0},
                {"x": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),
]
