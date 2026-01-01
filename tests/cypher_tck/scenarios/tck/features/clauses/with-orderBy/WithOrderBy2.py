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
        key="with-orderby2-1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2)\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) ASC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[1] Sort by a boolean expression in ascending order (sort=NOT (a.bool AND a.bool2) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) ASCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true, bool2: true})"},
                {"a": "(:D {bool: true, bool2: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-2-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[2] Sort by a boolean expression in descending order (sort=NOT (a.bool AND a.bool2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false, bool2: false})"},
                {"a": "(:C {bool: false, bool2: true})"},
                {"a": "(:E {bool: true, bool2: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-2-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[2] Sort by a boolean expression in descending order (sort=NOT (a.bool AND a.bool2) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY NOT (a.bool AND a.bool2) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true, bool2: true}),
                   (:B {bool: false, bool2: false}),
                   (:C {bool: false, bool2: true}),
                   (:D {bool: true, bool2: true}),
                   (:E {bool: true, bool2: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false, bool2: false})"},
                {"a": "(:C {bool: false, bool2: true})"},
                {"a": "(:E {bool: true, bool2: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-3-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-3-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-3-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[3] Sort by an integer expression in ascending order (sort=(a.num2 + (a.num * 2)) * -1 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054, num2: 1})"},
                {"a": "(:C {num: 30, num2: 3})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[4] Sort by an integer expression in descending order (sort=(a.num2 + (a.num * 2)) * -1 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11, num2: 2})"},
                {"a": "(:B {num: 5, num2: 4})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[4] Sort by an integer expression in descending order (sort=(a.num2 + (a.num * 2)) * -1 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num2 + (a.num * 2)) * -1 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, num2: 5}),
                   (:B {num: 5, num2: 4}),
                   (:C {num: 30, num2: 3}),
                   (:D {num: -11, num2: 2}),
                   (:E {num: 7054, num2: 1})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11, num2: 2})"},
                {"a": "(:B {num: 5, num2: 4})"},
                {"a": "(:A {num: 9, num2: 5})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-5-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-5-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-5-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[5] Sort by a float expression in ascending order (sort=(a.num + a.num2 * 2) * -1.01 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008, num2: 948.841})"},
                {"a": "(:B {num: 30.94857, num2: 0.00002})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-6-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[6] Sort by a float expression in descending order (sort=(a.num + a.num2 * 2) * -1.01 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943, num2: -8.5007})"},
                {"a": "(:A {num: 5.025648, num2: 1.96357})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-6-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[6] Sort by a float expression in descending order (sort=(a.num + a.num2 * 2) * -1.01 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY (a.num + a.num2 * 2) * -1.01 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648, num2: 1.96357}),
                   (:B {num: 30.94857, num2: 0.00002}),
                   (:C {num: 30.94856, num2: 0.00002}),
                   (:D {num: -11.2943, num2: -8.5007}),
                   (:E {num: 7054.008, num2: 948.841})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943, num2: -8.5007})"},
                {"a": "(:A {num: 5.025648, num2: 1.96357})"},
                {"a": "(:C {num: 30.94856, num2: 0.00002})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-7-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-7-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-7-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[7] Sort by a string expression in ascending order (sort=a.title + ' ' + a.name ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {name: 'lorem', title: 'dr.'})"},
                {"a": "(:B {name: 'ipsum', title: 'dr.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-8-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[8] Sort by a string expression in descending order (sort=a.title + ' ' + a.name DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {name: 'dolor', title: 'prof.'})"},
                {"a": "(:E {name: 'amet', title: 'prof.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-8-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[8] Sort by a string expression in descending order (sort=a.title + ' ' + a.name DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.title + ' ' + a.name DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem', title: 'dr.'}),
                   (:B {name: 'ipsum', title: 'dr.'}),
                   (:C {name: 'dolor', title: 'prof.'}),
                   (:D {name: 'sit', title: 'dr.'}),
                   (:E {name: 'amet', title: 'prof.'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {name: 'dolor', title: 'prof.'})"},
                {"a": "(:E {name: 'amet', title: 'prof.'})"},
                {"a": "(:D {name: 'sit', title: 'dr.'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-9-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby2-9-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby2-9-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[9] Sort by a list expression in ascending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0], list2: [1, -2]})"},
                {"a": "(:B {list: [1, 2], list2: [2, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby2-10-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[10] Sort by a list expression in descending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {list: [2, -2, 100], list2: [5, -2]})"},
                {"a": "(:D {list: [1, -20], list2: [4, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby2-10-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[10] Sort by a list expression in descending order (sort=[a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY [a.list2[1], a.list2[0], a.list[1]] + a.list + a.list2 DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2], list2: [3, -2]}),
                   (:B {list: [1, 2], list2: [2, -2]}),
                   (:C {list: [300, 0], list2: [1, -2]}),
                   (:D {list: [1, -20], list2: [4, -2]}),
                   (:E {list: [2, -2, 100], list2: [5, -2]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {list: [2, -2, 100], list2: [5, -2]})"},
                {"a": "(:D {list: [1, -20], list2: [4, -2]})"},
                {"a": "(:A {list: [2, -2], list2: [3, -2]})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby2-11-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2})\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-11-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) ASC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-11-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[11] Sort by a date expression in ascending order (sort=a.date + duration({months: 1, days: 2}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) ASCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {date: '1910-05-06'})"},
                {"a": "(:E {date: '1980-10-24'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-12-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[12] Sort by a date expression in descending order (sort=a.date + duration({months: 1, days: 2}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) DESC\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})"},
                {"a": "(:C {date: '1984-10-12'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-12-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[12] Sort by a date expression in descending order (sort=a.date + duration({months: 1, days: 2}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.date + duration({months: 1, days: 2}) DESCENDING\n  LIMIT 2\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {date: date({year: 1910, month: 5, day: 6})}),
                   (:B {date: date({year: 1980, month: 12, day: 24})}),
                   (:C {date: date({year: 1984, month: 10, day: 12})}),
                   (:D {date: date({year: 1985, month: 5, day: 6})}),
                   (:E {date: date({year: 1980, month: 10, day: 24})}),
                   (:F {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {date: '1985-05-06'})"},
                {"a": "(:C {date: '1984-10-12'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-13-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-13-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-13-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[13] Sort by a local time expression in ascending order (sort=a.time + duration({minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35'})"},
                {"a": "(:D {time: '12:30:14.645876123'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-14-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[14] Sort by a local time expression in descending order (sort=a.time + duration({minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})"},
                {"a": "(:C {time: '12:31:14.645876124'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-14-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[14] Sort by a local time expression in descending order (sort=a.time + duration({minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: localtime({hour: 10, minute: 35})}),
                   (:B {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {time: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124})}),
                   (:D {time: localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})}),
                   (:E {time: localtime({hour: 12, minute: 31, second: 15})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {time: '12:31:15'})"},
                {"a": "(:C {time: '12:31:14.645876124'})"},
                {"a": "(:B {time: '12:31:14.645876123'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-15-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-15-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-15-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[15] Sort by a time expression in ascending order (sort=a.time + duration({minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {time: '12:35:15+05:00'})"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-16-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[16] Sort by a time expression in descending order (sort=a.time + duration({minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-16-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[16] Sort by a time expression in descending order (sort=a.time + duration({minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.time + duration({minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {time: time({hour: 10, minute: 35, timezone: '-08:00'})}),
                   (:B {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})}),
                   (:C {time: time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'})}),
                   (:D {time: time({hour: 12, minute: 35, second: 15, timezone: '+05:00'})}),
                   (:E {time: time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {time: '10:35-08:00'})"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-17-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-17-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-17-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[17] Sort by a local date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-18-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[18] Sort by a local date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-18-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[18] Sort by a local date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12})}),
                   (:B {datetime: localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})}),
                   (:C {datetime: localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1})}),
                   (:D {datetime: localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999})}),
                   (:E {datetime: localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-19-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}))",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6})\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-19-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-19-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[19] Sort by a date time expression in ascending order (sort=a.datetime + duration({days: 4, minutes: 6}) ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) ASCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-20-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[20] Sort by a date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESC\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-20-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[20] Sort by a date time expression in descending order (sort=a.datetime + duration({days: 4, minutes: 6}) DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.datetime + duration({days: 4, minutes: 6}) DESCENDING\n  LIMIT 3\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'})}),
                   (:B {datetime: datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'})}),
                   (:C {datetime: datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'})}),
                   (:D {datetime: datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'})}),
                   (:E {datetime: datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal expressions are not supported",
        tags=("with", "orderby", "limit", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby2-21-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[21] Sort by an expression that is only partially orderable on a non-distinct binding table (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name\n  ORDER BY a.name + 'C' ASC\n  LIMIT 2\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
                {"name": "'A'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-21-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[21] Sort by an expression that is only partially orderable on a non-distinct binding table (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name\n  ORDER BY a.name + 'C' DESC\n  LIMIT 2\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'"},
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-22-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[22] Sort by an expression that is only partially orderable on a non-distinct binding table, but used as a grouping key (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name ASC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and aggregation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "xfail"),
    ),

    Scenario(
        key="with-orderby2-22-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[22] Sort by an expression that is only partially orderable on a non-distinct binding table, but used as a grouping key (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name DESC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and aggregation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "xfail"),
    ),

    Scenario(
        key="with-orderby2-23-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[23] Sort by an expression that is only partially orderable on a non-distinct binding table, but used in parts as a grouping key (dir=ASC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name + 'C' ASC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, aggregation, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-23-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[23] Sort by an expression that is only partially orderable on a non-distinct binding table, but used in parts as a grouping key (dir=DESC)",
        cypher="MATCH (a)\nWITH a.name AS name, count(*) AS cnt\n  ORDER BY a.name + 'C' DESC\n  LIMIT 1\nRETURN name, cnt",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'", "cnt": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, aggregation, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "aggregation", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby2-24-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[24] Sort by an expression that is only partially orderable on a non-distinct binding table, but made distinct (dir=ASC)",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\n  ORDER BY a.name ASC\n  LIMIT 1\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "limit", "distinct", "xfail"),
    ),

    Scenario(
        key="with-orderby2-24-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[24] Sort by an expression that is only partially orderable on a non-distinct binding table, but made distinct (dir=DESC)",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\n  ORDER BY a.name DESC\n  LIMIT 1\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'}),
                   ({name: 'C'}),
                   ({name: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "limit", "distinct", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(n))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(n)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(n.num1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(n.num1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=count(1 + n.num1))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY count(1 + n.num1)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASCENDING)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASCENDING\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESCENDING)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESCENDING\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2), n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2), n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASC, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASC, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) ASCENDING, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) ASCENDING, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESC, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESC, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=max(n.num2) DESCENDING, n.name)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY max(n.num2) DESCENDING, n.name\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name ASC, max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name ASC, max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name ASC, max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name ASC, max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name DESC, max(n.num2) ASC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name DESC, max(n.num2) ASC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name DESC, max(n.num2) DESC)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name DESC, max(n.num2) DESC\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, max(n.num2), n.name2)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, max(n.num2), n.name2\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.name, n.name2, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.name, n.name2, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n.num1, max(n.num2))",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n.num1, max(n.num2)\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, max(n.num2), n.num1)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, max(n.num2), n.num1\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby2-25-25",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy2.feature",
        scenario="[25] Fail on sorting by an aggregation (sort=n, count(n.num1), max(n.num2), n.num1)",
        cypher="MATCH (n)\nWITH n.num1 AS foo\n  ORDER BY n, count(n.num1), max(n.num2), n.num1\nRETURN foo AS foo",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY aggregation is not enforced",
        tags=("with", "orderby", "aggregation", "syntax-error", "xfail"),
    ),
]
