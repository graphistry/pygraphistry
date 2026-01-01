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
        key="with-orderby3-1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASC, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-1-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[1] Sort by two expressions, both in ascending order (sort=a.bool ASCENDING, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:D {num: -41, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASC, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASC, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASC, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASCENDING, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-2-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[2] Sort by two expressions, first in ascending order, second in descending order (sort=a.bool ASCENDING, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool ASCENDING, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:A {num: 9, bool: true})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESC, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num ASC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-3-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[3] Sort by two expressions, first in descending order, second in ascending order (sort=a.bool DESCENDING, a.num ASCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num ASCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:C {num: -30, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-4-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESC, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-4-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESC, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESC, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-4-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESCENDING, a.num DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num DESC\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-4-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[4] Sort by two expressions, both in descending order (sort=a.bool DESCENDING, a.num DESCENDING)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.bool DESCENDING, a.num DESCENDING\n  LIMIT 4\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9, bool: true}),
                   (:B {num: 5, bool: false}),
                   (:C {num: -30, bool: false}),
                   (:D {num: -41, bool: true}),
                   (:E {num: 7054, bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 9, bool: true})"},
                {"a": "(:D {num: -41, bool: true})"},
                {"a": "(:E {num: 7054, bool: false})"},
                {"a": "(:B {num: 5, bool: false})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-5-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 ASC, a.num, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 ASC, a.num, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 2, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-5-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 ASC, a.num, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 ASC, a.num, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 2, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-5-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 DESC, a.num, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 DESC, a.num, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-5-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[5] An expression without explicit sort direction is sorted in ascending order (sort=a.num % 2 DESC, a.num, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num % 2 DESC, a.num, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text ASC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, 4 + ((a.num * 2) % 2) DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num ASC, a.text DESC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 1, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text DESC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'b'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) ASC, a.num DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY 4 + ((a.num * 2) % 2) DESC, a.num DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) ASC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, 4 + ((a.num * 2) % 2) DESC, a.text ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) ASC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) ASC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-6-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[6] An constant expression does not influence the order determined by other expression before and after the constant expression (sort=a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) DESC)",
        cypher="MATCH (a)\nWITH a\n  ORDER BY a.num DESC, a.text ASC, 4 + ((a.num * 2) % 2) DESC\n  LIMIT 1\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 3, text: 'a'}),
                   ({num: 3, text: 'b'}),
                   ({num: 1, text: 'a'}),
                   ({num: 1, text: 'b'}),
                   ({num: 2, text: 'a'}),
                   ({num: 2, text: 'b'}),
                   ({num: 4, text: 'a'}),
                   ({num: 4, text: 'b'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "({num: 4, text: 'a'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression evaluation are not supported",
        tags=("with", "orderby", "limit", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a ASC, a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a ASC, a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a + 2 ASC, a + 2 DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a + 2 ASC, a + 2 DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a * a ASC, a * a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a * a ASC, a * a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a ASC, -1 * a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a ASC, -1 * a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=-1 * a DESC, a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY -1 * a DESC, a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a DESC, a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a DESC, a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a + 2 DESC, a + 2 ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a + 2 DESC, a + 2 ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a * a DESC, a * a ASC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a * a DESC, a * a ASC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=a DESC, -1 * a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY a DESC, -1 * a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-7-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[7] The order direction cannot be overwritten (sort=-1 * a ASC, a DESC)",
        cypher="UNWIND [1, 2, 3] AS a\nWITH a\n  ORDER BY -1 * a ASC, a DESC\n  LIMIT 1\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c ASC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c ASC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c DESC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c DESC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c ASC, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c ASC, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=a, c DESC, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c DESC, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c ASC, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c ASC, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c DESC, a, d)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c DESC, a, d\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=b, c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, b, c, d, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, b, c, d, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=out of scope, sort=c, d, b, b, d, c, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY c, d, b, b, d, c, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e ASC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e ASC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e DESC)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e DESC\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e ASC, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e ASC, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=a, e DESC, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, e DESC, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e ASC, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e ASC, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e DESC, a, f)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e DESC, a, f\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-23",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-24",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=b, e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-25",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, b, e, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, b, e, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-26",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=never defined, sort=e, f, b, b, f, e, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY e, f, b, b, f, e, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-27",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=a, c, e)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, e\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-28",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=a, c, e, b)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY a, c, e, b\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-29",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=b, c, a, f, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY b, c, a, f, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby3-8-30",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy3.feature",
        scenario="[8] Fail on sorting by any number of undefined variables in any position (example=mixed, sort=d, f, b, b, f, c, a)",
        cypher="WITH 1 AS a, 'b' AS b, 3 AS c, true AS d\nWITH a, b\nWITH a\n  ORDER BY d, f, b, b, f, c, a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
]
