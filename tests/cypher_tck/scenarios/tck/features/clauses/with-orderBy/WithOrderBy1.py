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
        key="with-orderby1-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[1] Sort booleans in ascending order",
        cypher="UNWIND [true, false] AS bools\nWITH bools\n  ORDER BY bools\n  LIMIT 1\nRETURN bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[2] Sort booleans in descending order",
        cypher="UNWIND [true, false] AS bools\nWITH bools\n  ORDER BY bools DESC\n  LIMIT 1\nRETURN bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[3] Sort integers in ascending order",
        cypher="UNWIND [1, 3, 2] AS ints\nWITH ints\n  ORDER BY ints\n  LIMIT 2\nRETURN ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 1},
                {"ints": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[4] Sort integers in descending order",
        cypher="UNWIND [1, 3, 2] AS ints\nWITH ints\n  ORDER BY ints DESC\n  LIMIT 2\nRETURN ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 3},
                {"ints": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[5] Sort floats in ascending order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nWITH floats\n  ORDER BY floats\n  LIMIT 2\nRETURN floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 1.3},
                {"floats": 1.5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[6] Sort floats in descending order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nWITH floats\n  ORDER BY floats DESC\n  LIMIT 2\nRETURN floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 999.99},
                {"floats": 1.5},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[7] Sort strings in ascending order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nWITH strings\n  ORDER BY strings\n  LIMIT 2\nRETURN strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "''"},
                {"strings": "' '"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[8] Sort strings in descending order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nWITH strings\n  ORDER BY strings DESC\n  LIMIT 2\nRETURN strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "'one'"},
                {"strings": "'.*'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[9] Sort lists in ascending order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nWITH lists\n  ORDER BY lists\n  LIMIT 4\nRETURN lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[]"},
                {"lists": "['a']"},
                {"lists": "['a', 1]"},
                {"lists": "[1]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[10] Sort lists in descending order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nWITH lists\n  ORDER BY lists DESC\n  LIMIT 4\nRETURN lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[null, 2]"},
                {"lists": "[null, 1]"},
                {"lists": "[1, null]"},
                {"lists": "[1, 'a']"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-11",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[11] Sort dates in ascending order",
        cypher="UNWIND [date({year: 1910, month: 5, day: 6}),\n        date({year: 1980, month: 12, day: 24}),\n        date({year: 1984, month: 10, day: 12}),\n        date({year: 1985, month: 5, day: 6}),\n        date({year: 1980, month: 10, day: 24}),\n        date({year: 1984, month: 10, day: 11})] AS dates\nWITH dates\n  ORDER BY dates\n  LIMIT 2\nRETURN dates",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"dates": "'1910-05-06'"},
                {"dates": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-12",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[12] Sort dates in descending order",
        cypher="UNWIND [date({year: 1910, month: 5, day: 6}),\n        date({year: 1980, month: 12, day: 24}),\n        date({year: 1984, month: 10, day: 12}),\n        date({year: 1985, month: 5, day: 6}),\n        date({year: 1980, month: 10, day: 24}),\n        date({year: 1984, month: 10, day: 11})] AS dates\nWITH dates\n  ORDER BY dates DESC\n  LIMIT 2\nRETURN dates",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"dates": "'1985-05-06'"},
                {"dates": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-13",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[13] Sort local times in ascending order",
        cypher="UNWIND [localtime({hour: 10, minute: 35}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n        localtime({hour: 12, minute: 35, second: 13}),\n        localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\nWITH localtimes\n  ORDER BY localtimes\n  LIMIT 3\nRETURN localtimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localtimes": "'10:35'"},
                {"localtimes": "'12:30:14.645876123'"},
                {"localtimes": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-14",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[14] Sort local times in descending order",
        cypher="UNWIND [localtime({hour: 10, minute: 35}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}),\n        localtime({hour: 12, minute: 35, second: 13}),\n        localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123})] AS localtimes\nWITH localtimes\n  ORDER BY localtimes DESC\n  LIMIT 3\nRETURN localtimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localtimes": "'12:35:13'"},
                {"localtimes": "'12:31:14.645876124'"},
                {"localtimes": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-15",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[15] Sort times in ascending order",
        cypher="UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n        time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n        time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\nWITH times\n  ORDER BY times\n  LIMIT 3\nRETURN times",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"times": "'12:35:15+05:00'"},
                {"times": "'12:30:14.645876123+01:01'"},
                {"times": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-16",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[16] Sort times in descending order",
        cypher="UNWIND [time({hour: 10, minute: 35, timezone: '-08:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}),\n        time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}),\n        time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}),\n        time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'})] AS times\nWITH times\n  ORDER BY times DESC\n  LIMIT 3\nRETURN times",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"times": "'10:35-08:00'"},
                {"times": "'12:31:14.645876124+01:00'"},
                {"times": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-17",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[17] Sort local date times in ascending order",
        cypher="UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n        localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n        localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n        localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\nWITH localdatetimes\n  ORDER BY localdatetimes\n  LIMIT 3\nRETURN localdatetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localdatetimes": "'0001-01-01T01:01:01.000000001'"},
                {"localdatetimes": "'1980-12-11T12:31:14'"},
                {"localdatetimes": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-18",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[18] Sort local date times in descending order",
        cypher="UNWIND [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}),\n        localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}),\n        localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}),\n        localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}),\n        localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS localdatetimes\nWITH localdatetimes\n  ORDER BY localdatetimes DESC\n  LIMIT 3\nRETURN localdatetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"localdatetimes": "'9999-09-09T09:59:59.999999999'"},
                {"localdatetimes": "'1984-10-11T12:31:14.645876123'"},
                {"localdatetimes": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-19",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[19] Sort date times in ascending order",
        cypher="UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n        datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n        datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n        datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n        datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\nWITH datetimes\n  ORDER BY datetimes\n  LIMIT 3\nRETURN datetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"datetimes": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"datetimes": "'1980-12-11T12:31:14-11:59'"},
                {"datetimes": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-20",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[20] Sort date times in descending order",
        cypher="UNWIND [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}),\n        datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}),\n        datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}),\n        datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}),\n        datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS datetimes\nWITH datetimes\n  ORDER BY datetimes DESC\n  LIMIT 3\nRETURN datetimes",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"datetimes": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"datetimes": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"datetimes": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "unwind", "limit", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-21",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[21] Sort distinct types in ascending order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nWITH types\n  ORDER BY types\n  LIMIT 5\nRETURN types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "{a: 'map'}"},
                {"types": "(:N)"},
                {"types": "[:REL]"},
                {"types": "['list']"},
                {"types": "<(:N)-[:REL]->()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-22",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[22] Sort distinct types in descending order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nWITH types\n  ORDER BY types DESC\n  LIMIT 5\nRETURN types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "null"},
                {"types": "NaN"},
                {"types": 1.5},
                {"types": "false"},
                {"types": "'text'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-23-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-23-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool ASC)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool ASC\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-23-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[23] Sort by a boolean variable projected from a node property in ascending order (sort=bool ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool ASCENDING\n  LIMIT 3\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {bool: false})", "bool": "false"},
                {"a": "(:C {bool: false})", "bool": "false"},
                {"a": "(:E {bool: false})", "bool": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-24-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[24] Sort by a boolean variable projected from a node property in descending order (sort=bool DESC)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool DESC\n  LIMIT 2\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true})", "bool": "true"},
                {"a": "(:D {bool: true})", "bool": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-24-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[24] Sort by a boolean variable projected from a node property in descending order (sort=bool DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.bool AS bool\nWITH a, bool\n  ORDER BY bool DESCENDING\n  LIMIT 2\nRETURN a, bool",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {bool: true}),
                   (:B {bool: false}),
                   (:C {bool: false}),
                   (:D {bool: true}),
                   (:E {bool: false})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {bool: true})", "bool": "true"},
                {"a": "(:D {bool: true})", "bool": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-25-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-25-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num ASC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-25-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[25] Sort by an integer variable projected from a node property in ascending order (sort=num ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11})", "num": -11},
                {"a": "(:B {num: 5})", "num": 5},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-26-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[26] Sort by an integer variable projected from a node property in descending order (sort=num DESC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054})", "num": 7054},
                {"a": "(:C {num: 30})", "num": 30},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-26-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[26] Sort by an integer variable projected from a node property in descending order (sort=num DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 9}),
                   (:B {num: 5}),
                   (:C {num: 30}),
                   (:D {num: -11}),
                   (:E {num: 7054})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054})", "num": 7054},
                {"a": "(:C {num: 30})", "num": 30},
                {"a": "(:A {num: 9})", "num": 9},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-27-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-27-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num ASC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-27-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[27] Sort by a float variable projected from a node property in ascending order (sort=num ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num ASCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {num: -11.2943})", "num": -11.2943},
                {"a": "(:A {num: 5.025648})", "num": 5.025648},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-28-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[28] Sort by a float variable projected from a node property in descending order (sort=num DESC)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESC\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008})", "num": 7054.008},
                {"a": "(:B {num: 30.94857})", "num": 30.94857},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-28-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[28] Sort by a float variable projected from a node property in descending order (sort=num DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.num AS num\nWITH a, num\n  ORDER BY num DESCENDING\n  LIMIT 3\nRETURN a, num",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {num: 5.025648}),
                   (:B {num: 30.94857}),
                   (:C {num: 30.94856}),
                   (:D {num: -11.2943}),
                   (:E {num: 7054.008})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {num: 7054.008})", "num": 7054.008},
                {"a": "(:B {num: 30.94857})", "num": 30.94857},
                {"a": "(:C {num: 30.94856})", "num": 30.94856},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-29-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-29-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name ASC)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name ASC\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-29-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[29] Sort by a string variable projected from a node property in ascending order (sort=name ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name ASCENDING\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:E {name: 'amet'})", "name": "'amet'"},
                {"a": "(:C {name: 'dolor'})", "name": "'dolor'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-30-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[30] Sort by a string variable projected from a node property in descending order (sort=name DESC)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name DESC\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {name: 'sit'})", "name": "'sit'"},
                {"a": "(:A {name: 'lorem'})", "name": "'lorem'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-30-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[30] Sort by a string variable projected from a node property in descending order (sort=name DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.name AS name\nWITH a, name\n  ORDER BY name DESCENDING\n  LIMIT 3\nRETURN a, name",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'lorem'}),
                   (:B {name: 'ipsum'}),
                   (:C {name: 'dolor'}),
                   (:D {name: 'sit'}),
                   (:E {name: 'amet'})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:D {name: 'sit'})", "name": "'sit'"},
                {"a": "(:A {name: 'lorem'})", "name": "'lorem'"},
                {"a": "(:B {name: 'ipsum'})", "name": "'ipsum'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "xfail"),
    ),

    Scenario(
        key="with-orderby1-31-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-31-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list ASC)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list ASC\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-31-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[31] Sort by a list variable projected from a node property in ascending order (sort=list ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list ASCENDING\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:B {list: [1, 2]})", "list": "[1, 2]"},
                {"a": "(:D {list: [1, -20]})", "list": "[1, -20]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-32-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[32] Sort by a list variable projected from a node property in descending order (sort=list DESC)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list DESC\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0]})", "list": "[300, 0]"},
                {"a": "(:E {list: [2, -2, 100]})", "list": "[2, -2, 100]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-32-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[32] Sort by a list variable projected from a node property in descending order (sort=list DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.list AS list\nWITH a, list\n  ORDER BY list DESCENDING\n  LIMIT 3\nRETURN a, list",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {list: [2, -2]}),
                   (:B {list: [1, 2]}),
                   (:C {list: [300, 0]}),
                   (:D {list: [1, -20]}),
                   (:E {list: [2, -2, 100]})
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:C {list: [300, 0]})", "list": "[300, 0]"},
                {"a": "(:E {list: [2, -2, 100]})", "list": "[2, -2, 100]"},
                {"a": "(:A {list: [2, -2]})", "list": "[2, -2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and expression aliasing are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-33-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date\n  LIMIT 2\nRETURN a, date",
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
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-33-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date ASC)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date ASC\n  LIMIT 2\nRETURN a, date",
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
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-33-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[33] Sort by a date variable projected from a node property in ascending order (sort=date ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date ASCENDING\n  LIMIT 2\nRETURN a, date",
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
                {"a": "(:A {date: '1910-05-06'})", "date": "'1910-05-06'"},
                {"a": "(:E {date: '1980-10-24'})", "date": "'1980-10-24'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-34-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[34] Sort by a date variable projected from a node property in descending order (sort=date DESC)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date DESC\n  LIMIT 2\nRETURN a, date",
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
                {"a": "(:D {date: '1985-05-06'})", "date": "'1985-05-06'"},
                {"a": "(:C {date: '1984-10-12'})", "date": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-34-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[34] Sort by a date variable projected from a node property in descending order (sort=date DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.date AS date\nWITH a, date\n  ORDER BY date DESCENDING\n  LIMIT 2\nRETURN a, date",
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
                {"a": "(:D {date: '1985-05-06'})", "date": "'1985-05-06'"},
                {"a": "(:C {date: '1984-10-12'})", "date": "'1984-10-12'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-35-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-35-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time ASC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASC\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-35-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[35] Sort by a local time variable projected from a node property in ascending order (sort=time ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASCENDING\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:A {time: '10:35'})", "time": "'10:35'"},
                {"a": "(:D {time: '12:30:14.645876123'})", "time": "'12:30:14.645876123'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-36-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[36] Sort by a local time variable projected from a node property in descending order (sort=time DESC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESC\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:E {time: '12:31:15'})", "time": "'12:31:15'"},
                {"a": "(:C {time: '12:31:14.645876124'})", "time": "'12:31:14.645876124'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-36-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[36] Sort by a local time variable projected from a node property in descending order (sort=time DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESCENDING\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:E {time: '12:31:15'})", "time": "'12:31:15'"},
                {"a": "(:C {time: '12:31:14.645876124'})", "time": "'12:31:14.645876124'"},
                {"a": "(:B {time: '12:31:14.645876123'})", "time": "'12:31:14.645876123'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-37-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-37-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time ASC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASC\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-37-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[37] Sort by a time variable projected from a node property in ascending order (sort=time ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time ASCENDING\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:D {time: '12:35:15+05:00'})", "time": "'12:35:15+05:00'"},
                {"a": "(:E {time: '12:30:14.645876123+01:01'})", "time": "'12:30:14.645876123+01:01'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-38-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[38] Sort by a time variable projected from a node property in descending order (sort=time DESC)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESC\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:A {time: '10:35-08:00'})", "time": "'10:35-08:00'"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})", "time": "'12:31:14.645876124+01:00'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-38-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[38] Sort by a time variable projected from a node property in descending order (sort=time DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.time AS time\nWITH a, time\n  ORDER BY time DESCENDING\n  LIMIT 3\nRETURN a, time",
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
                {"a": "(:A {time: '10:35-08:00'})", "time": "'10:35-08:00'"},
                {"a": "(:C {time: '12:31:14.645876124+01:00'})", "time": "'12:31:14.645876124+01:00'"},
                {"a": "(:B {time: '12:31:14.645876123+01:00'})", "time": "'12:31:14.645876123+01:00'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-39-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-39-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime ASC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASC\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-39-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[39] Sort by a local date time variable projected from a node property in ascending order (sort=datetime ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASCENDING\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001'})", "datetime": "'0001-01-01T01:01:01.000000001'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14'})", "datetime": "'1980-12-11T12:31:14'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-40-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[40] Sort by a local date time variable projected from a node property in descending order (sort=datetime DESC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESC\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})", "datetime": "'9999-09-09T09:59:59.999999999'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})", "datetime": "'1984-10-11T12:31:14.645876123'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-40-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[40] Sort by a local date time variable projected from a node property in descending order (sort=datetime DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESCENDING\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999'})", "datetime": "'9999-09-09T09:59:59.999999999'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123'})", "datetime": "'1984-10-11T12:31:14.645876123'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012'})", "datetime": "'1984-10-11T12:30:14.000000012'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-41-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-41-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime ASC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASC\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-41-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[41] Sort by a date time variable projected from a node property in ascending order (sort=datetime ASCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime ASCENDING\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:C {datetime: '0001-01-01T01:01:01.000000001-11:59'})", "datetime": "'0001-01-01T01:01:01.000000001-11:59'"},
                {"a": "(:E {datetime: '1980-12-11T12:31:14-11:59'})", "datetime": "'1980-12-11T12:31:14-11:59'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-42-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[42] Sort by a date time variable projected from a node property in descending order (sort=datetime DESC)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESC\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})", "datetime": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})", "datetime": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-42-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[42] Sort by a date time variable projected from a node property in descending order (sort=datetime DESCENDING)",
        cypher="MATCH (a)\nWITH a, a.datetime AS datetime\nWITH a, datetime\n  ORDER BY datetime DESCENDING\n  LIMIT 3\nRETURN a, datetime",
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
                {"a": "(:D {datetime: '9999-09-09T09:59:59.999999999+11:59'})", "datetime": "'9999-09-09T09:59:59.999999999+11:59'"},
                {"a": "(:A {datetime: '1984-10-11T12:30:14.000000012+00:15'})", "datetime": "'1984-10-11T12:30:14.000000012+00:15'"},
                {"a": "(:B {datetime: '1984-10-11T12:31:14.645876123+00:17'})", "datetime": "'1984-10-11T12:31:14.645876123+00:17'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, LIMIT, and temporal values are not supported",
        tags=("with", "orderby", "limit", "alias", "expression", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-43-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[43] Sort by a variable that is only partially orderable on a non-distinct binding table (dir=ASC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH x\n  ORDER BY x ASC\n  LIMIT 2\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 0},
                {"x": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-43-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[43] Sort by a variable that is only partially orderable on a non-distinct binding table (dir=DESC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH x\n  ORDER BY x DESC\n  LIMIT 2\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, and LIMIT are not supported",
        tags=("with", "orderby", "unwind", "limit", "xfail"),
    ),

    Scenario(
        key="with-orderby1-44-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[44] Sort by a variable that is only partially orderable on a non-distinct binding table, but made distinct (dir=ASC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH DISTINCT x\n  ORDER BY x ASC\n  LIMIT 1\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and DISTINCT are not supported",
        tags=("with", "orderby", "unwind", "limit", "distinct", "xfail"),
    ),

    Scenario(
        key="with-orderby1-44-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[44] Sort by a variable that is only partially orderable on a non-distinct binding table, but made distinct (dir=DESC)",
        cypher="UNWIND [0, 2, 1, 2, 0, 1] AS x\nWITH DISTINCT x\n  ORDER BY x DESC\n  LIMIT 1\nRETURN x",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"x": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, UNWIND, ORDER BY, LIMIT, and DISTINCT are not supported",
        tags=("with", "orderby", "unwind", "limit", "distinct", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=booleans)",
        cypher="WITH [true, false] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=integers)",
        cypher="WITH [351, -3974856, 93, -3, 123, 0, 3, -2, 20934587, 1, 20934585, 20934586, -10] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=floats)",
        cypher="WITH [351.5, -3974856.01, -3.203957, 123.0002, 123.0001, 123.00013, 123.00011, 0.0100000, 0.0999999, 0.00000001, 3.0, 209345.87, -10.654] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=string)",
        cypher="WITH ['Sort', 'order', ' ', 'should', 'be', '', 'consistent', 'with', 'comparisons', ', ', 'where', 'comparisons are', 'defined', '!'] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=lists)",
        cypher="WITH [[2, 2], [2, -2], [1, 2], [], [1], [300, 0], [1, -20], [2, -2, 100]] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=dates)",
        cypher="WITH [date({year: 1910, month: 5, day: 6}), date({year: 1980, month: 12, day: 24}), date({year: 1984, month: 10, day: 12}), date({year: 1985, month: 5, day: 6}), date({year: 1980, month: 10, day: 24}), date({year: 1984, month: 10, day: 11})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=localtimes)",
        cypher="WITH [localtime({hour: 10, minute: 35}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876124}), localtime({hour: 12, minute: 35, second: 13}), localtime({hour: 12, minute: 30, second: 14, nanosecond: 645876123}), localtime({hour: 12, minute: 31, second: 15})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=times)",
        cypher="WITH [time({hour: 10, minute: 35, timezone: '-08:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), time({hour: 12, minute: 31, second: 14, nanosecond: 645876124, timezone: '+01:00'}), time({hour: 12, minute: 35, second: 15, timezone: '+05:00'}), time({hour: 12, minute: 30, second: 14, nanosecond: 645876123, timezone: '+01:01'}), time({hour: 12, minute: 35, second: 15, timezone: '+01:00'})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=localdatetimes)",
        cypher="WITH [localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12}), localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), localdatetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1}), localdatetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999}), localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-45-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[45] Sort order should be consistent with comparisons where comparisons are defined (example=datetimes)",
        cypher="WITH [datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 30, second: 14, nanosecond: 12, timezone: '+00:15'}), datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+00:17'}), datetime({year: 1, month: 1, day: 1, hour: 1, minute: 1, second: 1, nanosecond: 1, timezone: '-11:59'}), datetime({year: 9999, month: 9, day: 9, hour: 9, minute: 59, second: 59, nanosecond: 999999999, timezone: '+11:59'}), datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '-11:59'})] AS values\nWITH values, size(values) AS numOfValues\nUNWIND values AS value\nWITH size([ x IN values WHERE x < value ]) AS x, value, numOfValues\n  ORDER BY value\nWITH numOfValues, collect(x) AS orderedX\nRETURN orderedX = range(0, numOfValues-1) AS equal",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"equal": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, list comprehensions, aggregation, ORDER BY, and comparisons are not supported",
        tags=("with", "orderby", "unwind", "aggregation", "expression", "list", "temporal", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-1",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-2",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c ASC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c ASC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-3",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c ASCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c ASCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-4",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c DESC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c DESC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-5",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=out of scope, sort=c DESCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY c DESCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-6",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-7",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d ASC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d ASC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-8",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d ASCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d ASCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-9",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d DESC)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d DESC\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with-orderby1-46-10",
        feature_path="tck/features/clauses/with-orderBy/WithOrderBy1.feature",
        scenario="[46] Fail on sorting by an undefined variable (example=never defined, sort=d DESCENDING)",
        cypher="MATCH (a:A), (b:B), (c:C)\nWITH a, b\nWITH a\n  ORDER BY d DESCENDING\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:A), (:B), (:B), (:C)
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for ORDER BY variable scoping is not enforced",
        tags=("with", "orderby", "syntax-error", "xfail"),
    ),
]
