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
        key='expr-temporal5-1',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[1] Should provide accessors for date',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: date({year: 1984, month: 10, day: 11})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.year': 1984, 'd.quarter': 4, 'd.month': 10, 'd.week': 41, 'd.weekYear': 1984, 'd.day': 11, 'd.ordinalDay': 285, 'd.weekDay': 4, 'd.dayOfQuarter': 11}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-2',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[2] Should provide accessors for date in last weekYear',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.year, d.weekYear, d.week, d.weekDay',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: date({year: 1984, month: 1, day: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.year': 1984, 'd.weekYear': 1983, 'd.week': 52, 'd.weekDay': 7}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-3',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[3] Should provide accessors for local time',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.hour': 12, 'd.minute': 31, 'd.second': 14, 'd.millisecond': 645, 'd.microsecond': 645876, 'd.nanosecond': 645876123}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-4',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[4] Should provide accessors for time',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond, d.timezone, d.offset, d.offsetMinutes, d.offsetSeconds',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.hour': 12, 'd.minute': 31, 'd.second': 14, 'd.millisecond': 645, 'd.microsecond': 645876, 'd.nanosecond': 645876123, 'd.timezone': "'+01:00'", 'd.offset': "'+01:00'", 'd.offsetMinutes': 60, 'd.offsetSeconds': 3600}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-5',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[5] Should provide accessors for local date time',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter,\n             d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.year': 1984, 'd.quarter': 4, 'd.month': 11, 'd.week': 45, 'd.weekYear': 1984, 'd.day': 11, 'd.ordinalDay': 316, 'd.weekDay': 7, 'd.dayOfQuarter': 42, 'd.hour': 12, 'd.minute': 31, 'd.second': 14, 'd.millisecond': 645, 'd.microsecond': 645876, 'd.nanosecond': 645876123}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-6',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[6] Should provide accessors for date time',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.year, d.quarter, d.month, d.week, d.weekYear, d.day, d.ordinalDay, d.weekDay, d.dayOfQuarter,\n             d.hour, d.minute, d.second, d.millisecond, d.microsecond, d.nanosecond,\n             d.timezone, d.offset, d.offsetMinutes, d.offsetSeconds, d.epochSeconds, d.epochMillis',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.year': 1984, 'd.quarter': 4, 'd.month': 11, 'd.week': 45, 'd.weekYear': 1984, 'd.day': 11, 'd.ordinalDay': 316, 'd.weekDay': 7, 'd.dayOfQuarter': 42, 'd.hour': 12, 'd.minute': 31, 'd.second': 14, 'd.millisecond': 645, 'd.microsecond': 645876, 'd.nanosecond': 645876123, 'd.timezone': "'Europe/Stockholm'", 'd.offset': "'+01:00'", 'd.offsetMinutes': 60, 'd.offsetSeconds': 3600, 'd.epochSeconds': 469020674, 'd.epochMillis': 469020674645}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal5-7',
        feature_path='tck/features/expressions/temporal/Temporal5.feature',
        scenario='[7] Should provide accessors for duration',
        cypher='MATCH (v:Val)\n      WITH v.date AS d\n      RETURN d.years, d.quarters, d.months, d.weeks, d.days,\n             d.hours, d.minutes, d.seconds, d.milliseconds, d.microseconds, d.nanoseconds,\n             d.quartersOfYear, d.monthsOfQuarter, d.monthsOfYear, d.daysOfWeek, d.minutesOfHour, d.secondsOfMinute, d.millisecondsOfSecond, d.microsecondsOfSecond, d.nanosecondsOfSecond',
        graph=graph_fixture_from_create(
            """
            CREATE (:Val {date: duration({years: 1, months: 4, days: 10, hours: 1, minutes: 1, seconds: 1, nanoseconds: 111111111})})
            """
        ),
        expected=Expected(
            rows=[
            {'d.years': 1, 'd.quarters': 5, 'd.months': 16, 'd.weeks': 1, 'd.days': 10, 'd.hours': 1, 'd.minutes': 61, 'd.seconds': 3661, 'd.milliseconds': 3661111, 'd.microseconds': 3661111111, 'd.nanoseconds': 3661111111111, 'd.quartersOfYear': 1, 'd.monthsOfQuarter': 1, 'd.monthsOfYear': 4, 'd.daysOfWeek': 3, 'd.minutesOfHour': 1, 'd.secondsOfMinute': 1, 'd.millisecondsOfSecond': 111, 'd.microsecondsOfSecond': 111111, 'd.nanosecondsOfSecond': 111111111}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
