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
        key='expr-temporal8-1-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[1] Should add or subtract duration to or from date (example 1)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-03-25'", 'diff': "'1972-04-27'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-1-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[1] Should add or subtract duration to or from date (example 2)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1984-10-28'", 'diff': "'1984-09-25'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-1-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[1] Should add or subtract duration to or from date (example 3)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-10-11'", 'diff': "'1971-10-12'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-2-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[2] Should add or subtract duration to or from local time (example 1)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'04:44:24.000000003'", 'diff': "'20:18:03.999999999'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-2-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[2] Should add or subtract duration to or from local time (example 2)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'04:20:24.000000001'", 'diff': "'20:42:04.000000001'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-2-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[2] Should add or subtract duration to or from local time (example 3)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'22:29:27.500000004'", 'diff': "'02:33:00.499999998'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-3-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[3] Should add or subtract duration to or from time (example 1)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'04:44:24.000000003+01:00'", 'diff': "'20:18:03.999999999+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-3-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[3] Should add or subtract duration to or from time (example 2)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'04:20:24.000000001+01:00'", 'diff': "'20:42:04.000000001+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-3-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[3] Should add or subtract duration to or from time (example 3)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'22:29:27.500000004+01:00'", 'diff': "'02:33:00.499999998+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-4-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[4] Should add or subtract duration to or from local date time (example 1)',
        cypher='WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-03-26T04:44:24.000000003'", 'diff': "'1972-04-26T20:18:03.999999999'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-4-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[4] Should add or subtract duration to or from local date time (example 2)',
        cypher='WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1984-10-29T04:20:24.000000001'", 'diff': "'1984-09-24T20:42:04.000000001'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-4-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[4] Should add or subtract duration to or from local date time (example 3)',
        cypher='WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-10-11T22:29:27.500000004'", 'diff': "'1971-10-12T02:33:00.499999998'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-5-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[5] Should add or subtract duration to or from date time (example 1)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 2})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-03-26T04:44:24.000000003+01:00'", 'diff': "'1972-04-26T20:18:03.999999999+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-5-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[5] Should add or subtract duration to or from date time (example 2)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1984-10-29T04:20:24.000000001+01:00'", 'diff': "'1984-09-24T20:42:04.000000001+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-5-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[5] Should add or subtract duration to or from date time (example 3)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 1, timezone: '+01:00'}) AS x\n      MATCH (d:Duration)\n      RETURN x + d.dur AS sum, x - d.dur AS diff",
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {dur: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'1997-10-11T22:29:27.500000004+01:00'", 'diff': "'1971-10-12T02:33:00.499999998+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 1)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
                  CREATE (:Duration2 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P24Y10M28DT32H26M20.000000002S'", 'diff': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 2)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
                  CREATE (:Duration2 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P12Y6MT32H2M20.000000001S'", 'diff': "'P12Y4M28DT24M0.000000001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 3)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
                  CREATE (:Duration2 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P25Y4M43DT50H11M23.500000004S'", 'diff': "'P-6M-15DT-17H-45M-3.500000002S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-4',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 4)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
                  CREATE (:Duration2 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P12Y6MT32H2M20.000000001S'", 'diff': "'P-12Y-4M-28DT-24M-0.000000001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-5',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 5)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
                  CREATE (:Duration2 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P2M-28DT31H38M20S'", 'diff': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-6',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 6)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
                  CREATE (:Duration2 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P13Y15DT49H47M23.500000003S'", 'diff': "'P-12Y-10M-43DT-18H-9M-3.500000003S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-7',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 7)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
                  CREATE (:Duration2 {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P25Y4M43DT50H11M23.500000004S'", 'diff': "'P6M15DT17H45M3.500000002S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-8',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 8)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
                  CREATE (:Duration2 {date: duration({months: 1, days: -14, hours: 16, minutes: -12, seconds: 70})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P13Y15DT49H47M23.500000003S'", 'diff': "'P12Y10M43DT18H9M3.500000003S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-6-9',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[6] Should add or subtract durations (example 9)',
        cypher='MATCH (dur:Duration1), (dur2: Duration2)\n      RETURN dur.date + dur2.date AS sum, dur.date - dur2.date AS diff',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration1 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
                  CREATE (:Duration2 {date: duration({years: 12.5, months: 5.5, days: 14.5, hours: 16.5, minutes: 12.5, seconds: 70.5, nanoseconds: 3})})
            """
        ),
        expected=Expected(
            rows=[
            {'sum': "'P25Y10M58DT67H56M27.000000006S'", 'diff': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-7-1',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[7] Should multiply or divide durations by numbers (example 1)',
        cypher='MATCH (d:Duration)\n      RETURN d.date * 1 AS prod, d.date / 1 AS div',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'prod': "'P12Y5M14DT16H13M10.000000001S'", 'div': "'P12Y5M14DT16H13M10.000000001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-7-2',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[7] Should multiply or divide durations by numbers (example 2)',
        cypher='MATCH (d:Duration)\n      RETURN d.date * 2 AS prod, d.date / 2 AS div',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'prod': "'P24Y10M28DT32H26M20.000000002S'", 'div': "'P6Y2M22DT13H21M8S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal8-7-3',
        feature_path='tck/features/expressions/temporal/Temporal8.feature',
        scenario='[7] Should multiply or divide durations by numbers (example 3)',
        cypher='MATCH (d:Duration)\n      RETURN d.date * 0.5 AS prod, d.date / 0.5 AS div',
        graph=graph_fixture_from_create(
            """
            CREATE (:Duration {date: duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1})})
            """
        ),
        expected=Expected(
            rows=[
            {'prod': "'P6Y2M22DT13H21M8S'", 'div': "'P24Y10M28DT32H26M20.000000002S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
