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
        key='expr-temporal6-1',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[1] Should serialize date',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS d\n      RETURN toString(d) AS ts, date(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'1984-10-11'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-2',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[2] Should serialize local time',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN toString(d) AS ts, localtime(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'12:31:14.645876123'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-3',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[3] Should serialize time',
        cypher="WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\n      RETURN toString(d) AS ts, time(toString(d)) = d AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'12:31:14.645876123+01:00'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-4',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[4] Should serialize local date time',
        cypher='WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'1984-10-11T12:31:14.645876123'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-5',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[5] Should serialize date time',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\n      RETURN toString(d) AS ts, datetime(toString(d)) = d AS b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'1984-10-11T12:31:14.645876123+01:00'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-1',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 1)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'P12Y5M14DT16H13M10.000000001S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-2',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 2)',
        cypher='WITH duration({years: 12, months: 5, days: -14, hours: 16}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'P12Y5M-14DT16H'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-3',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 3)',
        cypher='WITH duration({minutes: 12, seconds: -60}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT11M'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-4',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 4)',
        cypher='WITH duration({seconds: 2, milliseconds: -1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT1.999S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-5',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 5)',
        cypher='WITH duration({seconds: -2, milliseconds: 1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT-1.999S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-6',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 6)',
        cypher='WITH duration({seconds: -2, milliseconds: -1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT-2.001S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-7',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 7)',
        cypher='WITH duration({days: 1, milliseconds: 1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'P1DT0.001S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-8',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 8)',
        cypher='WITH duration({days: 1, milliseconds: -1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'P1DT-0.001S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-9',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 9)',
        cypher='WITH duration({seconds: 60, milliseconds: -1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT59.999S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-10',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 10)',
        cypher='WITH duration({seconds: -60, milliseconds: 1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT-59.999S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-6-11',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[6] Should serialize duration (example 11)',
        cypher='WITH duration({seconds: -60, milliseconds: -1}) AS d\n      RETURN toString(d) AS ts, duration(toString(d)) = d AS b',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'PT-1M-0.001S'", 'b': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal6-7',
        feature_path='tck/features/expressions/temporal/Temporal6.feature',
        scenario='[7] Should serialize timezones correctly',
        cypher="WITH datetime({year: 2017, month: 8, day: 8, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d\n      RETURN toString(d) AS ts",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'ts': "'2017-08-08T12:31:14.645876123+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
