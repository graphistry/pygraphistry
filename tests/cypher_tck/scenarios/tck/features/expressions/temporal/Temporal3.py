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
        key='expr-temporal3-1-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 1)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 2)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 3)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other, year: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'0028-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 4)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other, day: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 5)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other, week: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 6)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other, ordinalDay: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 7)',
        cypher='WITH date({year: 1984, month: 11, day: 11}) AS other\n      RETURN date({date: other, quarter: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 8)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 9)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 10)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other, year: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'0028-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 11)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other, day: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 12)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other, week: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-13',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 13)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other, ordinalDay: 28}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-14',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 14)',
        cypher='WITH localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN date({date: other, quarter: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-15',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 15)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-16',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 16)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-17',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 17)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other, year: 28}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'0028-11-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-18',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 18)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other, day: 28}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-11-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-19',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 19)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other, week: 1}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-20',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 20)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other, ordinalDay: 28}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-1-21',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[1] Should select date (example 21)',
        cypher="WITH datetime({year: 1984, month: 11, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN date({date: other, quarter: 3}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 1)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN localtime(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 2)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN localtime({time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 3)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN localtime({time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 4)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN localtime(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 5)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN localtime({time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 6)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN localtime({time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 7)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localtime(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 8)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localtime({time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 9)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localtime({time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 10)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localtime(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 11)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localtime({time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-2-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[2] Should select local time (example 12)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localtime({time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 1)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN time(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 2)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN time({time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 3)',
        cypher="WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN time({time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 4)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN time({time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 5)',
        cypher="WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN time({time: other, second: 42, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 6)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN time(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 7)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN time({time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 8)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN time({time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'16:31:14.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 9)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN time({time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 10)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN time({time: other, second: 42, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'16:31:42.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 11)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN time(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 12)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN time({time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-13',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 13)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN time({time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-14',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 14)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN time({time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-15',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 15)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN time({time: other, second: 42, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:42.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-16',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 16)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN time(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-17',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 17)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN time({time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-18',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 18)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN time({time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-19',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 19)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN time({time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:42+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-3-20',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[3] Should select time (example 20)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN time({time: other, second: 42, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'16:00:42+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 1)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN localdatetime({date: other, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 2)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN localdatetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 3)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({date: other, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 4)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 5)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({date: other, hour: 10, minute: 10, second: 10}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-4-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[4] Should select date into local date time (example 6)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 1)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 2)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 3)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 4)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 5)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 6)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 7)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-5-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[5] Should select time into local date time (example 8)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 1)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 2)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 3)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 4)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 5)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 6)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 7)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 8)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 9)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 10)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 11)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 12)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-13',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 13)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-14',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 14)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-15',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 15)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-16',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 16)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-17',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 17)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-18',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 18)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-19',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 19)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-20',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 20)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-21',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 21)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-22',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 22)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-23',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 23)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-6-24',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[6] Should select date and time into local date time (example 24)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherTime\n      RETURN localdatetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 1)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 2)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({datetime: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 3)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN localdatetime({datetime: other, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 4)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 5)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({datetime: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-7-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[7] Should select datetime into local date time (example 6)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN localdatetime({datetime: other, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 1)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 2)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 3)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 4)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 5)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 6)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T10:10:10+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 7)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 8)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T10:10:10-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 9)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 10)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN datetime({date: other, hour: 10, minute: 10, second: 10, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T10:10:10+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 11)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-8-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[8] Should select date into date time (example 12)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS other\n      RETURN datetime({date: other, day: 28, hour: 10, minute: 10, second: 10, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T10:10:10-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 1)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 2)',
        cypher="WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 3)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 4)',
        cypher="WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645876123-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 5)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 6)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:31:14.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 7)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 8)',
        cypher="WITH time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T01:31:42.645876-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 9)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 10)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 11)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 12)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:42.645-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-13',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 13)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-14',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 14)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-15',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 15)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:42+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-9-16',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[9] Should select time into date time (example 16)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({year: 1984, month: 10, day: 11, time: other, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T01:00:42-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 1)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 2)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 3)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 4)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 5)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 6)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:31:14.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 7)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 8)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T01:31:42.645876-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 9)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 10)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-11',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 11)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-12',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 12)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-13',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 13)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-14',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 14)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-15',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 15)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-16',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 16)',
        cypher="WITH date({year: 1984, month: 10, day: 11}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T01:00:42-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-17',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 17)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-18',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 18)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-19',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 19)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-20',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 20)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645876123-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-21',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 21)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-22',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 22)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T16:31:14.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-23',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 23)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-24',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 24)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T01:31:42.645876-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-25',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 25)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-26',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 26)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-27',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 27)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-28',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 28)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-29',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 29)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-30',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 30)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-31',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 31)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:00:42+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-32',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 32)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T00:00:42-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-33',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 33)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-34',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 34)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-35',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 35)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-36',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 36)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876123-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-37',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 37)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-38',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 38)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:31:14.645876+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-39',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 39)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-40',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 40)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T01:31:42.645876-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-41',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 41)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-42',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 42)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-43',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 43)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-44',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 44)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:31:42.645-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-45',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 45)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-46',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 46)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-47',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 47)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-10-48',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[10] Should select date and time into date time (example 48)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS otherDate, datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS otherTime\n      RETURN datetime({date: otherDate, time: otherTime, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T01:00:42-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-1',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 1)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime(other) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-2',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 2)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({datetime: other}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-3',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 3)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({datetime: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-4',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 4)',
        cypher='WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({datetime: other, day: 28, second: 42}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-5',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 5)',
        cypher="WITH localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS other\n      RETURN datetime({datetime: other, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-28T12:31:42.645-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-6',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 6)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime(other) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-7',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 7)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({datetime: other}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-8',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 8)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({datetime: other, timezone: '+05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T16:00+05:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-9',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 9)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({datetime: other, day: 28, second: 42}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T12:00:42+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal3-11-10',
        feature_path='tck/features/expressions/temporal/Temporal3.feature',
        scenario='[11] Should datetime into date time (example 10)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS other\n      RETURN datetime({datetime: other, day: 28, second: 42, timezone: 'Pacific/Honolulu'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-28T01:00:42-10:00[Pacific/Honolulu]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
