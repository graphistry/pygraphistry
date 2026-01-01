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
        key='expr-temporal7-1-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[1] Should compare dates (example 1)',
        cypher='WITH date({year: 1980, month: 12, day: 24}) AS x, date({year: 1984, month: 10, day: 11}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'true', 'x >= d': 'false', 'x <= d': 'true', 'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-1-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[1] Should compare dates (example 2)',
        cypher='WITH date({year: 1984, month: 10, day: 11}) AS x, date({year: 1984, month: 10, day: 11}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'false', 'x >= d': 'true', 'x <= d': 'true', 'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-2-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[2] Should compare local times (example 1)',
        cypher='WITH localtime({hour: 10, minute: 35}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'true', 'x >= d': 'false', 'x <= d': 'true', 'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-2-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[2] Should compare local times (example 2)',
        cypher='WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'false', 'x >= d': 'true', 'x <= d': 'true', 'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-3-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[3] Should compare times (example 1)',
        cypher="WITH time({hour: 10, minute: 0, timezone: '+01:00'}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'true', 'x >= d': 'false', 'x <= d': 'true', 'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-3-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[3] Should compare times (example 2)',
        cypher="WITH time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'false', 'x >= d': 'true', 'x <= d': 'true', 'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-4-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[4] Should compare local date times (example 1)',
        cypher='WITH localdatetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'true', 'x >= d': 'false', 'x <= d': 'true', 'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-4-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[4] Should compare local date times (example 2)',
        cypher='WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'false', 'x >= d': 'true', 'x <= d': 'true', 'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-5-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[5] Should compare date times (example 1)',
        cypher="WITH datetime({year: 1980, month: 12, day: 11, hour: 12, minute: 31, second: 14, timezone: '+00:00'}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'true', 'x >= d': 'false', 'x <= d': 'true', 'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-5-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[5] Should compare date times (example 2)',
        cypher="WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\n      RETURN x > d, x < d, x >= d, x <= d, x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x > d': '<gt>', 'x < d': 'false', 'x >= d': 'true', 'x <= d': 'true', 'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-1',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 1)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, date({year: 1984, month: 10, day: 11}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-2',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 2)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-3',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 3)',
        cypher="WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, time({hour: 9, minute: 35, second: 14, nanosecond: 645876123, timezone: '+00:00'}) AS d\n      RETURN x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-4',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 4)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-5',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 5)',
        cypher="WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+05:00'}) AS d\n      RETURN x = d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-6',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 6)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-7',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 7)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 14, hours: 16, minutes: 13, seconds: 10}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal7-6-8',
        feature_path='tck/features/expressions/temporal/Temporal7.feature',
        scenario='[6] Should compare durations for equality (example 8)',
        cypher='WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS x, duration({years: 12, months: 5, days: 13, hours: 40, minutes: 13, seconds: 10}) AS d\n      RETURN x = d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'x = d': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
