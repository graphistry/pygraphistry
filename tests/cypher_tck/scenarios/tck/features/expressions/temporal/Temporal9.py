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
        key='expr-temporal9-1-1',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 1)',
        cypher="RETURN date.truncate('millennium', date({year: 2017, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-2',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 2)',
        cypher="RETURN date.truncate('millennium', date({year: 2017, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-3',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 3)',
        cypher="RETURN date.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-4',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 4)',
        cypher="RETURN date.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-5',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 5)',
        cypher="RETURN date.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-6',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 6)',
        cypher="RETURN date.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-7',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 7)',
        cypher="RETURN date.truncate('century', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-8',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 8)',
        cypher="RETURN date.truncate('century', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-9',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 9)',
        cypher="RETURN date.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-10',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 10)',
        cypher="RETURN date.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-11',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 11)',
        cypher="RETURN date.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-12',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 12)',
        cypher="RETURN date.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-13',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 13)',
        cypher="RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-14',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 14)',
        cypher="RETURN date.truncate('decade', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-15',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 15)',
        cypher="RETURN date.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-16',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 16)',
        cypher="RETURN date.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-17',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 17)',
        cypher="RETURN date.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-18',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 18)',
        cypher="RETURN date.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-19',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 19)',
        cypher="RETURN date.truncate('year', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-20',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 20)',
        cypher="RETURN date.truncate('year', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-21',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 21)',
        cypher="RETURN date.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-22',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 22)',
        cypher="RETURN date.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-23',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 23)',
        cypher="RETURN date.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-24',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 24)',
        cypher="RETURN date.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-25',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 25)',
        cypher="RETURN date.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-05'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-26',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 26)',
        cypher="RETURN date.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-27',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 27)',
        cypher="RETURN date.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-28',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 28)',
        cypher="RETURN date.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-29',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 29)',
        cypher="RETURN date.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-30',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 30)',
        cypher="RETURN date.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-31',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 31)',
        cypher="RETURN date.truncate('quarter', date({year: 1984, month: 11, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-32',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 32)',
        cypher="RETURN date.truncate('quarter', date({year: 1984, month: 11, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-33',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 33)',
        cypher="RETURN date.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-34',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 34)',
        cypher="RETURN date.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-35',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 35)',
        cypher="RETURN date.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-36',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 36)',
        cypher="RETURN date.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-37',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 37)',
        cypher="RETURN date.truncate('month', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-38',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 38)',
        cypher="RETURN date.truncate('month', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-39',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 39)',
        cypher="RETURN date.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-40',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 40)',
        cypher="RETURN date.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-41',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 41)',
        cypher="RETURN date.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-42',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 42)',
        cypher="RETURN date.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-43',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 43)',
        cypher="RETURN date.truncate('week', date({year: 1984, month: 10, day: 11}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-44',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 44)',
        cypher="RETURN date.truncate('week', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-45',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 45)',
        cypher="RETURN date.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-46',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 46)',
        cypher="RETURN date.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-47',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 47)',
        cypher="RETURN date.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-48',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 48)',
        cypher="RETURN date.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-49',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 49)',
        cypher="RETURN date.truncate('day', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-50',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 50)',
        cypher="RETURN date.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-1-51',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[1] Should truncate date (example 51)',
        cypher="RETURN date.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-1',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 1)',
        cypher="RETURN datetime.truncate('millennium', date({year: 2017, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-2',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 2)',
        cypher="RETURN datetime.truncate('millennium', date({year: 2017, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-3',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 3)',
        cypher="RETURN datetime.truncate('millennium', date({year: 2017, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-4',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 4)',
        cypher="RETURN datetime.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-5',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 5)',
        cypher="RETURN datetime.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-6',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 6)',
        cypher="RETURN datetime.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-7',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 7)',
        cypher="RETURN datetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-8',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 8)',
        cypher="RETURN datetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-9',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 9)',
        cypher="RETURN datetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-10',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 10)',
        cypher="RETURN datetime.truncate('century', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-11',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 11)',
        cypher="RETURN datetime.truncate('century', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-12',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 12)',
        cypher="RETURN datetime.truncate('century', date({year: 2017, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-13',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 13)',
        cypher="RETURN datetime.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-14',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 14)',
        cypher="RETURN datetime.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-15',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 15)',
        cypher="RETURN datetime.truncate('century', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-16',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 16)',
        cypher="RETURN datetime.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-17',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 17)',
        cypher="RETURN datetime.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-18',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 18)',
        cypher="RETURN datetime.truncate('century', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-19',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 19)',
        cypher="RETURN datetime.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-20',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 20)',
        cypher="RETURN datetime.truncate('decade', date({year: 1984, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-21',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 21)',
        cypher="RETURN datetime.truncate('decade', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-22',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 22)',
        cypher="RETURN datetime.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-23',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 23)',
        cypher="RETURN datetime.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-24',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 24)',
        cypher="RETURN datetime.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-25',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 25)',
        cypher="RETURN datetime.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-26',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 26)',
        cypher="RETURN datetime.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-27',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 27)',
        cypher="RETURN datetime.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-28',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 28)',
        cypher="RETURN datetime.truncate('year', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-29',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 29)',
        cypher="RETURN datetime.truncate('year', date({year: 1984, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-30',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 30)',
        cypher="RETURN datetime.truncate('year', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-31',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 31)',
        cypher="RETURN datetime.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-32',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 32)',
        cypher="RETURN datetime.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-33',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 33)',
        cypher="RETURN datetime.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-34',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 34)',
        cypher="RETURN datetime.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-35',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 35)',
        cypher="RETURN datetime.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-36',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 36)',
        cypher="RETURN datetime.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-37',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 37)',
        cypher="RETURN datetime.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-05T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-38',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 38)',
        cypher="RETURN datetime.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-39',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 39)',
        cypher="RETURN datetime.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-40',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 40)',
        cypher="RETURN datetime.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-41',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 41)',
        cypher="RETURN datetime.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-42',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 42)',
        cypher="RETURN datetime.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-43',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 43)',
        cypher="RETURN datetime.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-44',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 44)',
        cypher="RETURN datetime.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-45',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 45)',
        cypher="RETURN datetime.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-46',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 46)',
        cypher="RETURN datetime.truncate('quarter', date({year: 1984, month: 11, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-47',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 47)',
        cypher="RETURN datetime.truncate('quarter', date({year: 1984, month: 11, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-48',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 48)',
        cypher="RETURN datetime.truncate('quarter', date({year: 1984, month: 11, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-49',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 49)',
        cypher="RETURN datetime.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-50',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 50)',
        cypher="RETURN datetime.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-51',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 51)',
        cypher="RETURN datetime.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-52',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 52)',
        cypher="RETURN datetime.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-53',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 53)',
        cypher="RETURN datetime.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-54',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 54)',
        cypher="RETURN datetime.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-55',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 55)',
        cypher="RETURN datetime.truncate('month', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-56',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 56)',
        cypher="RETURN datetime.truncate('month', date({year: 1984, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-57',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 57)',
        cypher="RETURN datetime.truncate('month', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-58',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 58)',
        cypher="RETURN datetime.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-59',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 59)',
        cypher="RETURN datetime.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-60',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 60)',
        cypher="RETURN datetime.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-61',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 61)',
        cypher="RETURN datetime.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-62',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 62)',
        cypher="RETURN datetime.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-63',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 63)',
        cypher="RETURN datetime.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-64',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 64)',
        cypher="RETURN datetime.truncate('week', date({year: 1984, month: 10, day: 11}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-65',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 65)',
        cypher="RETURN datetime.truncate('week', date({year: 1984, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-66',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 66)',
        cypher="RETURN datetime.truncate('week', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-67',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 67)',
        cypher="RETURN datetime.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-68',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 68)',
        cypher="RETURN datetime.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-69',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 69)',
        cypher="RETURN datetime.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-70',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 70)',
        cypher="RETURN datetime.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-71',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 71)',
        cypher="RETURN datetime.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-72',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 72)',
        cypher="RETURN datetime.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-73',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 73)',
        cypher="RETURN datetime.truncate('day', date({year: 1984, month: 10, day: 11}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-74',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 74)',
        cypher="RETURN datetime.truncate('day', date({year: 1984, month: 10, day: 11}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-75',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 75)',
        cypher="RETURN datetime.truncate('day', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-76',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 76)',
        cypher="RETURN datetime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-77',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 77)',
        cypher="RETURN datetime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-78',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 78)',
        cypher="RETURN datetime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-79',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 79)',
        cypher="RETURN datetime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-80',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 80)',
        cypher="RETURN datetime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-81',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 81)',
        cypher="RETURN datetime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-82',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 82)',
        cypher="RETURN datetime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-83',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 83)',
        cypher="RETURN datetime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal9-2-84',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 84)',
        cypher="RETURN datetime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-85',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 85)',
        cypher="RETURN datetime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-86',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 86)',
        cypher="RETURN datetime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal9-2-87',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 87)',
        cypher="RETURN datetime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-88',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 88)',
        cypher="RETURN datetime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-89',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 89)',
        cypher="RETURN datetime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-90',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 90)',
        cypher="RETURN datetime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-91',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 91)',
        cypher="RETURN datetime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-92',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 92)',
        cypher="RETURN datetime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-93',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 93)',
        cypher="RETURN datetime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-94',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 94)',
        cypher="RETURN datetime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.000000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-95',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 95)',
        cypher="RETURN datetime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-96',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 96)',
        cypher="RETURN datetime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-97',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 97)',
        cypher="RETURN datetime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-98',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 98)',
        cypher="RETURN datetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-99',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 99)',
        cypher="RETURN datetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-100',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 100)',
        cypher="RETURN datetime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-101',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 101)',
        cypher="RETURN datetime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-2-102',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 102)',
        cypher="RETURN datetime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-103',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 103)',
        cypher="RETURN datetime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-2-104',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 104)',
        cypher="RETURN datetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-2-105',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[2] Should truncate datetime (example 105)',
        cypher="RETURN datetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-1',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 1)',
        cypher="RETURN localdatetime.truncate('millennium', date({year: 2017, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-2',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 2)',
        cypher="RETURN localdatetime.truncate('millennium', date({year: 2017, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-3',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 3)',
        cypher="RETURN localdatetime.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-4',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 4)',
        cypher="RETURN localdatetime.truncate('millennium', datetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-5',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 5)',
        cypher="RETURN localdatetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-6',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 6)',
        cypher="RETURN localdatetime.truncate('millennium', localdatetime({year: 2017, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2000-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-7',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 7)',
        cypher="RETURN localdatetime.truncate('century', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-8',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 8)',
        cypher="RETURN localdatetime.truncate('century', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-9',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 9)',
        cypher="RETURN localdatetime.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-10',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 10)',
        cypher="RETURN localdatetime.truncate('century', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-11',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 11)',
        cypher="RETURN localdatetime.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-12',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 12)',
        cypher="RETURN localdatetime.truncate('century', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1900-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-13',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 13)',
        cypher="RETURN localdatetime.truncate('decade', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-14',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 14)',
        cypher="RETURN localdatetime.truncate('decade', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-15',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 15)',
        cypher="RETURN localdatetime.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-16',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 16)',
        cypher="RETURN localdatetime.truncate('decade', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-17',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 17)',
        cypher="RETURN localdatetime.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-18',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 18)',
        cypher="RETURN localdatetime.truncate('decade', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1980-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-19',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 19)',
        cypher="RETURN localdatetime.truncate('year', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-20',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 20)',
        cypher="RETURN localdatetime.truncate('year', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-21',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 21)',
        cypher="RETURN localdatetime.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-22',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 22)',
        cypher="RETURN localdatetime.truncate('year', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-23',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 23)',
        cypher="RETURN localdatetime.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-24',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 24)',
        cypher="RETURN localdatetime.truncate('year', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-25',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 25)',
        cypher="RETURN localdatetime.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-05T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-26',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 26)',
        cypher="RETURN localdatetime.truncate('weekYear', date({year: 1984, month: 2, day: 1}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-01-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-27',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 27)',
        cypher="RETURN localdatetime.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-28',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 28)',
        cypher="RETURN localdatetime.truncate('weekYear', datetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-29',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 29)',
        cypher="RETURN localdatetime.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 5}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-05T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-30',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 30)',
        cypher="RETURN localdatetime.truncate('weekYear', localdatetime({year: 1984, month: 1, day: 1, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1983-01-03T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-31',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 31)',
        cypher="RETURN localdatetime.truncate('quarter', date({year: 1984, month: 11, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-32',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 32)',
        cypher="RETURN localdatetime.truncate('quarter', date({year: 1984, month: 11, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-33',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 33)',
        cypher="RETURN localdatetime.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-34',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 34)',
        cypher="RETURN localdatetime.truncate('quarter', datetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-35',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 35)',
        cypher="RETURN localdatetime.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-36',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 36)',
        cypher="RETURN localdatetime.truncate('quarter', localdatetime({year: 1984, month: 11, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-37',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 37)',
        cypher="RETURN localdatetime.truncate('month', date({year: 1984, month: 10, day: 11}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-38',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 38)',
        cypher="RETURN localdatetime.truncate('month', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-39',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 39)',
        cypher="RETURN localdatetime.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-40',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 40)',
        cypher="RETURN localdatetime.truncate('month', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-41',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 41)',
        cypher="RETURN localdatetime.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {day: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-02T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-42',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 42)',
        cypher="RETURN localdatetime.truncate('month', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-43',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 43)',
        cypher="RETURN localdatetime.truncate('week', date({year: 1984, month: 10, day: 11}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-44',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 44)',
        cypher="RETURN localdatetime.truncate('week', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-45',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 45)',
        cypher="RETURN localdatetime.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-46',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 46)',
        cypher="RETURN localdatetime.truncate('week', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-47',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 47)',
        cypher="RETURN localdatetime.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {dayOfWeek: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-09T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-48',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 48)',
        cypher="RETURN localdatetime.truncate('week', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-08T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-49',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 49)',
        cypher="RETURN localdatetime.truncate('day', date({year: 1984, month: 10, day: 11}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-50',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 50)',
        cypher="RETURN localdatetime.truncate('day', date({year: 1984, month: 10, day: 11}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-51',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 51)',
        cypher="RETURN localdatetime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-52',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 52)',
        cypher="RETURN localdatetime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-53',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 53)',
        cypher="RETURN localdatetime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-54',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 54)',
        cypher="RETURN localdatetime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-55',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 55)',
        cypher="RETURN localdatetime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-56',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 56)',
        cypher="RETURN localdatetime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-3-57',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 57)',
        cypher="RETURN localdatetime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-58',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 58)',
        cypher="RETURN localdatetime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-3-59',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 59)',
        cypher="RETURN localdatetime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-60',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 60)',
        cypher="RETURN localdatetime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-61',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 61)',
        cypher="RETURN localdatetime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-62',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 62)',
        cypher="RETURN localdatetime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-63',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 63)',
        cypher="RETURN localdatetime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-64',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 64)',
        cypher="RETURN localdatetime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-65',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 65)',
        cypher="RETURN localdatetime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-66',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 66)',
        cypher="RETURN localdatetime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-67',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 67)',
        cypher="RETURN localdatetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-68',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 68)',
        cypher="RETURN localdatetime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-3-69',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 69)',
        cypher="RETURN localdatetime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-70',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 70)',
        cypher="RETURN localdatetime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-3-71',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 71)',
        cypher="RETURN localdatetime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-72',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 72)',
        cypher="RETURN localdatetime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-3-73',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 73)',
        cypher="RETURN localdatetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-3-74',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[3] Should truncate localdatetime (example 74)',
        cypher="RETURN localdatetime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-1',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 1)',
        cypher="RETURN localtime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-2',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 2)',
        cypher="RETURN localtime.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-3',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 3)',
        cypher="RETURN localtime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-4',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 4)',
        cypher="RETURN localtime.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-5',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 5)',
        cypher="RETURN localtime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-6',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 6)',
        cypher="RETURN localtime.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-4-7',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 7)',
        cypher="RETURN localtime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-8',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 8)',
        cypher="RETURN localtime.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-9',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 9)',
        cypher="RETURN localtime.truncate('hour', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-10',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 10)',
        cypher="RETURN localtime.truncate('hour', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-11',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 11)',
        cypher="RETURN localtime.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-12',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 12)',
        cypher="RETURN localtime.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-4-13',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 13)',
        cypher="RETURN localtime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-14',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 14)',
        cypher="RETURN localtime.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-15',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 15)',
        cypher="RETURN localtime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-16',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 16)',
        cypher="RETURN localtime.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-17',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 17)',
        cypher="RETURN localtime.truncate('minute', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-18',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 18)',
        cypher="RETURN localtime.truncate('minute', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-19',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 19)',
        cypher="RETURN localtime.truncate('minute', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-20',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 20)',
        cypher="RETURN localtime.truncate('minute', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-21',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 21)',
        cypher="RETURN localtime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-22',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 22)',
        cypher="RETURN localtime.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-23',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 23)',
        cypher="RETURN localtime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-24',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 24)',
        cypher="RETURN localtime.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-25',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 25)',
        cypher="RETURN localtime.truncate('second', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-26',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 26)',
        cypher="RETURN localtime.truncate('second', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-27',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 27)',
        cypher="RETURN localtime.truncate('second', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-28',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 28)',
        cypher="RETURN localtime.truncate('second', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-29',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 29)',
        cypher="RETURN localtime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-30',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 30)',
        cypher="RETURN localtime.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-4-31',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 31)',
        cypher="RETURN localtime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-32',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 32)',
        cypher="RETURN localtime.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-33',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 33)',
        cypher="RETURN localtime.truncate('millisecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-34',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 34)',
        cypher="RETURN localtime.truncate('millisecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-35',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 35)',
        cypher="RETURN localtime.truncate('millisecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-36',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 36)',
        cypher="RETURN localtime.truncate('millisecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-4-37',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 37)',
        cypher="RETURN localtime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-38',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 38)',
        cypher="RETURN localtime.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-4-39',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 39)',
        cypher="RETURN localtime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-40',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 40)',
        cypher="RETURN localtime.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-41',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 41)',
        cypher="RETURN localtime.truncate('microsecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-42',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 42)',
        cypher="RETURN localtime.truncate('microsecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-4-43',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 43)',
        cypher="RETURN localtime.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-4-44',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[4] Should truncate localtime (example 44)',
        cypher="RETURN localtime.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-5-1',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 1)',
        cypher="RETURN time.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00:00.000000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-2',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 2)',
        cypher="RETURN time.truncate('day', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-3',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 3)',
        cypher="RETURN time.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-4',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 4)',
        cypher="RETURN time.truncate('day', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-5',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 5)',
        cypher="RETURN time.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-6',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 6)',
        cypher="RETURN time.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: '+01:00'}) AS result",
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
        key='expr-temporal9-5-7',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 7)',
        cypher="RETURN time.truncate('hour', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-8',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 8)',
        cypher="RETURN time.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-9',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 9)',
        cypher="RETURN time.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: '+01:00'}) AS result",
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
        key='expr-temporal9-5-10',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 10)',
        cypher="RETURN time.truncate('hour', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-11',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 11)',
        cypher="RETURN time.truncate('hour', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-12',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 12)',
        cypher="RETURN time.truncate('hour', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {timezone: '+01:00'}) AS result",
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
        key='expr-temporal9-5-13',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 13)',
        cypher="RETURN time.truncate('hour', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-14',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 14)',
        cypher="RETURN time.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-15',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 15)',
        cypher="RETURN time.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {timezone: '+01:00'}) AS result",
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
        key='expr-temporal9-5-16',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 16)',
        cypher="RETURN time.truncate('hour', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:00-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-17',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 17)',
        cypher="RETURN time.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-18',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 18)',
        cypher="RETURN time.truncate('minute', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-19',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 19)',
        cypher="RETURN time.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-20',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 20)',
        cypher="RETURN time.truncate('minute', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-21',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 21)',
        cypher="RETURN time.truncate('minute', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-22',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 22)',
        cypher="RETURN time.truncate('minute', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-23',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 23)',
        cypher="RETURN time.truncate('minute', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:00.000000002-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-24',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 24)',
        cypher="RETURN time.truncate('minute', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '-01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-25',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 25)',
        cypher="RETURN time.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-26',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 26)',
        cypher="RETURN time.truncate('second', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-27',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 27)',
        cypher="RETURN time.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-28',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 28)',
        cypher="RETURN time.truncate('second', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-29',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 29)',
        cypher="RETURN time.truncate('second', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-30',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 30)',
        cypher="RETURN time.truncate('second', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-31',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 31)',
        cypher="RETURN time.truncate('second', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-32',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 32)',
        cypher="RETURN time.truncate('second', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-33',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 33)',
        cypher="RETURN time.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-34',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 34)',
        cypher="RETURN time.truncate('millisecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-35',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 35)',
        cypher="RETURN time.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-36',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 36)',
        cypher="RETURN time.truncate('millisecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-5-37',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 37)',
        cypher="RETURN time.truncate('millisecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-38',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 38)',
        cypher="RETURN time.truncate('millisecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
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
        key='expr-temporal9-5-39',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 39)',
        cypher="RETURN time.truncate('millisecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645000002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-40',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 40)',
        cypher="RETURN time.truncate('millisecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-41',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 41)',
        cypher="RETURN time.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-42',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 42)',
        cypher="RETURN time.truncate('microsecond', datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
        key='expr-temporal9-5-43',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 43)',
        cypher="RETURN time.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-44',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 44)',
        cypher="RETURN time.truncate('microsecond', localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-45',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 45)',
        cypher="RETURN time.truncate('microsecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-46',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 46)',
        cypher="RETURN time.truncate('microsecond', localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}), {}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-47',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 47)',
        cypher="RETURN time.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {nanosecond: 2}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876002+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal9-5-48',
        feature_path='tck/features/expressions/temporal/Temporal9.feature',
        scenario='[5] Should truncate time (example 48)',
        cypher="RETURN time.truncate('microsecond', time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}), {}) AS result",
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
]
