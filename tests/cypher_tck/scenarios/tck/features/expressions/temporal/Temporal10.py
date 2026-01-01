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
        key='expr-temporal10-1-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 1)',
        cypher="WITH duration.between(localdatetime('2018-01-01T12:00'), localdatetime('2018-01-02T10:00')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT22H'", 'dur.days': 0, 'dur.seconds': 79200, 'dur.nanosecondsOfSecond': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-1-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 2)',
        cypher="WITH duration.between(localdatetime('2018-01-02T10:00'), localdatetime('2018-01-01T12:00')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT-22H'", 'dur.days': 0, 'dur.seconds': -79200, 'dur.nanosecondsOfSecond': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-1-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 3)',
        cypher="WITH duration.between(localdatetime('2018-01-01T10:00:00.2'), localdatetime('2018-01-02T10:00:00.1')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT23H59M59.9S'", 'dur.days': 0, 'dur.seconds': 86399, 'dur.nanosecondsOfSecond': 900000000}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-1-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 4)',
        cypher="WITH duration.between(localdatetime('2018-01-02T10:00:00.1'), localdatetime('2018-01-01T10:00:00.2')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT-23H-59M-59.9S'", 'dur.days': 0, 'dur.seconds': -86400, 'dur.nanosecondsOfSecond': 100000000}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-1-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 5)',
        cypher="WITH duration.between(datetime('2017-10-28T23:00+02:00[Europe/Stockholm]'), datetime('2017-10-29T04:00+01:00[Europe/Stockholm]')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT6H'", 'dur.days': 0, 'dur.seconds': 21600, 'dur.nanosecondsOfSecond': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-1-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[1] Should split between boundaries correctly (example 6)',
        cypher="WITH duration.between(datetime('2017-10-29T04:00+01:00[Europe/Stockholm]'), datetime('2017-10-28T23:00+02:00[Europe/Stockholm]')) AS dur\n      RETURN dur, dur.days, dur.seconds, dur.nanosecondsOfSecond",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'dur': "'PT-6H'", 'dur.days': 0, 'dur.seconds': -21600, 'dur.nanosecondsOfSecond': 0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 1)',
        cypher="RETURN duration.between(date('1984-10-11'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P30Y8M13D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 2)',
        cypher="RETURN duration.between(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P31Y9M10DT21H45M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 3)',
        cypher="RETURN duration.between(date('1984-10-11'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P30Y9M10DT21H40M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 4)',
        cypher="RETURN duration.between(date('1984-10-11'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT16H30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 5)',
        cypher="RETURN duration.between(date('1984-10-11'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT16H30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 6)',
        cypher="RETURN duration.between(localtime('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-14H-30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-7',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 7)',
        cypher="RETURN duration.between(localtime('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H15M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-8',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 8)',
        cypher="RETURN duration.between(localtime('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H10M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 9)',
        cypher="RETURN duration.between(localtime('14:30'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 10)',
        cypher="RETURN duration.between(localtime('14:30'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-11',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 11)',
        cypher="RETURN duration.between(time('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-14H-30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-12',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 12)',
        cypher="RETURN duration.between(time('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H15M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-13',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 13)',
        cypher="RETURN duration.between(time('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT6H10M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-14',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 14)',
        cypher="RETURN duration.between(time('14:30'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-15',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 15)',
        cypher="RETURN duration.between(time('14:30'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT1H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-16',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 16)',
        cypher="RETURN duration.between(localdatetime('2015-07-21T21:40:32.142'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-27DT-21H-40M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-17',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 17)',
        cypher="RETURN duration.between(localdatetime('2015-07-21T21:40:32.142'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P1YT4M50S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-18',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 18)',
        cypher="RETURN duration.between(localdatetime('2015-07-21T21:40:32.142'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-19',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 19)',
        cypher="RETURN duration.between(localdatetime('2015-07-21T21:40:32.142'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-20',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 20)',
        cypher="RETURN duration.between(localdatetime('2015-07-21T21:40:32.142'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-21',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 21)',
        cypher="RETURN duration.between(datetime('2014-07-21T21:40:36.143+0200'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P11M2DT2H19M23.857S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-22',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 22)',
        cypher="RETURN duration.between(datetime('2014-07-21T21:40:36.143+0200'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P2YT4M45.999S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-23',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 23)',
        cypher="RETURN duration.between(datetime('2014-07-21T21:40:36.143+0200'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P1YT59M55.999S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-24',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 24)',
        cypher="RETURN duration.between(datetime('2014-07-21T21:40:36.143+0200'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-36.143S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-2-25',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[2] Should compute duration between two temporals (example 25)',
        cypher="RETURN duration.between(datetime('2014-07-21T21:40:36.143+0200'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-4H-10M-36.143S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 1)',
        cypher="RETURN duration.inMonths(date('1984-10-11'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P30Y8M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 2)',
        cypher="RETURN duration.inMonths(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P31Y9M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 3)',
        cypher="RETURN duration.inMonths(date('1984-10-11'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P30Y9M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 4)',
        cypher="RETURN duration.inMonths(date('1984-10-11'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 5)',
        cypher="RETURN duration.inMonths(date('1984-10-11'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 6)',
        cypher="RETURN duration.inMonths(localtime('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-7',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 7)',
        cypher="RETURN duration.inMonths(localtime('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-8',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 8)',
        cypher="RETURN duration.inMonths(localtime('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 9)',
        cypher="RETURN duration.inMonths(time('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 10)',
        cypher="RETURN duration.inMonths(time('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-11',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 11)',
        cypher="RETURN duration.inMonths(time('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-12',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 12)',
        cypher="RETURN duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-13',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 13)',
        cypher="RETURN duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P1Y'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-14',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 14)',
        cypher="RETURN duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-15',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 15)',
        cypher="RETURN duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-16',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 16)',
        cypher="RETURN duration.inMonths(localdatetime('2015-07-21T21:40:32.142'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-17',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 17)',
        cypher="RETURN duration.inMonths(datetime('2014-07-21T21:40:36.143+0200'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P11M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-18',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 18)',
        cypher="RETURN duration.inMonths(datetime('2014-07-21T21:40:36.143+0200'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P2Y'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-19',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 19)',
        cypher="RETURN duration.inMonths(datetime('2014-07-21T21:40:36.143+0200'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P1Y'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-20',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 20)',
        cypher="RETURN duration.inMonths(datetime('2014-07-21T21:40:36.143+0200'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-3-21',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[3] Should compute duration between two temporals in months (example 21)',
        cypher="RETURN duration.inMonths(datetime('2014-07-21T21:40:36.143+0200'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 1)',
        cypher="RETURN duration.inDays(date('1984-10-11'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P11213D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 2)',
        cypher="RETURN duration.inDays(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P11606D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 3)',
        cypher="RETURN duration.inDays(date('1984-10-11'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P11240D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 4)',
        cypher="RETURN duration.inDays(date('1984-10-11'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 5)',
        cypher="RETURN duration.inDays(date('1984-10-11'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 6)',
        cypher="RETURN duration.inDays(localtime('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-7',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 7)',
        cypher="RETURN duration.inDays(localtime('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-8',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 8)',
        cypher="RETURN duration.inDays(localtime('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 9)',
        cypher="RETURN duration.inDays(time('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 10)',
        cypher="RETURN duration.inDays(time('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-11',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 11)',
        cypher="RETURN duration.inDays(time('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-12',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 12)',
        cypher="RETURN duration.inDays(localdatetime('2015-07-21T21:40:32.142'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-27D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-13',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 13)',
        cypher="RETURN duration.inDays(localdatetime('2015-07-21T21:40:32.142'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P366D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-14',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 14)',
        cypher="RETURN duration.inDays(localdatetime('2015-07-21T21:40:32.142'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-15',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 15)',
        cypher="RETURN duration.inDays(localdatetime('2015-07-21T21:40:32.142'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-16',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 16)',
        cypher="RETURN duration.inDays(localdatetime('2015-07-21T21:40:32.142'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-17',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 17)',
        cypher="RETURN duration.inDays(datetime('2014-07-21T21:40:36.143+0200'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P337D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-18',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 18)',
        cypher="RETURN duration.inDays(datetime('2014-07-21T21:40:36.143+0200'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P731D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-19',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 19)',
        cypher="RETURN duration.inDays(datetime('2014-07-21T21:40:36.143+0200'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P365D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-20',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 20)',
        cypher="RETURN duration.inDays(datetime('2014-07-21T21:40:36.143+0200'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-4-21',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[4] Should compute duration between two temporals in days (example 21)',
        cypher="RETURN duration.inDays(datetime('2014-07-21T21:40:36.143+0200'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 1)',
        cypher="RETURN duration.inSeconds(date('1984-10-11'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT269112H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 2)',
        cypher="RETURN duration.inSeconds(date('1984-10-11'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT278565H45M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 3)',
        cypher="RETURN duration.inSeconds(date('1984-10-11'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT269781H40M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 4)',
        cypher="RETURN duration.inSeconds(date('1984-10-11'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT16H30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 5)',
        cypher="RETURN duration.inSeconds(date('1984-10-11'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT16H30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 6)',
        cypher="RETURN duration.inSeconds(localtime('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-14H-30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-7',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 7)',
        cypher="RETURN duration.inSeconds(localtime('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H15M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-8',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 8)',
        cypher="RETURN duration.inSeconds(localtime('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H10M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 9)',
        cypher="RETURN duration.inSeconds(localtime('14:30'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 10)',
        cypher="RETURN duration.inSeconds(localtime('14:30'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-11',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 11)',
        cypher="RETURN duration.inSeconds(time('14:30'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-14H-30M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-12',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 12)',
        cypher="RETURN duration.inSeconds(time('14:30'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT7H15M22.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-13',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 13)',
        cypher="RETURN duration.inSeconds(time('14:30'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT6H10M32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-14',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 14)',
        cypher="RETURN duration.inSeconds(time('14:30'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT2H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-15',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 15)',
        cypher="RETURN duration.inSeconds(time('14:30'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT1H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-16',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 16)',
        cypher="RETURN duration.inSeconds(localdatetime('2015-07-21T21:40:32.142'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-669H-40M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-17',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 17)',
        cypher="RETURN duration.inSeconds(localdatetime('2015-07-21T21:40:32.142'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT8784H4M50S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-18',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 18)',
        cypher="RETURN duration.inSeconds(localdatetime('2015-07-21T21:40:32.142'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-19',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 19)',
        cypher="RETURN duration.inSeconds(localdatetime('2015-07-21T21:40:32.142'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-20',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 20)',
        cypher="RETURN duration.inSeconds(localdatetime('2015-07-21T21:40:32.142'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-32.142S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-21',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 21)',
        cypher="RETURN duration.inSeconds(datetime('2014-07-21T21:40:36.143+0200'), date('2015-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT8090H19M23.857S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-22',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 22)',
        cypher="RETURN duration.inSeconds(datetime('2014-07-21T21:40:36.143+0200'), localdatetime('2016-07-21T21:45:22.142')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT17544H4M45.999S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-23',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 23)',
        cypher="RETURN duration.inSeconds(datetime('2014-07-21T21:40:36.143+0200'), datetime('2015-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT8760H59M55.999S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-24',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 24)',
        cypher="RETURN duration.inSeconds(datetime('2014-07-21T21:40:36.143+0200'), localtime('16:30')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-5H-10M-36.143S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-5-25',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[5] Should compute duration between two temporals in seconds (example 25)',
        cypher="RETURN duration.inSeconds(datetime('2014-07-21T21:40:36.143+0200'), time('16:30+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-4H-10M-36.143S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[6] Should compute duration between if they differ only by a fraction of a second and the first comes after the second.',
        cypher="RETURN duration.inSeconds(localdatetime('2014-07-21T21:40:36.143'), localdatetime('2014-07-21T21:40:36.142')) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'PT-0.001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-7-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[7] Should compute negative duration between in big units (example 1)',
        cypher="RETURN duration.inMonths(date('2018-03-11'), date('2016-06-24')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-1Y-8M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-7-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[7] Should compute negative duration between in big units (example 2)',
        cypher="RETURN duration.inMonths(date('2018-07-21'), datetime('2016-07-21T21:40:32.142+0100')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-1Y-11M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-7-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[7] Should compute negative duration between in big units (example 3)',
        cypher="RETURN duration.inMonths(localdatetime('2018-07-21T21:40:32.142'), date('2016-07-21')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-2Y'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-7-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[7] Should compute negative duration between in big units (example 4)',
        cypher="RETURN duration.inMonths(datetime('2018-07-21T21:40:36.143+0200'), localdatetime('2016-07-21T21:40:36.143')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-2Y'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-7-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[7] Should compute negative duration between in big units (example 5)',
        cypher="RETURN duration.inMonths(datetime('2018-07-21T21:40:36.143+0500'), datetime('1984-07-21T22:40:36.143+0200')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P-33Y-11M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 1)',
        cypher="RETURN duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), localdatetime({year: 2017, month: 10, day: 29, hour: 4})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT5H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 2)',
        cypher="RETURN duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), localtime({hour: 4})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT5H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 3)',
        cypher="RETURN duration.inSeconds(localdatetime({year: 2017, month: 10, day: 29, hour: 0 }), datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT5H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 4)',
        cypher="RETURN duration.inSeconds(localtime({hour: 0 }), datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT5H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 5)',
        cypher="RETURN duration.inSeconds(date({year: 2017, month: 10, day: 29}), datetime({year: 2017, month: 10, day: 29, hour: 4, timezone: 'Europe/Stockholm'})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT5H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-8-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[8] Should handle durations at daylight saving time day (example 6)',
        cypher="RETURN duration.inSeconds(datetime({year: 2017, month: 10, day: 29, hour: 0, timezone: 'Europe/Stockholm'}), date({year: 2017, month: 10, day: 30})) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT25H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[9] Should handle large durations',
        cypher="RETURN duration.between(date('-999999999-01-01'), date('+999999999-12-31')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'P1999999998Y11M30D'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[10] Should handle large durations in seconds',
        cypher="RETURN duration.inSeconds(localdatetime('-999999999-01-01'), localdatetime('+999999999-12-31T23:59:59')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT17531639991215H59M59S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 1)',
        cypher="RETURN duration.inSeconds(localtime('12:34:54.7'), localtime('12:34:54.3')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-0.4S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 2)',
        cypher="RETURN duration.inSeconds(localtime('12:34:54.3'), localtime('12:34:54.7')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0.4S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 3)',
        cypher="RETURN duration.inSeconds(localtime('12:34:54.7'), localtime('12:34:55.3')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0.6S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 4)',
        cypher="RETURN duration.inSeconds(localtime('12:34:54.7'), localtime('12:44:55.3')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT10M0.6S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 5)',
        cypher="RETURN duration.inSeconds(localtime('12:44:54.7'), localtime('12:34:55.3')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-9M-59.4S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-6',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 6)',
        cypher="RETURN duration.inSeconds(localtime('12:34:56'), localtime('12:34:55.7')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-0.3S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-7',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 7)',
        cypher="RETURN duration.inSeconds(localtime('12:34:56'), localtime('12:44:55.7')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT9M59.7S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-8',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 8)',
        cypher="RETURN duration.inSeconds(localtime('12:44:56'), localtime('12:34:55.7')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-10M-0.3S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-9',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 9)',
        cypher="RETURN duration.inSeconds(localtime('12:34:56.3'), localtime('12:34:54.7')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT-1.6S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-11-10',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[11] Should handle when seconds and subseconds have different signs (example 10)',
        cypher="RETURN duration.inSeconds(localtime('12:34:54.7'), localtime('12:34:56.3')) AS duration",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT1.6S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-12-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[12] Should compute durations with no difference (example 1)',
        cypher='RETURN duration.inSeconds(localtime(), localtime()) AS duration',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-12-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[12] Should compute durations with no difference (example 2)',
        cypher='RETURN duration.inSeconds(time(), time()) AS duration',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-12-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[12] Should compute durations with no difference (example 3)',
        cypher='RETURN duration.inSeconds(date(), date()) AS duration',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-12-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[12] Should compute durations with no difference (example 4)',
        cypher='RETURN duration.inSeconds(localdatetime(), localdatetime()) AS duration',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-12-5',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[12] Should compute durations with no difference (example 5)',
        cypher='RETURN duration.inSeconds(datetime(), datetime()) AS duration',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'duration': "'PT0S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-13-1',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[13] Should propagate null (example 1)',
        cypher='RETURN duration.between(null, null) AS t',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'t': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-13-2',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[13] Should propagate null (example 2)',
        cypher='RETURN duration.inMonths(null, null) AS t',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'t': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-13-3',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[13] Should propagate null (example 3)',
        cypher='RETURN duration.inDays(null, null) AS t',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'t': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal10-13-4',
        feature_path='tck/features/expressions/temporal/Temporal10.feature',
        scenario='[13] Should propagate null (example 4)',
        cypher='RETURN duration.inSeconds(null, null) AS t',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'t': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
