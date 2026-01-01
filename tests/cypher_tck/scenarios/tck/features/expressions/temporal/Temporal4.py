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
        key='expr-temporal4-1-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[1] Should store date (example 1)',
        cypher='CREATE ({created: date({year: 1984, month: 10, day: 11})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-2-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[2] Should store date array (example 1)',
        cypher='CREATE ({dates: [date({year: 1984, month: 10, day: 12})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-2-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[2] Should store date array (example 2)',
        cypher='CREATE ({dates: [date({year: 1984, month: 10, day: 13}), date({year: 1984, month: 10, day: 14}), date({year: 1984, month: 10, day: 15})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-3-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[3] Should store local time (example 1)',
        cypher='CREATE ({created: localtime({hour: 12})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-4-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[4] Should store local time array (example 1)',
        cypher='CREATE ({dates: [localtime({hour: 13})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-4-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[4] Should store local time array (example 2)',
        cypher='CREATE ({dates: [localtime({hour: 14}), localtime({hour: 15}), localtime({hour: 16})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-5-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[5] Should store time (example 1)',
        cypher='CREATE ({created: time({hour: 12})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-6-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[6] Should store time array (example 1)',
        cypher='CREATE ({dates: [time({hour: 13})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-6-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[6] Should store time array (example 2)',
        cypher='CREATE ({dates: [time({hour: 14}), time({hour: 15}), time({hour: 16})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-7-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[7] Should store local date time (example 1)',
        cypher='CREATE ({created: localdatetime({year: 1912})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-8-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[8] Should store local date time array (example 1)',
        cypher='CREATE ({dates: [localdatetime({year: 1913})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-8-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[8] Should store local date time array (example 2)',
        cypher='CREATE ({dates: [localdatetime({year: 1914}), localdatetime({year: 1915}), localdatetime({year: 1916})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-9-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[9] Should store date time (example 1)',
        cypher='CREATE ({created: datetime({year: 1912})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-10-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[10] Should store date time array (example 1)',
        cypher='CREATE ({dates: [datetime({year: 1913})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-10-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[10] Should store date time array (example 2)',
        cypher='CREATE ({dates: [datetime({year: 1914}), datetime({year: 1915}), datetime({year: 1916})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-11-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[11] Should store duration (example 1)',
        cypher='CREATE ({created: duration({seconds: 12})})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-12-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[12] Should store duration array (example 1)',
        cypher='CREATE ({dates: [duration({seconds: 13})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-12-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[12] Should store duration array (example 2)',
        cypher='CREATE ({dates: [duration({seconds: 14}), duration({seconds: 15}), duration({seconds: 16})]})',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal4-13-1',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 1)',
        cypher='RETURN date(null) AS t',
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
        key='expr-temporal4-13-2',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 2)',
        cypher='RETURN date.transaction(null) AS t',
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
        key='expr-temporal4-13-3',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 3)',
        cypher='RETURN date.statement(null) AS t',
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
        key='expr-temporal4-13-4',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 4)',
        cypher='RETURN date.realtime(null) AS t',
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
        key='expr-temporal4-13-5',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 5)',
        cypher='RETURN localtime(null) AS t',
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
        key='expr-temporal4-13-6',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 6)',
        cypher='RETURN localtime.transaction(null) AS t',
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
        key='expr-temporal4-13-7',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 7)',
        cypher='RETURN localtime.statement(null) AS t',
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
        key='expr-temporal4-13-8',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 8)',
        cypher='RETURN localtime.realtime(null) AS t',
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
        key='expr-temporal4-13-9',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 9)',
        cypher='RETURN time(null) AS t',
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
        key='expr-temporal4-13-10',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 10)',
        cypher='RETURN time.transaction(null) AS t',
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
        key='expr-temporal4-13-11',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 11)',
        cypher='RETURN time.statement(null) AS t',
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
        key='expr-temporal4-13-12',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 12)',
        cypher='RETURN time.realtime(null) AS t',
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
        key='expr-temporal4-13-13',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 13)',
        cypher='RETURN localdatetime(null) AS t',
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
        key='expr-temporal4-13-14',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 14)',
        cypher='RETURN localdatetime.transaction(null) AS t',
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
        key='expr-temporal4-13-15',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 15)',
        cypher='RETURN localdatetime.statement(null) AS t',
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
        key='expr-temporal4-13-16',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 16)',
        cypher='RETURN localdatetime.realtime(null) AS t',
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
        key='expr-temporal4-13-17',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 17)',
        cypher='RETURN datetime(null) AS t',
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
        key='expr-temporal4-13-18',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 18)',
        cypher='RETURN datetime.transaction(null) AS t',
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
        key='expr-temporal4-13-19',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 19)',
        cypher='RETURN datetime.statement(null) AS t',
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
        key='expr-temporal4-13-20',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 20)',
        cypher='RETURN datetime.realtime(null) AS t',
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
        key='expr-temporal4-13-21',
        feature_path='tck/features/expressions/temporal/Temporal4.feature',
        scenario='[13] Should propagate null (example 21)',
        cypher='RETURN duration(null) AS t',
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
