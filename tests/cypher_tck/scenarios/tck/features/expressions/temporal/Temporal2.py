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
        key='expr-temporal2-1-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 1)',
        cypher="RETURN date('2015-07-21') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 2)',
        cypher="RETURN date('20150721') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 3)',
        cypher="RETURN date('2015-07') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 4)',
        cypher="RETURN date('201507') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 5)',
        cypher="RETURN date('2015-W30-2') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 6)',
        cypher="RETURN date('2015W302') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 7)',
        cypher="RETURN date('2015-W30') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-20'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-8',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 8)',
        cypher="RETURN date('2015W30') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-20'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-9',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 9)',
        cypher="RETURN date('2015-202') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-10',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 10)',
        cypher="RETURN date('2015202') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-1-11',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[1] Should parse date from string (example 11)',
        cypher="RETURN date('2015') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 1)',
        cypher="RETURN localtime('21:40:32.142') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32.142'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 2)',
        cypher="RETURN localtime('214032.142') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32.142'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 3)',
        cypher="RETURN localtime('21:40:32') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 4)',
        cypher="RETURN localtime('214032') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 5)',
        cypher="RETURN localtime('21:40') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 6)',
        cypher="RETURN localtime('2140') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-2-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[2] Should parse local time from string (example 7)',
        cypher="RETURN localtime('21') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 1)',
        cypher="RETURN time('21:40:32.142+0100') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32.142+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 2)',
        cypher="RETURN time('214032.142Z') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32.142Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 3)',
        cypher="RETURN time('21:40:32+01:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 4)',
        cypher="RETURN time('214032-0100') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40:32-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 5)',
        cypher="RETURN time('21:40-01:30') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40-01:30'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 6)',
        cypher="RETURN time('2140-00:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 7)',
        cypher="RETURN time('2140-02') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'21:40-02:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-3-8',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[3] Should parse time from string (example 8)',
        cypher="RETURN time('22+18:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'22:00+18:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 1)',
        cypher="RETURN localdatetime('2015-07-21T21:40:32.142') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 2)',
        cypher="RETURN localdatetime('2015-W30-2T214032.142') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 3)',
        cypher="RETURN localdatetime('2015-202T21:40:32') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 4)',
        cypher="RETURN localdatetime('2015T214032') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-01-01T21:40:32'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 5)',
        cypher="RETURN localdatetime('20150721T21:40') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 6)',
        cypher="RETURN localdatetime('2015-W30T2140') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-20T21:40'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-4-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[4] Should parse local date time from string (example 7)',
        cypher="RETURN localdatetime('2015202T21') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 1)',
        cypher="RETURN datetime('2015-07-21T21:40:32.142+0100') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 2)',
        cypher="RETURN datetime('2015-W30-2T214032.142Z') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 3)',
        cypher="RETURN datetime('2015-202T21:40:32+01:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 4)',
        cypher="RETURN datetime('2015T214032-0100') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-01-01T21:40:32-01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 5)',
        cypher="RETURN datetime('20150721T21:40-01:30') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40-01:30'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 6)',
        cypher="RETURN datetime('2015-W30T2140-00:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-20T21:40Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 7)',
        cypher="RETURN datetime('2015-W30T2140-02') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-20T21:40-02:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-5-8',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[5] Should parse date time from string (example 8)',
        cypher="RETURN datetime('2015202T21+18:00') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:00+18:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-6-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[6] Should parse date time with named time zone from string (example 1)',
        cypher="RETURN datetime('2015-07-21T21:40:32.142+02:00[Europe/Stockholm]') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-6-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[6] Should parse date time with named time zone from string (example 2)',
        cypher="RETURN datetime('2015-07-21T21:40:32.142+0845[Australia/Eucla]') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142+08:45[Australia/Eucla]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-6-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[6] Should parse date time with named time zone from string (example 3)',
        cypher="RETURN datetime('2015-07-21T21:40:32.142-04[America/New_York]') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142-04:00[America/New_York]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-6-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[6] Should parse date time with named time zone from string (example 4)',
        cypher="RETURN datetime('2015-07-21T21:40:32.142[Europe/London]') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'2015-07-21T21:40:32.142+01:00[Europe/London]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-6-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[6] Should parse date time with named time zone from string (example 5)',
        cypher="RETURN datetime('1818-07-21T21:40:32.142[Europe/Stockholm]') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1818-07-21T21:40:32.142+00:53:28[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-1',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 1)',
        cypher="RETURN duration('P14DT16H12M') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P14DT16H12M'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-2',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 2)',
        cypher="RETURN duration('P5M1.5D') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P5M1DT12H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-3',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 3)',
        cypher="RETURN duration('P0.75M') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P22DT19H51M49.5S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-4',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 4)',
        cypher="RETURN duration('PT0.75M') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'PT45S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-5',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 5)',
        cypher="RETURN duration('P2.5W') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P17DT12H'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-6',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 6)',
        cypher="RETURN duration('P12Y5M14DT16H12M70S') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P12Y5M14DT16H13M10S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal2-7-7',
        feature_path='tck/features/expressions/temporal/Temporal2.feature',
        scenario='[7] Should parse duration from string (example 7)',
        cypher="RETURN duration('P2012-02-02T14:37:21.545') AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P2012Y2M2DT14H37M21.545S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
