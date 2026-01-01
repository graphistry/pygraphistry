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
        key='expr-temporal1-1-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 1)',
        cypher='RETURN date({year: 1816, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-01-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 2)',
        cypher='RETURN date({year: 1816, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-23'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 3)',
        cypher='RETURN date({year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-30'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 4)',
        cypher='RETURN date({year: 1817, week: 10}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-03-03'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 5)',
        cypher='RETURN date({year: 1817, week: 30}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-07-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 6)',
        cypher='RETURN date({year: 1817, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-22'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 7)',
        cypher='RETURN date({year: 1818, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-29'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 8)',
        cypher='RETURN date({year: 1818, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-21'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 9)',
        cypher='RETURN date({year: 1818, week: 53}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-28'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 10)',
        cypher='RETURN date({year: 1819, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-01-04'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 11)',
        cypher='RETURN date({year: 1819, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-12-27'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 12)',
        cypher='RETURN date({dayOfWeek: 2, year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 13)',
        cypher="RETURN date({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-08'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 14)',
        cypher="RETURN date({date: date('1816-12-31'), week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-1-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[1] Should construct week date (example 15)',
        cypher="RETURN date({date: date('1816-12-31'), year: 1817, week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 1)',
        cypher='RETURN localdatetime({year: 1816, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-01-01T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 2)',
        cypher='RETURN localdatetime({year: 1816, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-23T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 3)',
        cypher='RETURN localdatetime({year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-30T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 4)',
        cypher='RETURN localdatetime({year: 1817, week: 10}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-03-03T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 5)',
        cypher='RETURN localdatetime({year: 1817, week: 30}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-07-21T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 6)',
        cypher='RETURN localdatetime({year: 1817, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-22T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 7)',
        cypher='RETURN localdatetime({year: 1818, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-29T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 8)',
        cypher='RETURN localdatetime({year: 1818, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-21T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 9)',
        cypher='RETURN localdatetime({year: 1818, week: 53}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-28T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 10)',
        cypher='RETURN localdatetime({year: 1819, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-01-04T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 11)',
        cypher='RETURN localdatetime({year: 1819, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-12-27T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 12)',
        cypher='RETURN localdatetime({dayOfWeek: 2, year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-31T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 13)',
        cypher="RETURN localdatetime({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-08T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 14)',
        cypher="RETURN localdatetime({date: date('1816-12-31'), week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-2-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[2] Should construct week localdatetime (example 15)',
        cypher="RETURN localdatetime({date: date('1816-12-31'), year: 1817, week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 1)',
        cypher='RETURN datetime({year: 1816, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-01-01T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 2)',
        cypher='RETURN datetime({year: 1816, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-23T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 3)',
        cypher='RETURN datetime({year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-30T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 4)',
        cypher='RETURN datetime({year: 1817, week: 10}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-03-03T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 5)',
        cypher='RETURN datetime({year: 1817, week: 30}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-07-21T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 6)',
        cypher='RETURN datetime({year: 1817, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-22T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 7)',
        cypher='RETURN datetime({year: 1818, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-12-29T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 8)',
        cypher='RETURN datetime({year: 1818, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-21T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 9)',
        cypher='RETURN datetime({year: 1818, week: 53}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1818-12-28T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 10)',
        cypher='RETURN datetime({year: 1819, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-01-04T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 11)',
        cypher='RETURN datetime({year: 1819, week: 52}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1819-12-27T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 12)',
        cypher='RETURN datetime({dayOfWeek: 2, year: 1817, week: 1}) AS d',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1816-12-31T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 13)',
        cypher="RETURN datetime({date: date('1816-12-30'), week: 2, dayOfWeek: 3}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-08T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 14)',
        cypher="RETURN datetime({date: date('1816-12-31'), week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-3-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[3] Should construct week datetime (example 15)',
        cypher="RETURN datetime({date: date('1816-12-31'), year: 1817, week: 2}) AS d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d': "'1817-01-07T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-4-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 1)',
        cypher='RETURN date({year: 1984, month: 10, day: 11}) AS result',
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
        key='expr-temporal1-4-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 2)',
        cypher='RETURN date({year: 1984, month: 10}) AS result',
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
        key='expr-temporal1-4-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 3)',
        cypher='RETURN date({year: 1984, week: 10, dayOfWeek: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-4-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 4)',
        cypher='RETURN date({year: 1984, week: 10}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-05'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-4-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 5)',
        cypher='RETURN date({year: 1984}) AS result',
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
        key='expr-temporal1-4-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 6)',
        cypher='RETURN date({year: 1984, ordinalDay: 202}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-4-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 7)',
        cypher='RETURN date({year: 1984, quarter: 3, dayOfQuarter: 45}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-4-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[4] Should construct date (example 8)',
        cypher='RETURN date({year: 1984, quarter: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-01'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-5-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 1)',
        cypher='RETURN localtime({hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.123456789'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-5-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 2)',
        cypher='RETURN localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-5-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 3)',
        cypher='RETURN localtime({hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
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
        key='expr-temporal1-5-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 4)',
        cypher='RETURN localtime({hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-5-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 5)',
        cypher='RETURN localtime({hour: 12, minute: 31, second: 14}) AS result',
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
        key='expr-temporal1-5-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 6)',
        cypher='RETURN localtime({hour: 12, minute: 31}) AS result',
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
        key='expr-temporal1-5-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[5] Should construct local time (example 7)',
        cypher='RETURN localtime({hour: 12}) AS result',
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
        key='expr-temporal1-6-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 1)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.123456789Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-6-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 2)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-6-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 3)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14, nanosecond: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.000000003Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-6-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 4)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
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
        key='expr-temporal1-6-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 5)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-6-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 6)',
        cypher='RETURN time({hour: 12, minute: 31, second: 14}) AS result',
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
        key='expr-temporal1-6-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 7)',
        cypher='RETURN time({hour: 12, minute: 31}) AS result',
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
        key='expr-temporal1-6-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 8)',
        cypher='RETURN time({hour: 12}) AS result',
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
        key='expr-temporal1-6-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 9)',
        cypher="RETURN time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31:14.645876123+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-6-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 10)',
        cypher="RETURN time({hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-6-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 11)',
        cypher="RETURN time({hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-6-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 12)',
        cypher="RETURN time({hour: 12, minute: 31, second: 14, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-6-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 13)',
        cypher="RETURN time({hour: 12, minute: 31, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:31+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-6-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[6] Should construct time (example 14)',
        cypher="RETURN time({hour: 12, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-7-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 1)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.123456789'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 2)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-7-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 3)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.000000003'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 4)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
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
        key='expr-temporal1-7-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 5)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-7-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 6)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14}) AS result',
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
        key='expr-temporal1-7-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 7)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31}) AS result',
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
        key='expr-temporal1-7-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 8)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11, hour: 12}) AS result',
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
        key='expr-temporal1-7-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 9)',
        cypher='RETURN localdatetime({year: 1984, month: 10, day: 11}) AS result',
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
        key='expr-temporal1-7-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 10)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-7-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 11)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
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
        key='expr-temporal1-7-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 12)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-7-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 13)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 14)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 15)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12}) AS result',
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
        key='expr-temporal1-7-16',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 16)',
        cypher='RETURN localdatetime({year: 1984, week: 10, dayOfWeek: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-17',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 17)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-18',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 18)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-19',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 19)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-20',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 20)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-21',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 21)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-22',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 22)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202, hour: 12}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-23',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 23)',
        cypher='RETURN localdatetime({year: 1984, ordinalDay: 202}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-24',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 24)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876123'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-25',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 25)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-26',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 26)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-27',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 27)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-28',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 28)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-29',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 29)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-30',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 30)',
        cypher='RETURN localdatetime({year: 1984, quarter: 3, dayOfQuarter: 45}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T00:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-7-31',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[7] Should construct local date time (example 31)',
        cypher='RETURN localdatetime({year: 1984}) AS result',
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
        key='expr-temporal1-8-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 1)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 789, millisecond: 123, microsecond: 456}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.123456789Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 2)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-8-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 3)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
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
        key='expr-temporal1-8-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 4)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-8-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 5)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14}) AS result',
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
        key='expr-temporal1-8-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 6)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31}) AS result',
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
        key='expr-temporal1-8-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 7)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11, hour: 12}) AS result',
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
        key='expr-temporal1-8-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 8)',
        cypher='RETURN datetime({year: 1984, month: 10, day: 11}) AS result',
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
        key='expr-temporal1-8-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 9)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
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
        key='expr-temporal1-8-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 10)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 11)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
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
        key='expr-temporal1-8-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 12)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 13)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 14)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 15)',
        cypher='RETURN datetime({year: 1984, week: 10, dayOfWeek: 3}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-16',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 16)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-17',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 17)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-18',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 18)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-19',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 19)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-20',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 20)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-21',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 21)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202, hour: 12}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-22',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 22)',
        cypher='RETURN datetime({year: 1984, ordinalDay: 202}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-23',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 23)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876123Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-24',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 24)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-25',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 25)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-26',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 26)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-27',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 27)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-28',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 28)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-29',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 29)',
        cypher='RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T00:00Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-8-30',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[8] Should construct date time with default time zone (example 30)',
        cypher='RETURN datetime({year: 1984}) AS result',
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
        key='expr-temporal1-9-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 1)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 2)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-9-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 3)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-9-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 4)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-9-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 5)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 6)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 7)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-9-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 8)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876123+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 9)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-9-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 10)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 11)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 12)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 13)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 14)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 15)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876123+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-16',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 16)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-17',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 17)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-18',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 18)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-19',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 19)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-20',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 20)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-21',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 21)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-22',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 22)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876123+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-23',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 23)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-24',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 24)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-25',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 25)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-26',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 26)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-27',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 27)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-28',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 28)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, timezone: '+01:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T00:00+01:00'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-9-29',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[9] Should construct date time with offset time zone (example 29)',
        cypher="RETURN datetime({year: 1984, timezone: '+01:00'}) AS result",
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
        key='expr-temporal1-10-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 1)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876123+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 2)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645876+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 3)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14.645+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 4)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:31:14+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 5)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal1-10-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 6)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal1-10-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 7)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal1-10-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 8)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876123+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 9)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645876+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-10',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 10)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14.645+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 11)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31:14+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-12',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 12)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T12:31+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-13',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 13)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, hour: 12, timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal1-10-14',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 14)',
        cypher="RETURN datetime({year: 1984, week: 10, dayOfWeek: 3, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-03-07T00:00+01:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-15',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 15)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876123+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-16',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 16)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645876+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-17',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 17)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14.645+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-18',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 18)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31:14+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-19',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 19)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:31+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-20',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 20)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, hour: 12, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T12:00+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-21',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 21)',
        cypher="RETURN datetime({year: 1984, ordinalDay: 202, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-07-20T00:00+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-22',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 22)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876123+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-23',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 23)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, microsecond: 645876, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645876+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-24',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 24)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, millisecond: 645, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14.645+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-25',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 25)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, second: 14, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31:14+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-26',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 26)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, minute: 31, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:31+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-27',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 27)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, hour: 12, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T12:00+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-28',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 28)',
        cypher="RETURN datetime({year: 1984, quarter: 3, dayOfQuarter: 45, timezone: 'Europe/Stockholm'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-08-14T00:00+02:00[Europe/Stockholm]'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-10-29',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[10] Should construct date time with named time zone (example 29)',
        cypher="RETURN datetime({year: 1984, timezone: 'Europe/Stockholm'}) AS result",
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
        key='expr-temporal1-11',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[11] Should construct date time from epoch',
        cypher='RETURN datetime.fromepoch(416779, 999999999) AS d1,\n             datetime.fromepochmillis(237821673987) AS d2',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'d1': "'1970-01-05T19:46:19.999999999Z'", 'd2': "'1977-07-15T13:34:33.987Z'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-12-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 1)',
        cypher='RETURN duration({days: 14, hours: 16, minutes: 12}) AS result',
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
        key='expr-temporal1-12-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 2)',
        cypher='RETURN duration({months: 5, days: 1.5}) AS result',
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
        key='expr-temporal1-12-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 3)',
        cypher='RETURN duration({months: 0.75}) AS result',
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
        key='expr-temporal1-12-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 4)',
        cypher='RETURN duration({weeks: 2.5}) AS result',
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
        key='expr-temporal1-12-5',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 5)',
        cypher='RETURN duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70}) AS result',
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
        key='expr-temporal1-12-6',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 6)',
        cypher='RETURN duration({days: 14, seconds: 70, milliseconds: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P14DT1M10.001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-12-7',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 7)',
        cypher='RETURN duration({days: 14, seconds: 70, microseconds: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P14DT1M10.000001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-12-8',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 8)',
        cypher='RETURN duration({days: 14, seconds: 70, nanoseconds: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'P14DT1M10.000000001S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-12-9',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[12] Should construct duration (example 9)',
        cypher='RETURN duration({minutes: 1.5, seconds: 1}) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'PT1M31S'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-13-1',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[13] Should construct temporal with time offset with second precision (example 1)',
        cypher="RETURN time({hour: 12, minute: 34, second: 56, timezone: '+02:05:00'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:34:56+02:05'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-13-2',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[13] Should construct temporal with time offset with second precision (example 2)',
        cypher="RETURN time({hour: 12, minute: 34, second: 56, timezone: '+02:05:59'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:34:56+02:05:59'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-13-3',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[13] Should construct temporal with time offset with second precision (example 3)',
        cypher="RETURN time({hour: 12, minute: 34, second: 56, timezone: '-02:05:07'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'12:34:56-02:05:07'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-temporal1-13-4',
        feature_path='tck/features/expressions/temporal/Temporal1.feature',
        scenario='[13] Should construct temporal with time offset with second precision (example 4)',
        cypher="RETURN datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 34, second: 56, timezone: '+02:05:59'}) AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'1984-10-11T12:34:56+02:05:59'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'temporal', 'meta-xfail', 'xfail'),
    ),
]
