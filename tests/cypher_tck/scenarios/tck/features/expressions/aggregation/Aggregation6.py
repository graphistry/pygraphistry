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
        key='expr-aggregation6-1-1',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[1] `percentileDisc()` (example 1)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 10.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-1-2',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[1] `percentileDisc()` (example 2)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 20.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-1-3',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[1] `percentileDisc()` (example 3)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 30.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-2-1',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[2] `percentileCont()` (example 1)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 10.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-2-2',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[2] `percentileCont()` (example 2)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 20.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-2-3',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[2] `percentileCont()` (example 3)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $percentile) AS p',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0}),
                         ({price: 20.0}),
                         ({price: 30.0})
            """
        ),
        expected=Expected(
            rows=[
            {'p': 30.0}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-3-1',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[3] `percentileCont()` failing on bad arguments (example 1)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-3-2',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[3] `percentileCont()` failing on bad arguments (example 2)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-3-3',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[3] `percentileCont()` failing on bad arguments (example 3)',
        cypher='MATCH (n)\n      RETURN percentileCont(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-4-1',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[4] `percentileDisc()` failing on bad arguments (example 1)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-4-2',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[4] `percentileDisc()` failing on bad arguments (example 2)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-4-3',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[4] `percentileDisc()` failing on bad arguments (example 3)',
        cypher='MATCH (n)\n      RETURN percentileDisc(n.price, $param)',
        graph=graph_fixture_from_create(
            """
            CREATE ({price: 10.0})
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-aggregation6-5',
        feature_path='tck/features/expressions/aggregation/Aggregation6.feature',
        scenario='[5] `percentileDisc()` failing in more involved query',
        cypher='MATCH (n:S)\n      WITH n, size([(n)-->() | 1]) AS deg\n      WHERE deg > 2\n      WITH deg\n      LIMIT 100\n      RETURN percentileDisc(0.90, deg), deg',
        graph=graph_fixture_from_create(
            """
            UNWIND range(0, 10) AS i
                  CREATE (s:S)
                  WITH s, i
                  UNWIND range(0, i) AS j
                  CREATE (s)-[:REL]->()
            """
        ),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'aggregation', 'meta-xfail', 'xfail'),
    ),
]
