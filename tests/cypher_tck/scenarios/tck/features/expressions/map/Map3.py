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
        key='expr-map3-1',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[1] Using `keys()` on a literal map',
        cypher="RETURN keys({name: 'Alice', age: 38, address: {city: 'London', residential: true}}) AS k",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'k': "['name', 'age', 'address']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-2',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[2] Using `keys()` on a parameter map',
        cypher='RETURN keys($param) AS k',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'k': "['address', 'name', 'age']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-map3-3',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[3] Using `keys()` on null map',
        cypher='WITH null AS m\n      RETURN keys(m), keys(null)',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys(m)': 'null', 'keys(null)': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-1',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 1)',
        cypher='RETURN keys({}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': '[]'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-2',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 2)',
        cypher='RETURN keys({k: 1}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-3',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 3)',
        cypher='RETURN keys({k: null}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-4',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 4)',
        cypher='RETURN keys({k: null, l: 1}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k', 'l']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-5',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 5)',
        cypher='RETURN keys({k: 1, l: null}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k', 'l']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-6',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 6)',
        cypher='RETURN keys({k: null, l: null}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k', 'l']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-4-7',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[4] Using `keys()` on map with null values (example 7)',
        cypher='RETURN keys({k: 1, l: null, m: 1}) AS keys',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'keys': "['k', 'l', 'm']"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map3-5',
        feature_path='tck/features/expressions/map/Map3.feature',
        scenario='[5] Using `keys()` and `IN` to check field existence',
        cypher="WITH {exists: 42, notMissing: null} AS map\n      RETURN 'exists' IN keys(map) AS a,\n             'notMissing' IN keys(map) AS b,\n             'missing' IN keys(map) AS c",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),
]
