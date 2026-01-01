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
        key='expr-map1-1',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[1] Statically access a field of a non-null map',
        cypher='WITH {existing: 42, notMissing: null} AS m\n      RETURN m.missing, m.notMissing, m.existing',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'m.missing': 'null', 'm.notMissing': 'null', 'm.existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-2',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[2] Statically access a field of a null map',
        cypher='WITH null AS m\n      RETURN m.missing',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'m.missing': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-3',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[3] Statically access a field of a map resulting from an expression',
        cypher='WITH [123, {existing: 42, notMissing: null}] AS list\n      RETURN (list[1]).missing, (list[1]).notMissing, (list[1]).existing',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'(list[1]).missing': 'null', '(list[1]).notMissing': 'null', '(list[1]).existing': 42}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-4-1',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[4] Statically access a field is case-sensitive (example 1)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map.name AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Mats'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-4-2',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[4] Statically access a field is case-sensitive (example 2)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map.name AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Mats'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-4-3',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[4] Statically access a field is case-sensitive (example 3)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map.Name AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Pontus'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-4-4',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[4] Statically access a field is case-sensitive (example 4)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map.nAMe AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-1',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 1)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map.`name` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Mats'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-2',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 2)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map.`nome` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Pontus'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-3',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 3)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map.`Mats` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-4',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 4)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map.`null` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-5',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 5)',
        cypher="WITH {null: 'Mats', NULL: 'Pontus'} AS map\n      RETURN map.`null` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Mats'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-5-6',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[5] Statically access a field with a delimited identifier (example 6)',
        cypher="WITH {null: 'Mats', NULL: 'Pontus'} AS map\n      RETURN map.`NULL` AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'Pontus'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-1',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 1)',
        cypher='WITH 123 AS nonMap\n      RETURN nonMap.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-2',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 2)',
        cypher='WITH 42.45 AS nonMap\n      RETURN nonMap.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-3',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 3)',
        cypher='WITH true AS nonMap\n      RETURN nonMap.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-4',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 4)',
        cypher='WITH false AS nonMap\n      RETURN nonMap.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-5',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 5)',
        cypher="WITH 'string' AS nonMap\n      RETURN nonMap.num",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map1-6-6',
        feature_path='tck/features/expressions/map/Map1.feature',
        scenario='[6] Fail when performing property access on a non-map (example 6)',
        cypher='WITH [123, true] AS nonMap\n      RETURN nonMap.num',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'runtime-error', 'xfail'),
    ),
]
