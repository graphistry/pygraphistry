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
        key='expr-map2-1',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[1] Dynamically access a field based on parameters when there is no type information',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[idx] AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-map2-2',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[2] Dynamically access a field based on parameters when there is rhs type information',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[toString(idx)] AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-map2-3',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[3] Dynamically access a field on null results in null',
        cypher="WITH null AS expr, 'x' AS idx\n      RETURN expr[idx] AS value",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map2-4',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[4] Dynamically access a field with null results in null',
        cypher="WITH {name: 'Mats'} AS expr, null AS idx\n      RETURN expr[idx] AS value",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'map', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-map2-5-1',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 1)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map['name'] AS result",
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
        key='expr-map2-5-2',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 2)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map['name'] AS result",
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
        key='expr-map2-5-3',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 3)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map['Name'] AS result",
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
        key='expr-map2-5-4',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 4)',
        cypher="WITH {name: 'Mats', Name: 'Pontus'} AS map\n      RETURN map['nAMe'] AS result",
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
        key='expr-map2-5-5',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 5)',
        cypher="WITH {name: 'Mats', nome: 'Pontus'} AS map\n      RETURN map['null'] AS result",
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
        key='expr-map2-5-6',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 6)',
        cypher="WITH {null: 'Mats', NULL: 'Pontus'} AS map\n      RETURN map['null'] AS result",
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
        key='expr-map2-5-7',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[5] Dynamically access a field is case-sensitive (example 7)',
        cypher="WITH {null: 'Mats', NULL: 'Pontus'} AS map\n      RETURN map['NULL'] AS result",
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
        key='expr-map2-6',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[6] Fail at runtime when attempting to index with an Int into a Map',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map2-7',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[7] Fail at runtime when trying to index into a map with a non-string',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-map2-8',
        feature_path='tck/features/expressions/map/Map2.feature',
        scenario='[8] Fail at runtime when trying to index something which is not a map',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'map', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),
]
