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
        key='expr-null3-1',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[1] The inverse of a null is a null',
        cypher='RETURN NOT null AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null3-2',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[2] It is unknown - i.e. null - if a null is equal to a null',
        cypher='RETURN null = null AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null3-3',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[3] It is unknown - i.e. null - if a null is not equal to a null',
        cypher='RETURN null <> null AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'null', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-1',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 1)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-2',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 2)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-3',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 3)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-4',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 4)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-5',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 5)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-6',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 6)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-null3-4-7',
        feature_path='tck/features/expressions/null/Null3.feature',
        scenario='[4] Using null in IN (example 7)',
        cypher='RETURN $elt IN $coll AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'null'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'null', 'meta-xfail', 'params', 'xfail'),
    ),
]
