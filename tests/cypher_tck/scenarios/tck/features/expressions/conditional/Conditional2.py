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
        key='expr-conditional2-1-1',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 1)',
        cypher="RETURN CASE -10\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'minus ten'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-2',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 2)',
        cypher="RETURN CASE 0\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'zero'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-3',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 3)',
        cypher="RETURN CASE 1\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'one'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-4',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 4)',
        cypher="RETURN CASE 5\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'five'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-5',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 5)',
        cypher="RETURN CASE 10\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'ten'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-6',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 6)',
        cypher="RETURN CASE 3000\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'three thousand'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-7',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 7)',
        cypher="RETURN CASE -30\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-8',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 8)',
        cypher="RETURN CASE 3\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-9',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 9)',
        cypher="RETURN CASE 3001\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-10',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 10)',
        cypher="RETURN CASE '0'\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-11',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 11)',
        cypher="RETURN CASE true\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-conditional2-1-12',
        feature_path='tck/features/expressions/conditional/Conditional2.feature',
        scenario='[1] Simple cases over integers (example 12)',
        cypher="RETURN CASE 10.1\n          WHEN -10 THEN 'minus ten'\n          WHEN 0 THEN 'zero'\n          WHEN 1 THEN 'one'\n          WHEN 5 THEN 'five'\n          WHEN 10 THEN 'ten'\n          WHEN 3000 THEN 'three thousand'\n          ELSE 'something else'\n        END AS result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': "'something else'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'conditional', 'meta-xfail', 'xfail'),
    ),
]
