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
        key='expr-boolean5-1',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[1] Disjunction is distributive over conjunction on non-null',
        cypher='UNWIND [true, false] AS a\n      UNWIND [true, false] AS b\n      UNWIND [true, false] AS c\n      RETURN a, b, c, (a OR (b AND c)) = ((a OR b) AND (a OR c)) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'false', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-2',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[2] Disjunction is distributive over conjunction on null',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH a, b, c WHERE a IS NULL OR b IS NULL OR c IS NULL\n      RETURN a, b, c, (a OR (b AND c)) IS NULL = ((a OR b) AND (a OR c)) IS NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'null', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-3',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[3] Conjunction is distributive over disjunction on non-null',
        cypher='UNWIND [true, false] AS a\n      UNWIND [true, false] AS b\n      UNWIND [true, false] AS c\n      RETURN a, b, c, (a AND (b OR c)) = ((a AND b) OR (a AND c)) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'false', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-4',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[4] Conjunction is distributive over disjunction on null',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH a, b, c WHERE a IS NULL OR b IS NULL OR c IS NULL\n      RETURN a, b, c, (a AND (b OR c)) IS NULL = ((a AND b) OR (a AND c)) IS NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'null', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-5',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[5] Conjunction is distributive over exclusive disjunction on non-null',
        cypher='UNWIND [true, false] AS a\n      UNWIND [true, false] AS b\n      UNWIND [true, false] AS c\n      RETURN a, b, c, (a AND (b XOR c)) = ((a AND b) XOR (a AND c)) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'false', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-6',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario='[6] Conjunction is not distributive over exclusive disjunction on null',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH a, b, c WHERE a IS NULL OR b IS NULL OR c IS NULL\n      RETURN a, b, c, (a AND (b XOR c)) IS NULL = ((a AND b) XOR (a AND c)) IS NULL AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'true', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'null', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'true', 'result': 'false'},
            {'a': 'null', 'b': 'true', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'true', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'false', 'c': 'null', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'true', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'false', 'result': 'true'},
            {'a': 'null', 'b': 'null', 'c': 'null', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-7',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario="[7] De Morgan's law on non-null: the negation of a disjunction is the conjunction of the negations",
        cypher='UNWIND [true, false] AS a\n      UNWIND [true, false] AS b\n      RETURN a, b, NOT (a OR b) = (NOT (a) AND NOT (b)) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-boolean5-8',
        feature_path='tck/features/expressions/boolean/Boolean5.feature',
        scenario="[8] De Morgan's law on non-null: the negation of a conjunction is the disjunction of the negations",
        cypher='UNWIND [true, false] AS a\n      UNWIND [true, false] AS b\n      RETURN a, b, NOT (a AND b) = (NOT (a) OR NOT (b)) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'result': 'true'},
            {'a': 'true', 'b': 'false', 'result': 'true'},
            {'a': 'false', 'b': 'true', 'result': 'true'},
            {'a': 'false', 'b': 'false', 'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'boolean', 'meta-xfail', 'xfail'),
    ),
]
