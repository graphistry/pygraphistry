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
        key='expr-quantifier10-1',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[1] Single quantifier is always false if the predicate is statically false and the list is not empty',
        cypher="WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH list WHERE size(list) > 0\n      WITH single(x IN list WHERE false) AS result, count(*) AS cnt\n      RETURN result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-2',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[2] Single quantifier is always false if the predicate is statically true and the list has more than one element',
        cypher="WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH list WHERE size(list) > 1\n      WITH single(x IN list WHERE true) AS result, count(*) AS cnt\n      RETURN result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-3',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[3] Single quantifier is always true if the predicate is statically true and the list has exactly one non-null element',
        cypher="WITH [1, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n      UNWIND inputList AS element\n      WITH single(x IN [element] WHERE true) AS result, count(*) AS cnt\n      RETURN result",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-4-1',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one (example 1)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH single(x IN list WHERE <predicate>) = (size([x IN list WHERE x = 2 | x]) = 1) AS result, count(*) AS cnt\n      RETURN result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-4-2',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one (example 2)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH single(x IN list WHERE <predicate>) = (size([x IN list WHERE x % 2 = 0 | x]) = 1) AS result, count(*) AS cnt\n      RETURN result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-4-3',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one (example 3)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH single(x IN list WHERE <predicate>) = (size([x IN list WHERE x % 3 = 0 | x]) = 1) AS result, count(*) AS cnt\n      RETURN result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-4-4',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one (example 4)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH single(x IN list WHERE <predicate>) = (size([x IN list WHERE x < 7 | x]) = 1) AS result, count(*) AS cnt\n      RETURN result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-quantifier10-4-5',
        feature_path='tck/features/expressions/quantifier/Quantifier10.feature',
        scenario='[4] Single quantifier is always equal whether the size of the list filtered with same the predicate is one (example 5)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH single(x IN list WHERE <predicate>) = (size([x IN list WHERE x >= 3 | x]) = 1) AS result, count(*) AS cnt\n      RETURN result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'quantifier', 'meta-xfail', 'xfail'),
    ),
]
