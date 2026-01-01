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
        key='expr-quantifier11-1',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[1] Any quantifier is always false if the predicate is statically false and the list is not empty',
        cypher="WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH list WHERE size(list) > 0\n      WITH any(x IN list WHERE false) AS result, count(*) AS cnt\n      RETURN result",
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
        key='expr-quantifier11-2',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[2] Any quantifier is always true if the predicate is statically true and the list is not empty',
        cypher="WITH [1, null, true, 4.5, 'abc', false, '', [234, false], {a: null, b: true, c: 15.2}, {}, [], [null], [[{b: [null]}]]] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH list WHERE size(list) > 0\n      WITH any(x IN list WHERE true) AS result, count(*) AS cnt\n      RETURN result",
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
        key='expr-quantifier11-3-1',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[3] Any quantifier is always true if the single or the all quantifier is true (example 1)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH list WHERE single(<operands>) OR all(x IN list WHERE x = 2)\n      WITH any(x IN list WHERE x = 2) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-3-2',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[3] Any quantifier is always true if the single or the all quantifier is true (example 2)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH list WHERE single(<operands>) OR all(x IN list WHERE x % 2 = 0)\n      WITH any(x IN list WHERE x % 2 = 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-3-3',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[3] Any quantifier is always true if the single or the all quantifier is true (example 3)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH list WHERE single(<operands>) OR all(x IN list WHERE x % 3 = 0)\n      WITH any(x IN list WHERE x % 3 = 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-3-4',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[3] Any quantifier is always true if the single or the all quantifier is true (example 4)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH list WHERE single(<operands>) OR all(x IN list WHERE x < 7)\n      WITH any(x IN list WHERE x < 7) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-3-5',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[3] Any quantifier is always true if the single or the all quantifier is true (example 5)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH list WHERE single(<operands>) OR all(x IN list WHERE x >= 3)\n      WITH any(x IN list WHERE x >= 3) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-4-1',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[4] Any quantifier is always equal the boolean negative of the none quantifier (example 1)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT none(x IN list WHERE x = 2)) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-4-2',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[4] Any quantifier is always equal the boolean negative of the none quantifier (example 2)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT none(x IN list WHERE x % 2 = 0)) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-4-3',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[4] Any quantifier is always equal the boolean negative of the none quantifier (example 3)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT none(x IN list WHERE x % 3 = 0)) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-4-4',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[4] Any quantifier is always equal the boolean negative of the none quantifier (example 4)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT none(x IN list WHERE x < 7)) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-4-5',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[4] Any quantifier is always equal the boolean negative of the none quantifier (example 5)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT none(x IN list WHERE x >= 3)) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-5-1',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate (example 1)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT all(x IN list WHERE NOT (x = 2))) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-5-2',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate (example 2)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT all(x IN list WHERE NOT (x % 2 = 0))) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-5-3',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate (example 3)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT all(x IN list WHERE NOT (x % 3 = 0))) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-5-4',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate (example 4)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT all(x IN list WHERE NOT (x < 7))) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-5-5',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[5] Any quantifier is always equal the boolean negative of the all quantifier on the boolean negative of the predicate (example 5)',
        cypher='WITH [1, 2, 3, 4, 5, 6, 7, 8, 9] AS inputList\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH any(x IN list WHERE <predicate>) = (NOT all(x IN list WHERE NOT (x >= 3))) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-6-1',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero (example 1)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH any(x IN list WHERE <predicate>) = (size([x IN list WHERE x = 2 | x]) > 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-6-2',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero (example 2)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH any(x IN list WHERE <predicate>) = (size([x IN list WHERE x % 2 = 0 | x]) > 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-6-3',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero (example 3)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH any(x IN list WHERE <predicate>) = (size([x IN list WHERE x % 3 = 0 | x]) > 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-6-4',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero (example 4)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH any(x IN list WHERE <predicate>) = (size([x IN list WHERE x < 7 | x]) > 0) AS result, count(*) AS cnt\n      RETURN result',
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
        key='expr-quantifier11-6-5',
        feature_path='tck/features/expressions/quantifier/Quantifier11.feature',
        scenario='[6] Any quantifier is always equal whether the size of the list filtered with same the predicate is grater zero (example 5)',
        cypher='UNWIND [{list: [2], fixed: true},\n              {list: [6], fixed: true},\n              {list: [7], fixed: true},\n              {list: [1, 2, 3, 4, 5, 6, 7, 8, 9], fixed: false}] AS input\n      WITH CASE WHEN input.fixed THEN input.list ELSE null END AS fixedList,\n           CASE WHEN NOT input.fixed THEN input.list ELSE [1] END AS inputList\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      UNWIND inputList AS x\n      WITH fixedList, inputList, x, [ y IN inputList WHERE rand() > 0.5 | y] AS list\n      WITH fixedList, inputList, CASE WHEN rand() < 0.5 THEN reverse(list) ELSE list END + x AS list\n      WITH coalesce(fixedList, list) AS list\n      WITH any(x IN list WHERE <predicate>) = (size([x IN list WHERE x >= 3 | x]) > 0) AS result, count(*) AS cnt\n      RETURN result',
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
