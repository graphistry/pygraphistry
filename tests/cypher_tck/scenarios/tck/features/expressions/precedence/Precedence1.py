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
        key='expr-precedence1-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[1] Exclusive disjunction takes precedence over inclusive disjunction',
        cypher='RETURN true OR true XOR true AS a,\n             true OR (true XOR true) AS b,\n             (true OR true) XOR true AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[2] Conjunction disjunction takes precedence over exclusive disjunction',
        cypher='RETURN true XOR false AND false AS a,\n             true XOR (false AND false) AS b,\n             (true XOR false) AND false AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[3] Conjunction disjunction takes precedence over inclusive disjunction',
        cypher='RETURN true OR false AND false AS a,\n             true OR (false AND false) AS b,\n             (true OR false) AND false AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-4',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[4] Negation takes precedence over conjunction',
        cypher='RETURN NOT true AND false AS a,\n             (NOT true) AND false AS b,\n             NOT (true AND false) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-5',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[5] Negation takes precedence over inclusive disjunction',
        cypher='RETURN NOT false OR true AS a,\n             (NOT false) OR true AS b,\n             NOT (false OR true) AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-6',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[6] Comparison operator takes precedence over boolean negation',
        cypher='RETURN NOT false >= false AS a,\n             NOT (false >= false) AS b,\n             (NOT false) >= false AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-7',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[7] Comparison operator takes precedence over binary boolean operator',
        cypher='RETURN true OR false = false AS a,\n             true OR (false = false) AS b,\n             (true OR false) = false AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-8',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[8] Null predicate takes precedence over comparison operator',
        cypher='RETURN false = true IS NULL AS a,\n             false = (true IS NULL) AS b,\n             (false = true) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-9',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[9] Null predicate takes precedence over negation',
        cypher='RETURN NOT false IS NULL AS a,\n             NOT (false IS NULL) AS b,\n             (NOT false) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-10',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[10] Null predicate takes precedence over boolean operator',
        cypher='RETURN true OR false IS NULL AS a,\n             true OR (false IS NULL) AS b,\n             (true OR false) IS NULL AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'true', 'b': 'true', 'c': 'false'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-11',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[11] List predicate takes precedence over comparison operator',
        cypher='RETURN false = true IN [true, false] AS a,\n             false = (true IN [true, false]) AS b,\n             (false = true) IN [true, false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-12',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[12] List predicate takes precedence over negation',
        cypher='RETURN NOT true IN [true, false] AS a,\n             NOT (true IN [true, false]) AS b,\n             (NOT true) IN [true, false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-13',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[13] List predicate takes precedence over boolean operator',
        cypher='RETURN false AND true IN [true, false] AS a,\n             false AND (true IN [true, false]) AS b,\n             (false AND true) IN [true, false] AS c',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'a': 'false', 'b': 'false', 'c': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-14',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[14] Exclusive disjunction takes precedence over inclusive disjunction in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a OR b XOR c) = (a OR (b XOR c))) AS eq,\n           collect((a OR b XOR c) <> ((a OR b) XOR c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-15',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[15] Conjunction takes precedence over exclusive disjunction in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a XOR b AND c) = (a XOR (b AND c))) AS eq,\n           collect((a XOR b AND c) <> ((a XOR b) AND c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-16',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[16] Conjunction takes precedence over inclusive disjunction in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a OR b AND c) = (a OR (b AND c))) AS eq,\n           collect((a OR b AND c) <> ((a OR b) AND c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-17',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[17] Negation takes precedence over conjunction in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT a AND b) = ((NOT a) AND b)) AS eq,\n           collect((NOT a AND b) <> (NOT (a AND b))) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-18',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[18] Negation takes precedence over inclusive disjunction in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT a OR b) = ((NOT a) OR b)) AS eq,\n           collect((NOT a OR b) <> (NOT (a OR b))) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-20-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[20] Pairs of comparison operators and boolean negation that are associative in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT (a = b)) = ((NOT a) = b)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-20-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[20] Pairs of comparison operators and boolean negation that are associative in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT (a <> b)) = ((NOT a) <> b)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-21-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[21] Comparison operators take precedence over binary boolean operators in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a OR b = c) = (a OR (b = c))) AS eq,\n           collect((a OR b = c) <> ((a OR b) = c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-22-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[22] Pairs of comparison operators and binary boolean operators that are associative in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a XOR (b = c)) = ((a XOR b) = c)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-22-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[22] Pairs of comparison operators and binary boolean operators that are associative in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a OR (b >= c)) = ((a OR b) >= c)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-22-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[22] Pairs of comparison operators and binary boolean operators that are associative in every combination of truth values (example 3)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a AND (b > c)) = ((a AND b) > c)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-22-4',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[22] Pairs of comparison operators and binary boolean operators that are associative in every combination of truth values (example 4)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [true, false, null] AS c\n      WITH collect((a XOR (b <> c)) = ((a XOR b) <> c)) AS eq\n      RETURN all(x IN eq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a = b IS NULL) = (a = (b IS NULL))) AS eq,\n           collect((a = b IS NULL) <> ((a = b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a = b IS NOT NULL) = (a = (b IS NOT NULL))) AS eq,\n           collect((a = b IS NOT NULL) <> ((a = b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 3)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a <= b IS NULL) = (a <= (b IS NULL))) AS eq,\n           collect((a <= b IS NULL) <> ((a <= b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-4',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 4)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a <= b IS NOT NULL) = (a <= (b IS NOT NULL))) AS eq,\n           collect((a <= b IS NOT NULL) <> ((a <= b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-5',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 5)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a >= b IS NULL) = (a >= (b IS NULL))) AS eq,\n           collect((a >= b IS NULL) <> ((a >= b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-6',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 6)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a >= b IS NOT NULL) = (a >= (b IS NOT NULL))) AS eq,\n           collect((a >= b IS NOT NULL) <> ((a >= b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-7',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 7)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a < b IS NULL) = (a < (b IS NULL))) AS eq,\n           collect((a < b IS NULL) <> ((a < b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-8',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 8)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a < b IS NOT NULL) = (a < (b IS NOT NULL))) AS eq,\n           collect((a < b IS NOT NULL) <> ((a < b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-9',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 9)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a > b IS NULL) = (a > (b IS NULL))) AS eq,\n           collect((a > b IS NULL) <> ((a > b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-10',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 10)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a > b IS NOT NULL) = (a > (b IS NOT NULL))) AS eq,\n           collect((a > b IS NOT NULL) <> ((a > b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-11',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 11)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a <> b IS NULL) = (a <> (b IS NULL))) AS eq,\n           collect((a <> b IS NULL) <> ((a <> b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-23-12',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[23] Null predicates take precedence over comparison operators in every combination of truth values (example 12)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a <> b IS NOT NULL) = (a <> (b IS NOT NULL))) AS eq,\n           collect((a <> b IS NOT NULL) <> ((a <> b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-24-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[24] Null predicates take precedence over boolean negation on every truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT a IS NULL) = (NOT (a IS NULL))) AS eq,\n           collect((NOT a IS NULL) <> ((NOT a) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-24-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[24] Null predicates take precedence over boolean negation on every truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((NOT a IS NOT NULL) = (NOT (a IS NOT NULL))) AS eq,\n           collect((NOT a IS NOT NULL) <> ((NOT a) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a OR b IS NULL) = (a OR (b IS NULL))) AS eq,\n           collect((a OR b IS NULL) <> ((a OR b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a OR b IS NOT NULL) = (a OR (b IS NOT NULL))) AS eq,\n           collect((a OR b IS NOT NULL) <> ((a OR b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 3)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a XOR b IS NULL) = (a XOR (b IS NULL))) AS eq,\n           collect((a XOR b IS NULL) <> ((a XOR b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-4',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 4)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a XOR b IS NOT NULL) = (a XOR (b IS NOT NULL))) AS eq,\n           collect((a XOR b IS NOT NULL) <> ((a XOR b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-5',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 5)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a AND b IS NULL) = (a AND (b IS NULL))) AS eq,\n           collect((a AND b IS NULL) <> ((a AND b) IS NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-25-6',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[25] Null predicates take precedence over binary boolean operators in every combination of truth values (example 6)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      WITH collect((a AND b IS NOT NULL) = (a AND (b IS NOT NULL))) AS eq,\n           collect((a AND b IS NOT NULL) <> ((a AND b) IS NOT NULL)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a = b IN c) = (a = (b IN c))) AS eq,\n           collect((a = b IN c) <> ((a = b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a <= b IN c) = (a <= (b IN c))) AS eq,\n           collect((a <= b IN c) <> ((a <= b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 3)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a >= b IN c) = (a >= (b IN c))) AS eq,\n           collect((a >= b IN c) <> ((a >= b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-4',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 4)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a < b IN c) = (a < (b IN c))) AS eq,\n           collect((a < b IN c) <> ((a < b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-5',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 5)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a > b IN c) = (a > (b IN c))) AS eq,\n           collect((a > b IN c) <> ((a > b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-26-6',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[26] List predicate takes precedence over comparison operators in every combination of truth values (example 6)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a <> b IN c) = (a <> (b IN c))) AS eq,\n           collect((a <> b IN c) <> ((a <> b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-27',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[27] List predicate takes precedence over negation in every combination of truth values',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS b\n      WITH collect((NOT a IN b) = (NOT (a IN b))) AS eq,\n           collect((NOT a IN b) <> ((NOT a) IN b)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-28-1',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[28] List predicate takes precedence over binary boolean operators in every combination of truth values (example 1)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a OR b IN c) = (a OR (b IN c))) AS eq,\n           collect((a OR b IN c) <> ((a OR b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-28-2',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[28] List predicate takes precedence over binary boolean operators in every combination of truth values (example 2)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a XOR b IN c) = (a XOR (b IN c))) AS eq,\n           collect((a XOR b IN c) <> ((a XOR b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-precedence1-28-3',
        feature_path='tck/features/expressions/precedence/Precedence1.feature',
        scenario='[28] List predicate takes precedence over binary boolean operators in every combination of truth values (example 3)',
        cypher='UNWIND [true, false, null] AS a\n      UNWIND [true, false, null] AS b\n      UNWIND [[], [true], [false], [null], [true, false], [true, false, null]] AS c\n      WITH collect((a AND b IN c) = (a AND (b IN c))) AS eq,\n           collect((a AND b IN c) <> ((a AND b) IN c)) AS neq\n      RETURN all(x IN eq WHERE x) AND any(x IN neq WHERE x) AS result',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'result': 'true'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'precedence', 'meta-xfail', 'xfail'),
    ),
]
