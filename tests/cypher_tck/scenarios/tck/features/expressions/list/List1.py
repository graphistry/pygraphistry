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
        key='expr-list1-1',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[1] Indexing into literal list',
        cypher='RETURN [1, 2, 3][0] AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list1-2',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[2] Indexing into nested literal lists',
        cypher='RETURN [[1]][0][0]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'[[1]][0][0]': 1}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'list', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-list1-3',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[3] Use list lookup based on parameters when there is no type information',
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
        tags=('expr', 'list', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-list1-4',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[4] Use list lookup based on parameters when there is lhs type information',
        cypher="WITH ['Apa'] AS expr\n      RETURN expr[$idx] AS value",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-list1-5',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[5] Use list lookup based on parameters when there is rhs type information',
        cypher='WITH $expr AS expr, $idx AS idx\n      RETURN expr[toInteger(idx)] AS value',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
            {'value': "'Apa'"}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Parameter binding is not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'xfail'),
    ),

    Scenario(
        key='expr-list1-6-1',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[6] Fail when indexing a non-list #Example: <exampleName> (example 1)',
        cypher='WITH true AS list, 0 AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-6-2',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[6] Fail when indexing a non-list #Example: <exampleName> (example 2)',
        cypher='WITH 123 AS list, 0 AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-6-3',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[6] Fail when indexing a non-list #Example: <exampleName> (example 3)',
        cypher='WITH 4.7 AS list, 0 AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-6-4',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[6] Fail when indexing a non-list #Example: <exampleName> (example 4)',
        cypher="WITH '1' AS list, 0 AS idx\n      RETURN list[idx]",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-7-1',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[7] Fail when indexing a non-list given by a parameter #Example: <exampleName> (example 1)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-7-2',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[7] Fail when indexing a non-list given by a parameter #Example: <exampleName> (example 2)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-7-3',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[7] Fail when indexing a non-list given by a parameter #Example: <exampleName> (example 3)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-7-4',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[7] Fail when indexing a non-list given by a parameter #Example: <exampleName> (example 4)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-8-1',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[8] Fail when indexing with a non-integer #Example: <exampleName> (example 1)',
        cypher='WITH [1, 2, 3, 4, 5] AS list, true AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-8-2',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[8] Fail when indexing with a non-integer #Example: <exampleName> (example 2)',
        cypher='WITH [1, 2, 3, 4, 5] AS list, 4.7 AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-8-3',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[8] Fail when indexing with a non-integer #Example: <exampleName> (example 3)',
        cypher="WITH [1, 2, 3, 4, 5] AS list, '1' AS idx\n      RETURN list[idx]",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-8-4',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[8] Fail when indexing with a non-integer #Example: <exampleName> (example 4)',
        cypher='WITH [1, 2, 3, 4, 5] AS list, [1] AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-8-5',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[8] Fail when indexing with a non-integer #Example: <exampleName> (example 5)',
        cypher='WITH [1, 2, 3, 4, 5] AS list, {x: 3} AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-9-1',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[9] Fail when indexing with a non-integer given by a parameter #Example: <exampleName> (example 1)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-9-2',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[9] Fail when indexing with a non-integer given by a parameter #Example: <exampleName> (example 2)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-9-3',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[9] Fail when indexing with a non-integer given by a parameter #Example: <exampleName> (example 3)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-9-4',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[9] Fail when indexing with a non-integer given by a parameter #Example: <exampleName> (example 4)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),

    Scenario(
        key='expr-list1-9-5',
        feature_path='tck/features/expressions/list/List1.feature',
        scenario='[9] Fail when indexing with a non-integer given by a parameter #Example: <exampleName> (example 5)',
        cypher='WITH $expr AS list, $idx AS idx\n      RETURN list[idx]',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Runtime error semantics are not supported',
        tags=('expr', 'list', 'meta-xfail', 'params', 'runtime-error', 'xfail'),
    ),
]
