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
        key='expr-pattern1-1',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[1] Matching on any single outgoing directed connection',
        cypher='MATCH (n) WHERE (n)-[]->() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-2',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[2] Matching on a single undirected connection',
        cypher='MATCH (n) WHERE (n)-[]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'},
            {'n': '(:C)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-3',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[3] Matching on any single incoming directed connection',
        cypher='MATCH (n) WHERE (n)<-[]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'},
            {'n': '(:C)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-4',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[4] Matching on a specific type of single outgoing directed connection',
        cypher='MATCH (n) WHERE (n)-[:REL1]->() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-5',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[5] Matching on a specific type of single undirected connection',
        cypher='MATCH (n) WHERE (n)-[:REL1]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-6',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[6] Matching on a specific type of single incoming directed connection',
        cypher='MATCH (n) WHERE (n)<-[:REL1]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-7',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[7] Matching on a specific type of a variable length outgoing directed connection',
        cypher='MATCH (n) WHERE (n)-[:REL1*]->() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-8',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[8] Matching on a specific type of variable length undirected connection',
        cypher='MATCH (n) WHERE (n)-[:REL1*]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-9',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[9] Matching on a specific type of variable length incoming directed connection',
        cypher='MATCH (n) WHERE (n)<-[:REL1*]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Matching on a specific type of undirected connection with length 2',
        cypher='MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-1',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 1)',
        cypher='MATCH (n) WHERE (a) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-2',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 2)',
        cypher='MATCH (n) WHERE (n)-[r]->(a) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-3',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 3)',
        cypher='MATCH (n) WHERE (a)-[r]->(n) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-4',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 4)',
        cypher='MATCH (n) WHERE (n)<-[r {}]-(a) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-5',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 5)',
        cypher='MATCH (n) WHERE (n)-[r {}]-(a) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-6',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 6)',
        cypher='MATCH (n) WHERE (n)-[r]->() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-7',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 7)',
        cypher='MATCH (n) WHERE ()-[r]->(n) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-8',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 8)',
        cypher='MATCH (n) WHERE (n)<-[r]-() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-9',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 9)',
        cypher='MATCH (n) WHERE (n)-[r]-() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-10',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 10)',
        cypher='MATCH (n) WHERE ()-[r]->() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-11',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 11)',
        cypher='MATCH (n) WHERE ()<-[r]-() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-12',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 12)',
        cypher='MATCH (n) WHERE ()-[r]-() RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-13',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 13)',
        cypher='MATCH (n) WHERE (n)-[r:REL]->(a {num: 5}) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-14',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 14)',
        cypher='MATCH (n) WHERE (n)-[r:REL*0..2]->(a {num: 5}) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-10-15',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[10] Fail on introducing unbounded variables in pattern (example 15)',
        cypher='MATCH (n) WHERE (n)-[r:REL]->(:C)<-[s:REL]-(a {num: 5}) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-11',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[11] Fail on checking self pattern',
        cypher='MATCH (n) WHERE (n) RETURN n',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-12',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[12] Matching two nodes on a single directed connection between them',
        cypher='MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:B)', 'm': '(:A)'},
            {'n': '(:A)', 'm': '(:C)'},
            {'n': '(:A)', 'm': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-13',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[13] Fail on matching two nodes on a single undirected connection between them',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:B)', 'm': '(:A)'},
            {'n': '(:A)', 'm': '(:C)'},
            {'n': '(:A)', 'm': '(:D)'},
            {'n': '(:C)', 'm': '(:A)'},
            {'n': '(:D)', 'm': '(:A)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-14',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[14] Matching two nodes on a specific type of single outgoing directed connection',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:A)', 'm': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-15',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[15] Matching two nodes on a specific type of single undirected connection',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1]-(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:B)', 'm': '(:A)'},
            {'n': '(:A)', 'm': '(:D)'},
            {'n': '(:D)', 'm': '(:A)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-16',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[16] Matching two nodes on a specific type of a variable length outgoing directed connection',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:A)', 'm': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-17',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[17] Matching two nodes on a specific type of variable length undirected connection',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)', 'm': '(:B)'},
            {'n': '(:A)', 'm': '(:D)'},
            {'n': '(:B)', 'm': '(:A)'},
            {'n': '(:B)', 'm': '(:D)'},
            {'n': '(:D)', 'm': '(:A)'},
            {'n': '(:D)', 'm': '(:B)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-18',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[18] Matching two nodes on a specific type of undirected connection with length 2',
        cypher='MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:D)', 'm': '(:B)'},
            {'n': '(:B)', 'm': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-19',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[19] Using a negated existential pattern predicate',
        cypher='MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:C)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-20',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[20] Using two existential pattern predicates in a conjunction',
        cypher='MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-21',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[21] Using two existential pattern predicates in a disjunction',
        cypher='MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n',
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL1]->(b:B), (b)-[:REL2]->(a), (a)-[:REL3]->(:C), (a)-[:REL1]->(:D)
            """
        ),
        expected=Expected(
            rows=[
            {'n': '(:A)'},
            {'n': '(:B)'},
            {'n': '(:D)'}
            ],
        ),
        gfql=None,
        status="xfail",
        reason='Expression evaluation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-22',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[22] Fail on using pattern in RETURN projection',
        cypher='MATCH (n) RETURN (n)-[]->()',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-23',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[23] Fail on using pattern in WITH projection',
        cypher='MATCH (n) WITH (n)-[]->() AS x RETURN x',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),

    Scenario(
        key='expr-pattern1-24',
        feature_path='tck/features/expressions/pattern/Pattern1.feature',
        scenario='[24] Fail on using pattern in right-hand side of SET',
        cypher='MATCH (n) SET n.prop = head(nodes(head((n)-[:REL]->()))).foo',
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason='Compile-time validation is not supported',
        tags=('expr', 'pattern', 'meta-xfail', 'syntax-error', 'xfail'),
    ),
]
