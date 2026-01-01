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
        key="call1-1",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[1] Standalone call to procedure that takes no arguments and yields no results",
        cypher="CALL test.doNothing()",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-2",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[2] Standalone call to procedure that takes no arguments and yields no results, called with implicit arguments",
        cypher="CALL test.doNothing",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-3",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[3] In-query call to procedure that takes no arguments and yields no results",
        cypher="MATCH (n)\nCALL test.doNothing()\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-4",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[4] In-query call to procedure that takes no arguments and yields no results and consumes no rows",
        cypher="MATCH (n)\nCALL test.doNothing()\nRETURN n.name AS `name`",
        graph=graph_fixture_from_create(
            """
            CREATE (:A {name: 'a'})
            CREATE (:B {name: 'b'})
            CREATE (:C {name: 'c'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'a'"},
                {"name": "'b'"},
                {"name": "'c'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-5",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[5] Standalone call to STRING procedure that takes no arguments",
        cypher="CALL test.labels()",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"label": "'A'"},
                {"label": "'B'"},
                {"label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-6",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[6] In-query call to STRING procedure that takes no arguments",
        cypher="CALL test.labels() YIELD label\nRETURN label",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"label": "'A'"},
                {"label": "'B'"},
                {"label": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call1-7",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[7] Standalone call to procedure should fail if explicit argument is missing",
        cypher="CALL test.my.proc('Dobby')",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-8",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[8] In-query call to procedure should fail if explicit argument is missing",
        cypher="CALL test.my.proc('Dobby') YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-9",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[9] Standalone call to procedure should fail if too many explicit argument are given",
        cypher="CALL test.my.proc(1, 2, 3, 4)",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-10",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[10] In-query call to procedure should fail if too many explicit argument are given",
        cypher="CALL test.my.proc(1, 2, 3, 4) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-11",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[11] Standalone call to procedure should fail if implicit argument is missing",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedures and parameter binding are not supported",
        tags=("call", "params", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-12",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[12] In-query call to procedure that has outputs fails if no outputs are yielded",
        cypher="CALL test.my.proc(1)\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call1-13",
        feature_path="tck/features/clauses/call/Call1.feature",
        scenario="[13] Standalone call to unknown procedure should fail",
        cypher="CALL test.my.proc",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure registry is not supported",
        tags=("call", "procedure-error", "xfail"),
    ),
]
