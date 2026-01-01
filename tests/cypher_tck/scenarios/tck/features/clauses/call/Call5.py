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
        key="call5-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[1] Explicit procedure result projection",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN out",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[2] Explicit procedure result projection with RETURN *",
        cypher="CALL test.my.proc(null) YIELD out\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"out": "'nix'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-3-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[3] The order of yield items is irrelevant (example=1, yield=a, b)",
        cypher="CALL test.my.proc(null) YIELD a, b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-3-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[3] The order of yield items is irrelevant (example=2, yield=b, a)",
        cypher="CALL test.my.proc(null) YIELD b, a\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-1",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=1, rename=a AS c, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS d\nRETURN c, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-2",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=2, rename=a AS b, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS b, b AS d\nRETURN b, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"b": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-3",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=3, rename=a AS c, b AS a)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS a\nRETURN c, a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "a": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-4",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=4, rename=a AS b, b AS a)",
        cypher="CALL test.my.proc(null) YIELD a AS b, b AS a\nRETURN b, a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"b": 1, "a": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-5",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=5, rename=a AS c, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS b\nRETURN c, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-6",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=6, rename=a AS c, b)",
        cypher="CALL test.my.proc(null) YIELD a AS c, b\nRETURN c, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"c": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-7",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=7, rename=a AS a, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b AS d\nRETURN a, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-8",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=8, rename=a, b AS d)",
        cypher="CALL test.my.proc(null) YIELD a, b AS d\nRETURN a, d",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "d": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-9",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=9, rename=a AS a, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b AS b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-10",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=10, rename=a AS a, b)",
        cypher="CALL test.my.proc(null) YIELD a AS a, b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-4-11",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[4] Rename outputs to unbound variable names (example=11, rename=a, b AS b)",
        cypher="CALL test.my.proc(null) YIELD a, b AS b\nRETURN a, b",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"a": 1, "b": 2},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures and YIELD projections are not supported",
        tags=("call", "xfail"),
    ),

    Scenario(
        key="call5-5",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[5] Fail on renaming to an already bound variable name",
        cypher="CALL test.my.proc(null) YIELD a, b AS a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call5-6",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[6] Fail on renaming all outputs to the same variable name",
        cypher="CALL test.my.proc(null) YIELD a AS c, b AS c\nRETURN c",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call5-7",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[7] Fail on in-query call to procedure with YIELD *",
        cypher="CALL test.my.proc('Stefan', 1) YIELD *\nRETURN city, country_code",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="CALL procedure validation is not supported",
        tags=("call", "syntax-error", "xfail"),
    ),

    Scenario(
        key="call5-8",
        feature_path="tck/features/clauses/call/Call5.feature",
        scenario="[8] Allow standalone call to procedure with YIELD *",
        cypher="CALL test.my.proc('Stefan', 1) YIELD *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"city": "'Berlin'", "country_code": 49},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="CALL procedures are not supported",
        tags=("call", "xfail"),
    ),
]
