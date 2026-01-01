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
        key="with1-1",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[1] Forwarind a node variable 1",
        cypher="MATCH (a:A)\nWITH a\nMATCH (a)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and RETURN * projections are not supported",
        tags=("with", "pipeline", "return-star", "xfail"),
    ),

    Scenario(
        key="with1-2",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[2] Forwarind a node variable 2",
        cypher="MATCH (a:A)\nWITH a\nMATCH (x:X), (a)-->(b)\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            CREATE (:X)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)", "x": "(:X)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, cartesian products, and RETURN * projections are not supported",
        tags=("with", "pipeline", "cartesian", "return-star", "xfail"),
    ),

    Scenario(
        key="with1-3",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[3] Forwarding a relationship variable",
        cypher="MATCH ()-[r1]->(:X)\nWITH r1 AS r2\nMATCH ()-[r2]->()\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(:X),
                   ()-[:T2]->(:X),
                   ()-[:T3]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and relationship variable aliasing are not supported",
        tags=("with", "relationship", "alias", "xfail"),
    ),

    Scenario(
        key="with1-4",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[4] Forwarding a path variable",
        cypher="MATCH p = (a)\nWITH p\nRETURN p",
        graph=graph_fixture_from_create(
            """
            CREATE ()
            """
        ),
        expected=Expected(
            rows=[
                {"p": "<()>"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and path variables are not supported",
        tags=("with", "path", "xfail"),
    ),

    Scenario(
        key="with1-5",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[5] Forwarding null",
        cypher="OPTIONAL MATCH (a:Start)\nWITH a\nMATCH (a)-->(b)\nRETURN *",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(rows=[]),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and WITH pipelines are not supported",
        tags=("with", "optional-match", "null", "xfail"),
    ),

    Scenario(
        key="with1-6",
        feature_path="tck/features/clauses/with/With1.feature",
        scenario="[6] Forwarding a node variable possibly null",
        cypher="OPTIONAL MATCH (a:A)\nWITH a AS a\nMATCH (b:B)\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Single), (a:A {num: 42}),
                   (b:B {num: 46}), (c:C)
            CREATE (s)-[:REL]->(a),
                   (s)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (b)-[:LOOP]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A {num: 42})", "b": "(:B {num: 46})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics and WITH pipelines are not supported",
        tags=("with", "optional-match", "xfail"),
    ),
]
