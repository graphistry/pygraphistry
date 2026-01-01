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
        key="with-where7-1",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[1] WHERE sees a variable bound before but not after WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE a.name2 = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and variable scoping rules are not supported",
        tags=("with", "where", "xfail"),
    ),

    Scenario(
        key="with-where7-2",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[2] WHERE sees a variable bound after but not before WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE name = 'B'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and variable scoping rules are not supported",
        tags=("with", "where", "xfail"),
    ),

    Scenario(
        key="with-where7-3",
        feature_path="tck/features/clauses/with-where/WithWhere7.feature",
        scenario="[3] WHERE sees both, variable bound before but not after WITH and variable bound after but not before WITH",
        cypher="MATCH (a)\nWITH a.name2 AS name\nWHERE name = 'B' OR a.name2 = 'C'\nRETURN *",
        graph=graph_fixture_from_create(
            """
            CREATE ({name2: 'A'}),
                   ({name2: 'B'}),
                   ({name2: 'C'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'B'"},
                {"name": "'C'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, variable scoping rules, and OR predicates are not supported",
        tags=("with", "where", "or", "xfail"),
    ),
]
