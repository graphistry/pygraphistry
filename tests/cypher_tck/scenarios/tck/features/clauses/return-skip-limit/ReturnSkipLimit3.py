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
        key="return-skip-limit3-1",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[1] Get rows in the middle",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP 2\nLIMIT 2",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP/LIMIT and ORDER BY are not supported",
        tags=("return", "skip", "limit", "orderby", "xfail"),
    ),

    Scenario(
        key="return-skip-limit3-2",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[2] Get rows in the middle by param",
        cypher="MATCH (n)\nRETURN n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
              ({name: 'B'}),
              ({name: 'C'}),
              ({name: 'D'}),
              ({name: 'E'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "({name: 'C'})"},
                {"n": "({name: 'D'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="SKIP/LIMIT, ORDER BY, and parameter binding are not supported",
        tags=("return", "skip", "limit", "orderby", "params", "xfail"),
    ),

    Scenario(
        key="return-skip-limit3-3",
        feature_path="tck/features/clauses/return-skip-limit/ReturnSkipLimit3.feature",
        scenario="[3] Limiting amount of rows when there are fewer left than the LIMIT argument",
        cypher="MATCH (a)\nRETURN a.count\nORDER BY a.count\nSKIP 10\nLIMIT 10",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "count": i} for i in range(16)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"a.count": 10},
                {"a.count": 11},
                {"a.count": 12},
                {"a.count": 13},
                {"a.count": 14},
                {"a.count": 15},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("return", "skip", "limit", "orderby", "unwind", "xfail"),
    ),
]
