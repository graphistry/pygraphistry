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
        key="with-skip-limit3-1",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[1] Get rows in the middle",
        cypher="MATCH (n)\nWITH n\nORDER BY n.name ASC\nSKIP 2\nLIMIT 2\nRETURN n",
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
        reason="WITH pipelines, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("with", "skip", "limit", "orderby", "xfail"),
    ),

    Scenario(
        key="with-skip-limit3-2",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[2] Get rows in the middle by param",
        cypher="MATCH (n)\nWITH n\nORDER BY n.name ASC\nSKIP $s\nLIMIT $l\nRETURN n",
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
        reason="WITH pipelines, SKIP/LIMIT, ORDER BY, and parameter binding are not supported",
        tags=("with", "skip", "limit", "orderby", "params", "xfail"),
    ),

    Scenario(
        key="with-skip-limit3-3",
        feature_path="tck/features/clauses/with-skip-limit/WithSkipLimit3.feature",
        scenario="[3] Limiting amount of rows when there are fewer left than the LIMIT argument",
        cypher="MATCH (a)\nWITH a.count AS count\n  ORDER BY a.count\n  SKIP 10\n  LIMIT 10\nRETURN count",
        graph=GraphFixture(
            nodes=[{"id": f"n{i}", "labels": [], "count": i} for i in range(16)],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 10},
                {"count": 11},
                {"count": 12},
                {"count": 13},
                {"count": 14},
                {"count": 15},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, SKIP/LIMIT, and ORDER BY are not supported",
        tags=("with", "skip", "limit", "orderby", "xfail"),
    ),
]
