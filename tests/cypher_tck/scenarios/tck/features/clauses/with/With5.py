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
        key="with5-1",
        feature_path="tck/features/clauses/with/With5.feature",
        scenario="[1] DISTINCT on an expression",
        cypher="MATCH (a)\nWITH DISTINCT a.name AS name\nRETURN name",
        graph=graph_fixture_from_create(
            """
            CREATE ({name: 'A'}),
                   ({name: 'A'}),
                   ({name: 'B'})
            """
        ),
        expected=Expected(
            rows=[
                {"name": "'A'"},
                {"name": "'B'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections are not supported",
        tags=("with", "distinct", "xfail"),
    ),

    Scenario(
        key="with5-2",
        feature_path="tck/features/clauses/with/With5.feature",
        scenario="[2] Handling DISTINCT with lists in maps",
        cypher="MATCH (n)\nWITH DISTINCT {name: n.list} AS map\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE ({list: ['A', 'B']}), ({list: ['A', 'B']})
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH DISTINCT projections and aggregations are not supported",
        tags=("with", "distinct", "aggregation", "xfail"),
    ),
]
