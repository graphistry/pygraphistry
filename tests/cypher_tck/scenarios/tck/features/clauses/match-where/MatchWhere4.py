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
        key="match-where4-1",
        feature_path="tck/features/clauses/match-where/MatchWhere4.feature",
        scenario="[1] Join nodes on inequality",
        cypher="MATCH (a), (b)\nWHERE a <> b\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:B)"},
                {"a": "(:B)", "b": "(:A)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Cartesian products, variable inequality joins, and row projections are not supported",
        tags=("match-where", "join", "inequality", "xfail"),
    ),

    Scenario(
        key="match-where4-2",
        feature_path="tck/features/clauses/match-where/MatchWhere4.feature",
        scenario="[2] Join with disjunctive multi-part predicates including patterns",
        cypher="MATCH (a), (b)\nWHERE a.id = 0\n  AND (a)-[:T]->(b:TheLabel)\n  OR (a)-[:T*]->(b:MissingLabel)\nRETURN DISTINCT b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:TheLabel {id: 0}), (b:TheLabel {id: 1}), (c:TheLabel {id: 2})
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c)
            """
        ),
        expected=Expected(
            node_ids=["b"],
            rows=[
                {"b": "(:TheLabel {id: 1})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Pattern predicates, variable-length relationships, OR, DISTINCT, and row projections not supported",
        tags=("match-where", "or", "pattern-predicate", "variable-length", "distinct", "xfail"),
    ),
]
