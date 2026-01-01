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
        key="with2-1",
        feature_path="tck/features/clauses/with/With2.feature",
        scenario="[1] Forwarding a property to express a join",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\nMATCH (b)\nWHERE b.id = property\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End {num: 42, id: 0}),
                   (:End {num: 3}),
                   (:Begin {num: a.id})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42, id: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, joins, and row projections are not supported",
        tags=("with", "join", "xfail"),
    ),

    Scenario(
        key="with2-2",
        feature_path="tck/features/clauses/with/With2.feature",
        scenario="[2] Forwarding a nested map literal",
        cypher="WITH {name: {name2: 'baz'}} AS nestedMap\nRETURN nestedMap.name.name2",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"nestedMap.name.name2": "'baz'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and map projections are not supported",
        tags=("with", "map", "projection", "xfail"),
    ),
]
