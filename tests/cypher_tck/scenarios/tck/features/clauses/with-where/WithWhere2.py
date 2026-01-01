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
        key="with-where2-1",
        feature_path="tck/features/clauses/with-where/WithWhere2.feature",
        scenario="[1] Filter nodes with conjunctive two-part property predicate on multi variables with multiple bindings",
        cypher="MATCH (a)--(b)--(c)--(d)--(a), (b)--(d)\nWITH a, c, d\nWHERE a.id = 1\n  AND c.id = 2\nRETURN d",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B {id: 1}), (c:C {id: 2}), (d:D)
            CREATE (a)-[:T]->(b),
                   (a)-[:T]->(c),
                   (a)-[:T]->(d),
                   (b)-[:T]->(c),
                   (b)-[:T]->(d),
                   (c)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"d": "(:A)"},
                {"d": "(:D)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and multi-variable WHERE predicates are not supported",
        tags=("with", "where", "multi-var", "xfail"),
    ),

    Scenario(
        key="with-where2-2",
        feature_path="tck/features/clauses/with-where/WithWhere2.feature",
        scenario="[2] Filter node with conjunctive multi-part property predicates on multi variables with multiple bindings",
        cypher="MATCH (advertiser)-[:ADV_HAS_PRODUCT]->(out)-[:AP_HAS_VALUE]->(red)<-[:AA_HAS_VALUE]-(a)\nWITH a, advertiser, red, out\nWHERE advertiser.id = $1\n  AND a.id = $2\n  AND red.name = 'red'\n  AND out.name = 'product1'\nRETURN out.name",
        graph=graph_fixture_from_create(
            """
            CREATE (advertiser {name: 'advertiser1', id: 0}),
                   (thing {name: 'Color', id: 1}),
                   (red {name: 'red'}),
                   (p1 {name: 'product1'}),
                   (p2 {name: 'product4'})
            CREATE (advertiser)-[:ADV_HAS_PRODUCT]->(p1),
                   (advertiser)-[:ADV_HAS_PRODUCT]->(p2),
                   (thing)-[:AA_HAS_VALUE]->(red),
                   (p1)-[:AP_HAS_VALUE]->(red),
                   (p2)-[:AP_HAS_VALUE]->(red)
            """
        ),
        expected=Expected(
            rows=[
                {"out.name": "'product1'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, parameter binding, and multi-variable WHERE predicates are not supported",
        tags=("with", "where", "params", "multi-var", "xfail"),
    ),
]
