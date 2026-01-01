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
        key="with-where5-1",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[1] Filter out on null",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te'\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, and null semantics are not supported",
        tags=("with", "null", "comparison", "xfail"),
    ),

    Scenario(
        key="with-where5-2",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[2] Filter out on null if the AND'd predicate evaluates to false",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te' AND i:TextNode\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, label predicates, and null semantics are not supported",
        tags=("with", "null", "comparison", "label-predicate", "xfail"),
    ),

    Scenario(
        key="with-where5-3",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[3] Filter out on null if the AND'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i:TextNode)\nWITH i\nWHERE i.var > 'te' AND i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, comparison predicates, IS NOT NULL, and null semantics are not supported",
        tags=("with", "null", "comparison", "is-not-null", "xfail"),
    ),

    Scenario(
        key="with-where5-4",
        feature_path="tck/features/clauses/with-where/WithWhere5.feature",
        scenario="[4] Do not filter out on null if the OR'd predicate evaluates to true",
        cypher="MATCH (:Root {name: 'x'})-->(i)\nWITH i\nWHERE i.var > 'te' OR i.var IS NOT NULL\nRETURN i",
        graph=graph_fixture_from_create(
            """
            CREATE (root:Root {name: 'x'}),
                   (child1:TextNode {var: 'text'}),
                   (child2:IntNode {var: 0})
            CREATE (root)-[:T]->(child1),
                   (root)-[:T]->(child2)
            """
        ),
        expected=Expected(
            rows=[
                {"i": "(:TextNode {var: 'text'})"},
                {"i": "(:IntNode {var: 0})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, OR predicates, comparison predicates, and null semantics are not supported",
        tags=("with", "null", "or", "comparison", "xfail"),
    ),
]
