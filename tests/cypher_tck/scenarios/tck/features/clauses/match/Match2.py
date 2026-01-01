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
        key="match2-1",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[1] Match non-existent relationships returns empty",
        cypher="MATCH ()-[r]->()\nRETURN r",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(edge_ids=[]),
        gfql=[n(), e_forward(), n()],
        tags=("match", "relationship", "empty-graph"),
    ),

    Scenario(
        key="match2-2",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[2] Matching a relationship pattern using a label predicate on both sides",
        cypher="MATCH (:A)-[r]->(:B)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:T1]->(:B),
                   (:B)-[:T2]->(:A),
                   (:B)-[:T3]->(:B),
                   (:A)-[:T4]->(:A)
            """
        ),
        expected=Expected(
            edge_ids=["rel_1"],
            rows=[
                {"r": "[:T1]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Label predicates on both sides of relationship matches are not supported in the harness",
        tags=("match", "relationship", "label", "xfail"),
    ),

    Scenario(
        key="match2-3",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[3] Matching a self-loop with an undirected relationship pattern",
        cypher="MATCH ()-[r]-()\nRETURN type(r) AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a)
            CREATE (a)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="type(r) projection and row comparisons are not supported",
        tags=("match", "relationship", "type-projection", "xfail"),
    ),

    Scenario(
        key="match2-4",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[4] Matching a self-loop with a directed relationship pattern",
        cypher="MATCH ()-[r]->()\nRETURN type(r) AS r",
        graph=graph_fixture_from_create(
            """
            CREATE (a)
            CREATE (a)-[:T]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "'T'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="type(r) projection and row comparisons are not supported",
        tags=("match", "relationship", "type-projection", "xfail"),
    ),

    Scenario(
        key="match2-5",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[5] Match relationship with inline property value",
        cypher="MATCH (node)-[r:KNOWS {name: 'monkey'}]->(a)\nRETURN a",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)<-[:KNOWS {name: 'monkey'}]-()-[:KNOWS {name: 'woot'}]->(:B)
            """
        ),
        expected=Expected(
            node_ids=["anon_1"],
            rows=[
                {"a": "(:A)"},
            ],
        ),
        gfql=[n(), e_forward({"type": "KNOWS", "name": "monkey"}), n(name="a")],
        return_alias="a",
        tags=("match", "relationship", "property"),
    ),

    Scenario(
        key="match2-6",
        feature_path="tck/features/clauses/match/Match2.feature",
        scenario="[6] Match relationships with multiple types",
        cypher="MATCH (n)-[r:KNOWS|HATES]->(x)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}),
              (b {name: 'B'}),
              (c {name: 'C'}),
              (a)-[:KNOWS]->(b),
              (a)-[:HATES]->(c),
              (a)-[:WONDERS]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[:KNOWS]"},
                {"r": "[:HATES]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Multiple relationship types and row projections are not supported",
        tags=("match", "relationship", "multi-type", "xfail"),
    ),
]
