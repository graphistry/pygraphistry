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
        key="match8-1",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[1] Pattern independent of bound variables results in cross product",
        cypher="MATCH (a)\nWITH a\nMATCH (b)\nRETURN a, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "b": "(:A)"},
                {"a": "(:A)", "b": "(:B)"},
                {"a": "(:B)", "b": "(:A)"},
                {"a": "(:B)", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, cartesian products, and row projections are not supported",
        tags=("match", "with", "cartesian", "xfail"),
    ),

    Scenario(
        key="match8-2",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[2] Counting rows after MATCH, MERGE, OPTIONAL MATCH",
        cypher="MATCH (a)\nMERGE (b)\nWITH *\nOPTIONAL MATCH (a)--(b)\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B)
            CREATE (a)-[:T1]->(b),
                   (b)-[:T2]->(a)
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 6},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="MERGE, OPTIONAL MATCH semantics, aggregations, and row projections are not supported",
        tags=("match", "merge", "optional-match", "aggregation", "xfail"),
    ),

    Scenario(
        key="match8-3",
        feature_path="tck/features/clauses/match/Match8.feature",
        scenario="[3] Matching and disregarding output, then matching again",
        cypher="MATCH ()-->()\nWITH 1 AS x\nMATCH ()-[r1]->()<--()\nRETURN sum(r1.times)",
        graph=graph_fixture_from_create(
            """
            CREATE (andres {name: 'Andres'}),
                   (michael {name: 'Michael'}),
                   (peter {name: 'Peter'}),
                   (bread {type: 'Bread'}),
                   (veggies {type: 'Veggies'}),
                   (meat {type: 'Meat'})
            CREATE (andres)-[:ATE {times: 10}]->(bread),
                   (andres)-[:ATE {times: 8}]->(veggies),
                   (michael)-[:ATE {times: 4}]->(veggies),
                   (michael)-[:ATE {times: 6}]->(bread),
                   (michael)-[:ATE {times: 9}]->(meat),
                   (peter)-[:ATE {times: 7}]->(veggies),
                   (peter)-[:ATE {times: 7}]->(bread),
                   (peter)-[:ATE {times: 4}]->(meat)
            """
        ),
        expected=Expected(
            rows=[
                {"sum(r1.times)": 776},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, aggregations, and row projections are not supported",
        tags=("match", "with", "aggregation", "xfail"),
    ),
]
