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
        key="with7-1",
        feature_path="tck/features/clauses/with/With7.feature",
        scenario="[1] A simple pattern with one bound endpoint",
        cypher="MATCH (a:A)-[r:REL]->(b:B)\nWITH a AS b, b AS tmp, r AS r\nWITH b AS a, r\nLIMIT 1\nMATCH (a)-[r]->(b)\nRETURN a, r, b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A)-[:REL]->(:B)
            """
        ),
        expected=Expected(
            rows=[
                {"a": "(:A)", "r": "[:REL]", "b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, LIMIT, and row projections are not supported",
        tags=("with", "limit", "xfail"),
    ),

    Scenario(
        key="with7-2",
        feature_path="tck/features/clauses/with/With7.feature",
        scenario="[2] Multiple WITHs using a predicate and aggregation",
        cypher="MATCH (david {name: 'David'})--(otherPerson)-->()\nWITH otherPerson, count(*) AS foaf\nWHERE foaf > 1\nWITH otherPerson\nWHERE otherPerson.name <> 'NotOther'\nRETURN count(*)",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'David'}),
                   (b {name: 'Other'}),
                   (c {name: 'NotOther'}),
                   (d {name: 'NotOther2'}),
                   (a)-[:REL]->(b),
                   (a)-[:REL]->(c),
                   (a)-[:REL]->(d),
                   (b)-[:REL]->(),
                   (b)-[:REL]->(),
                   (c)-[:REL]->(),
                   (c)-[:REL]->(),
                   (d)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, WHERE filtering, and aggregations are not supported",
        tags=("with", "where", "aggregation", "xfail"),
    ),
]
