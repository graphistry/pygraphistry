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
        key="with4-1",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[1] Aliasing relationship variable",
        cypher="MATCH ()-[r1]->()\nWITH r1 AS r2\nRETURN r2 AS rel",
        graph=graph_fixture_from_create(
            """
            CREATE ()-[:T1]->(),
                   ()-[:T2]->()
            """
        ),
        expected=Expected(
            rows=[
                {"rel": "[:T1]"},
                {"rel": "[:T2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and relationship aliasing are not supported",
        tags=("with", "alias", "relationship", "xfail"),
    ),

    Scenario(
        key="with4-2",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[2] Aliasing expression to new variable name",
        cypher="MATCH (a:Begin)\nWITH a.num AS property\nMATCH (b:End)\nWHERE property = b.num\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (:Begin {num: 42}),
                   (:End {num: 42}),
                   (:End {num: 3})
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:End {num: 42})"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, expression aliasing, and row projections are not supported",
        tags=("with", "alias", "projection", "xfail"),
    ),

    Scenario(
        key="with4-3",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[3] Aliasing expression to existing variable name",
        cypher="MATCH (n)\nWITH n.name AS n\nRETURN n",
        graph=graph_fixture_from_create(
            """
            CREATE ({num: 1, name: 'King Kong'}),
              ({num: 2, name: 'Ann Darrow'})
            """
        ),
        expected=Expected(
            rows=[
                {"n": "'Ann Darrow'"},
                {"n": "'King Kong'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines and expression projections are not supported",
        tags=("with", "alias", "projection", "xfail"),
    ),

    Scenario(
        key="with4-4",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[4] Fail when forwarding multiple aliases with the same name",
        cypher="WITH 1 AS a, 2 AS a\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for duplicate WITH aliases is not enforced",
        tags=("with", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with4-5",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[5] Fail when not aliasing expressions in WITH",
        cypher="MATCH (a)\nWITH a, count(*)\nRETURN a",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(),
        gfql=None,
        status="xfail",
        reason="Compile-time validation for WITH expression aliasing is not enforced",
        tags=("with", "syntax-error", "xfail"),
    ),

    Scenario(
        key="with4-6",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[6] Reusing variable names in WITH",
        cypher="MATCH (person:Person)<--(message)<-[like]-(:Person)\nWITH like.creationDate AS likeTime, person AS person\n  ORDER BY likeTime, message.id\nWITH head(collect({likeTime: likeTime})) AS latestLike, person AS person\nWITH latestLike.likeTime AS likeTime\n  ORDER BY likeTime\nRETURN likeTime",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Person), (b:Person), (m:Message {id: 10})
            CREATE (a)-[:LIKE {creationDate: 20160614}]->(m)-[:POSTED_BY]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"likeTime": 20160614},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, ORDER BY, aggregations, and list/map expressions are not supported",
        tags=("with", "orderby", "aggregation", "xfail"),
    ),

    Scenario(
        key="with4-7",
        feature_path="tck/features/clauses/with/With4.feature",
        scenario="[7] Multiple aliasing and backreferencing",
        cypher="CREATE (m {id: 0})\nWITH {first: m.id} AS m\nWITH {second: m.first} AS m\nRETURN m.second",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"m.second": 0},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH pipelines, map projections, and side effect validation are not supported",
        tags=("with", "map", "projection", "xfail"),
    ),
]
