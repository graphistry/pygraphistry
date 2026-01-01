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
        key="return5-1",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[1] DISTINCT inside aggregation should work with lists in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: n.list}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and map/list projections are not supported",
        tags=("return", "distinct", "aggregation", "xfail"),
    ),

    Scenario(
        key="return5-2",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[2] DISTINCT on nullable values",
        cypher="MATCH (n)\nRETURN DISTINCT n.name",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "name": "Florescu"},
                {"id": "n2", "labels": []},
                {"id": "n3", "labels": []},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"n.name": "'Florescu'"},
                {"n.name": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT projections and null semantics are not supported",
        tags=("return", "distinct", "null", "xfail"),
    ),

    Scenario(
        key="return5-3",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[3] DISTINCT inside aggregation should work with nested lists in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: [[n.list, n.list], [n.list, n.list]]}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and nested list projections are not supported",
        tags=("return", "distinct", "aggregation", "list", "xfail"),
    ),

    Scenario(
        key="return5-4",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[4] DISTINCT inside aggregation should work with nested lists of maps in maps",
        cypher="MATCH (n)\nRETURN count(DISTINCT {name: [{name2: n.list}, {baz: {apa: n.list}}]}) AS count",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "list": ["A", "B"]},
                {"id": "n2", "labels": [], "list": ["A", "B"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"count": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT aggregation and nested map/list projections are not supported",
        tags=("return", "distinct", "aggregation", "map", "xfail"),
    ),

    Scenario(
        key="return5-5",
        feature_path="tck/features/clauses/return/Return5.feature",
        scenario="[5] Aggregate on list values",
        cypher="MATCH (a)\nRETURN DISTINCT a.color, count(*)",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": [], "color": ["red"]},
                {"id": "n2", "labels": [], "color": ["blue"]},
                {"id": "n3", "labels": [], "color": ["red"]},
            ],
            edges=[],
        ),
        expected=Expected(
            rows=[
                {"a.color": "['red']", "count(*)": 2},
                {"a.color": "['blue']", "count(*)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="DISTINCT projections, aggregations, and list values are not supported",
        tags=("return", "distinct", "aggregation", "list", "xfail"),
    ),
]
