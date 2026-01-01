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
        key="return-orderby1-1",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[1] ORDER BY should order booleans in the expected order",
        cypher="UNWIND [true, false] AS bools\nRETURN bools\nORDER BY bools",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "false"},
                {"bools": "true"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-2",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[2] ORDER BY DESC should order booleans in the expected order",
        cypher="UNWIND [true, false] AS bools\nRETURN bools\nORDER BY bools DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"bools": "true"},
                {"bools": "false"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-3",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[3] ORDER BY should order strings in the expected order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nRETURN strings\nORDER BY strings",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "''"},
                {"strings": "' '"},
                {"strings": "'.*'"},
                {"strings": "'one'"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-4",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[4] ORDER BY DESC should order strings in the expected order",
        cypher="UNWIND ['.*', '', ' ', 'one'] AS strings\nRETURN strings\nORDER BY strings DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"strings": "'one'"},
                {"strings": "'.*'"},
                {"strings": "' '"},
                {"strings": "''"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-5",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[5] ORDER BY should order ints in the expected order",
        cypher="UNWIND [1, 3, 2] AS ints\nRETURN ints\nORDER BY ints",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 1},
                {"ints": 2},
                {"ints": 3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-6",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[6] ORDER BY DESC should order ints in the expected order",
        cypher="UNWIND [1, 3, 2] AS ints\nRETURN ints\nORDER BY ints DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"ints": 3},
                {"ints": 2},
                {"ints": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-7",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[7] ORDER BY should order floats in the expected order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nRETURN floats\nORDER BY floats",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 1.3},
                {"floats": 1.5},
                {"floats": 999.99},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-8",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[8] ORDER BY DESC should order floats in the expected order",
        cypher="UNWIND [1.5, 1.3, 999.99] AS floats\nRETURN floats\nORDER BY floats DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"floats": 999.99},
                {"floats": 1.5},
                {"floats": 1.3},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-9",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[9] ORDER BY should order lists in the expected order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nRETURN lists\nORDER BY lists",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[]"},
                {"lists": "['a']"},
                {"lists": "['a', 1]"},
                {"lists": "[1]"},
                {"lists": "[1, 'a']"},
                {"lists": "[1, null]"},
                {"lists": "[null, 1]"},
                {"lists": "[null, 2]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-10",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[10] ORDER BY DESC should order lists in the expected order",
        cypher="UNWIND [[], ['a'], ['a', 1], [1], [1, 'a'], [1, null], [null, 1], [null, 2]] AS lists\nRETURN lists\nORDER BY lists DESC",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"lists": "[null, 2]"},
                {"lists": "[null, 1]"},
                {"lists": "[1, null]"},
                {"lists": "[1, 'a']"},
                {"lists": "[1]"},
                {"lists": "['a', 1]"},
                {"lists": "['a']"},
                {"lists": "[]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND and ORDER BY are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-11",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[11] ORDER BY should order distinct types in the expected order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nRETURN types\nORDER BY types",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "{a: 'map'}"},
                {"types": "(:N)"},
                {"types": "[:REL]"},
                {"types": "['list']"},
                {"types": "<(:N)-[:REL]->()>"},
                {"types": "'text'"},
                {"types": "false"},
                {"types": 1.5},
                {"types": "NaN"},
                {"types": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),

    Scenario(
        key="return-orderby1-12",
        feature_path="tck/features/clauses/return-orderby/ReturnOrderBy1.feature",
        scenario="[12] ORDER BY DESC should order distinct types in the expected order",
        cypher="MATCH p = (n:N)-[r:REL]->()\nUNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types\nRETURN types\nORDER BY types DESC",
        graph=graph_fixture_from_create(
            """
            CREATE (:N)-[:REL]->()
            """
        ),
        expected=Expected(
            rows=[
                {"types": "null"},
                {"types": "NaN"},
                {"types": 1.5},
                {"types": "false"},
                {"types": "'text'"},
                {"types": "<(:N)-[:REL]->()>"},
                {"types": "['list']"},
                {"types": "[:REL]"},
                {"types": "(:N)"},
                {"types": "{a: 'map'}"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="UNWIND, ORDER BY, and heterogeneous type ordering are not supported",
        tags=("return", "orderby", "unwind", "xfail"),
    ),
]
