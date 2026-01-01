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
        key="match9-1",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[1] Variable length relationship variables are lists of relationships",
        cypher="MATCH ()-[r*0..1]-()\nRETURN last(r) AS l",
        graph=graph_fixture_from_create(
            """
            CREATE (a), (b), (c)
            CREATE (a)-[:T]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"l": "[:T]"},
                {"l": "[:T]"},
                {"l": "null"},
                {"l": "null"},
                {"l": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists, list functions, and row projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-2",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[2] Return relationships by collecting them as a list - directed, one way",
        cypher="MATCH (a)-[r:REL*2..2]->(b:End)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(e:End)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num: 1}], [:REL {num: 2}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-3",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[3] Return relationships by collecting them as a list - undirected, starting from two extremes",
        cypher="MATCH (a)-[r:REL*2..2]-(b:End)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (a:End)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:End)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num:1}], [:REL {num:2}]]"},
                {"r": "[[:REL {num:2}], [:REL {num:1}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-4",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[4] Return relationships by collecting them as a list - undirected, starting from one extreme",
        cypher="MATCH (a:Start)-[r:REL*2..2]-(b)\nRETURN r",
        graph=graph_fixture_from_create(
            """
            CREATE (s:Start)-[:REL {num: 1}]->(b:B)-[:REL {num: 2}]->(c:C)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:REL {num: 1}], [:REL {num: 2}]]"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length relationship lists and list projections are not supported",
        tags=("match", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-5",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[5] Variable length pattern with label predicate on both sides",
        cypher="MATCH (a:Blue)-[r*]->(b:Green)\nRETURN count(r)",
        graph=graph_fixture_from_create(
            """
            CREATE (a:Blue), (b:Red), (c:Green), (d:Yellow)
            CREATE (a)-[:T]->(b),
                   (b)-[:T]->(c),
                   (b)-[:T]->(d)
            """
        ),
        expected=Expected(
            rows=[
                {"count(r)": 1},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="Variable-length patterns, label predicates, and aggregations are not supported",
        tags=("match", "variable-length", "label", "aggregation", "xfail"),
    ),

    Scenario(
        key="match9-6",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[6] Matching relationships into a list and matching variable length using the list, with bound nodes",
        cypher="MATCH (a)-[r1]->()-[r2]->(b)\nWITH [r1, r2] AS rs, a AS first, b AS second\n  LIMIT 1\nMATCH (first)-[rs*]->(second)\nRETURN first, second",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:Y]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[
                {"first": "(:A)", "second": "(:C)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT pipelines, variable-length patterns, and relationship list matching are not supported",
        tags=("match", "with", "limit", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-7",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[7] Matching relationships into a list and matching variable length using the list, with bound nodes, wrong direction",
        cypher="MATCH (a)-[r1]->()-[r2]->(b)\nWITH [r1, r2] AS rs, a AS second, b AS first\n  LIMIT 1\nMATCH (first)-[rs*]->(second)\nRETURN first, second",
        graph=graph_fixture_from_create(
            """
            CREATE (a:A), (b:B), (c:C)
            CREATE (a)-[:Y]->(b),
                   (b)-[:Y]->(c)
            """
        ),
        expected=Expected(
            rows=[],
        ),
        gfql=None,
        status="xfail",
        reason="WITH/LIMIT pipelines, variable-length patterns, and relationship list matching are not supported",
        tags=("match", "with", "limit", "variable-length", "list", "xfail"),
    ),

    Scenario(
        key="match9-8",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[8] Variable length relationship in OPTIONAL MATCH",
        cypher="MATCH (a:A), (b:B)\nOPTIONAL MATCH (a)-[r*]-(b)\nWHERE r IS NULL\n  AND a <> b\nRETURN b",
        graph=graph_fixture_from_create(
            """
            CREATE (:A), (:B)
            """
        ),
        expected=Expected(
            rows=[
                {"b": "(:B)"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length patterns, null predicates, and variable comparisons are not supported",
        tags=("match", "optional-match", "variable-length", "null", "comparison", "xfail"),
    ),

    Scenario(
        key="match9-9",
        feature_path="tck/features/clauses/match/Match9.feature",
        scenario="[9] Optionally matching named paths with variable length patterns",
        cypher="MATCH (a {name: 'A'}), (x)\nWHERE x.name IN ['B', 'C']\nOPTIONAL MATCH p = (a)-[r*]->(x)\nRETURN r, x, p",
        graph=graph_fixture_from_create(
            """
            CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
            CREATE (a)-[:X]->(b)
            """
        ),
        expected=Expected(
            rows=[
                {"r": "[[:X]]", "x": "({name: 'B'})", "p": "<({name: 'A'})-[:X]->({name: 'B'})>"},
                {"r": "null", "x": "({name: 'C'})", "p": "null"},
            ],
        ),
        gfql=None,
        status="xfail",
        reason="OPTIONAL MATCH semantics, variable-length relationship lists, IN predicates, and named path returns are not supported",
        tags=("match", "optional-match", "variable-length", "list", "path", "in-predicate", "xfail"),
    ),
]
