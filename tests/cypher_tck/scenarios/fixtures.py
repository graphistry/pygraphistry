from tests.cypher_tck.parse_cypher import graph_fixture_from_create


MATCH5_GRAPH = graph_fixture_from_create(
    """
    CREATE (n0:A {name: 'n0'}),
           (n00:B {name: 'n00'}),
           (n01:B {name: 'n01'}),
           (n000:C {name: 'n000'}),
           (n001:C {name: 'n001'}),
           (n010:C {name: 'n010'}),
           (n011:C {name: 'n011'}),
           (n0000:D {name: 'n0000'}),
           (n0001:D {name: 'n0001'}),
           (n0010:D {name: 'n0010'}),
           (n0011:D {name: 'n0011'}),
           (n0100:D {name: 'n0100'}),
           (n0101:D {name: 'n0101'}),
           (n0110:D {name: 'n0110'}),
           (n0111:D {name: 'n0111'})
    CREATE (n0)-[:LIKES]->(n00),
           (n0)-[:LIKES]->(n01),
           (n00)-[:LIKES]->(n000),
           (n00)-[:LIKES]->(n001),
           (n01)-[:LIKES]->(n010),
           (n01)-[:LIKES]->(n011),
           (n000)-[:LIKES]->(n0000),
           (n000)-[:LIKES]->(n0001),
           (n001)-[:LIKES]->(n0010),
           (n001)-[:LIKES]->(n0011),
           (n010)-[:LIKES]->(n0100),
           (n010)-[:LIKES]->(n0101),
           (n011)-[:LIKES]->(n0110),
           (n011)-[:LIKES]->(n0111)
    """
)

MATCH7_GRAPH_SINGLE = graph_fixture_from_create(
    """
    CREATE (s:Single), (a:A {num: 42}),
           (b:B {num: 46}), (c:C)
    CREATE (s)-[:REL]->(a),
           (s)-[:REL]->(b),
           (a)-[:REL]->(c),
           (b)-[:LOOP]->(b)
    """
)

MATCH7_GRAPH_AB = graph_fixture_from_create(
    """
    CREATE (:A)-[:T]->(:B)
    """
)

MATCH7_GRAPH_ABC = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
    CREATE (a)-[:KNOWS]->(b),
           (b)-[:KNOWS]->(c)
    """
)

MATCH7_GRAPH_REL = graph_fixture_from_create(
    """
    CREATE (a:A {num: 1})-[:REL {name: 'r1'}]->(b:B {num: 2})-[:REL {name: 'r2'}]->(c:C {num: 3})
    """
)

MATCH7_GRAPH_X = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'}), (c {name: 'C'})
    CREATE (a)-[:X]->(b)
    """
)

MATCH7_GRAPH_AB_X = graph_fixture_from_create(
    """
    CREATE (a {name: 'A'}), (b {name: 'B'})
    CREATE (a)-[:X]->(b)
    """
)

MATCH7_GRAPH_LABELS = graph_fixture_from_create(
    """
    CREATE (:X), (x:X), (y1:Y), (y2:Y:Z)
    CREATE (x)-[:REL]->(y1),
           (x)-[:REL]->(y2)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_BOTH = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team)
    CREATE (a)-[:PLAYS_FOR]->(b),
           (a)-[:SUPPORTS]->(b)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_SINGLE = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team)
    CREATE (a)-[:PLAYS_FOR]->(b)
    """
)

MATCH7_GRAPH_PLAYER_TEAM_DIFF = graph_fixture_from_create(
    """
    CREATE (a:Player), (b:Team), (c:Team)
    CREATE (a)-[:PLAYS_FOR]->(b),
           (a)-[:SUPPORTS]->(c)
    """
)

WITH_ORDERBY4_GRAPH = graph_fixture_from_create(
    """
    CREATE (:A {num: 1, num2: 4}),
           (:A {num: 5, num2: 2}),
           (:A {num: 9, num2: 0}),
           (:A {num: 3, num2: 3}),
           (:A {num: 7, num2: 1})
    """
)

BINARY_TREE_1_GRAPH = graph_fixture_from_create(
    """
    CREATE (a:A {name: 'a'}),
           (b1:X {name: 'b1'}),
           (b2:X {name: 'b2'}),
           (b3:X {name: 'b3'}),
           (b4:X {name: 'b4'}),
           (c11:X {name: 'c11'}),
           (c12:X {name: 'c12'}),
           (c21:X {name: 'c21'}),
           (c22:X {name: 'c22'}),
           (c31:X {name: 'c31'}),
           (c32:X {name: 'c32'}),
           (c41:X {name: 'c41'}),
           (c42:X {name: 'c42'})
    CREATE (a)-[:KNOWS]->(b1),
           (a)-[:KNOWS]->(b2),
           (a)-[:FOLLOWS]->(b3),
           (a)-[:FOLLOWS]->(b4)
    CREATE (b1)-[:FRIEND]->(c11),
           (b1)-[:FRIEND]->(c12),
           (b2)-[:FRIEND]->(c21),
           (b2)-[:FRIEND]->(c22),
           (b3)-[:FRIEND]->(c31),
           (b3)-[:FRIEND]->(c32),
           (b4)-[:FRIEND]->(c41),
           (b4)-[:FRIEND]->(c42)
    CREATE (b1)-[:FRIEND]->(b2),
           (b2)-[:FRIEND]->(b3),
           (b3)-[:FRIEND]->(b4),
           (b4)-[:FRIEND]->(b1);
    """
)

BINARY_TREE_2_GRAPH = graph_fixture_from_create(
    """
    CREATE (a:A {name: 'a'}),
           (b1:X {name: 'b1'}),
           (b2:X {name: 'b2'}),
           (b3:X {name: 'b3'}),
           (b4:X {name: 'b4'}),
           (c11:X {name: 'c11'}),
           (c12:Y {name: 'c12'}),
           (c21:X {name: 'c21'}),
           (c22:Y {name: 'c22'}),
           (c31:X {name: 'c31'}),
           (c32:Y {name: 'c32'}),
           (c41:X {name: 'c41'}),
           (c42:Y {name: 'c42'})
    CREATE (a)-[:KNOWS]->(b1),
           (a)-[:KNOWS]->(b2),
           (a)-[:FOLLOWS]->(b3),
           (a)-[:FOLLOWS]->(b4)
    CREATE (b1)-[:FRIEND]->(c11),
           (b1)-[:FRIEND]->(c12),
           (b2)-[:FRIEND]->(c21),
           (b2)-[:FRIEND]->(c22),
           (b3)-[:FRIEND]->(c31),
           (b3)-[:FRIEND]->(c32),
           (b4)-[:FRIEND]->(c41),
           (b4)-[:FRIEND]->(c42)
    CREATE (b1)-[:FRIEND]->(b2),
           (b2)-[:FRIEND]->(b3),
           (b3)-[:FRIEND]->(b4),
           (b4)-[:FRIEND]->(b1);
    """
)
