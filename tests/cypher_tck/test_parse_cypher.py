from tests.cypher_tck.parse_cypher import graph_fixture_from_create


def test_parse_create_nodes_only():
    script = """
    CREATE (:A), (:B {name: 'b'}), ({name: 'c'})
    """
    fixture = graph_fixture_from_create(script)
    nodes = fixture.nodes
    assert len(nodes) == 3
    labels = {tuple(node.get("labels", [])) for node in nodes}
    assert ("A",) in labels
    assert ("B",) in labels
    assert () in labels
    by_name = {node.get("name"): node for node in nodes}
    assert "b" in by_name
    assert "c" in by_name


def test_parse_create_relationship():
    script = """
    CREATE (a:A {name: 'a'}), (b:B)
    CREATE (a)-[:KNOWS]->(b)
    """
    fixture = graph_fixture_from_create(script)
    nodes = {node["id"]: node for node in fixture.nodes}
    assert set(nodes) == {"a", "b"}
    assert nodes["a"].get("labels") == ["A"]
    assert nodes["b"].get("labels") == ["B"]

    assert len(fixture.edges) == 1
    edge = fixture.edges[0]
    assert edge["src"] == "a"
    assert edge["dst"] == "b"
    assert edge["type"] == "KNOWS"
