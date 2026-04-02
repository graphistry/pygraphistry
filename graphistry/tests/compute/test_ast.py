from graphistry.compute.ast import from_json, ASTNode, ASTEdge, n, e, e_forward, e_reverse, e_undirected

def test_serialization_node():

    node = n(query='zzz', name='abc')
    o = node.to_json()
    node2 = from_json(o)
    assert isinstance(node2, ASTNode)
    assert node2.query == 'zzz'
    assert node2._name == 'abc'
    o2 = node2.to_json()
    assert o == o2

def test_serialization_edge():

    edge = e(edge_query='zzz', name='abc')
    o = edge.to_json()
    edge2 = from_json(o)
    assert isinstance(edge2, ASTEdge)
    assert edge2.edge_query == 'zzz'
    assert edge2._name == 'abc'
    o2 = edge2.to_json()
    assert o == o2


def test_serialization_edge_open_range_does_not_collapse_to_single_hop():

    edge = e_forward({"type": "REPLY_OF"}, min_hops=0, to_fixed_point=True)
    o = edge.to_json(validate=False)
    assert o["hops"] is None
    assert o["min_hops"] == 0
    assert o["to_fixed_point"] is True

    edge2 = from_json(o, validate=False)
    assert isinstance(edge2, ASTEdge)
    assert edge2.hops is None
    assert edge2.min_hops == 0
    assert edge2.to_fixed_point is True
