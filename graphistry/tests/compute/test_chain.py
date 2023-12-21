from graphistry.compute.ast import ASTNode, ASTEdge, n, e
from graphistry.compute.chain import to_json as chain_to_json, from_json as chain_from_json

def test_chain_serialization_mt():
    o = chain_to_json([])
    d = chain_from_json(o)
    assert d == []
    assert o == []

def test_chain_serialization_node():
    o = chain_to_json([n(query='zzz', name='abc')])
    d = chain_from_json(o)
    assert isinstance(d[0], ASTNode)
    assert d[0]._query == 'zzz'
    assert d[0]._name == 'abc'
    o2 = chain_to_json(d)
    assert o == o2

def test_chain_serialization_edge():
    o = chain_to_json([e(edge_query='zzz', name='abc')])
    d = chain_from_json(o)
    assert isinstance(d[0], ASTEdge)
    assert d[0]._edge_query == 'zzz'
    assert d[0]._name == 'abc'
    o2 = chain_to_json(d)
    assert o == o2

def test_chain_serialization_multi():
    o = chain_to_json([n(query='zzz', name='abc'), e(edge_query='zzz', name='abc')])
    d = chain_from_json(o)
    assert isinstance(d[0], ASTNode)
    assert d[0]._query == 'zzz'
    assert d[0]._name == 'abc'
    assert isinstance(d[1], ASTEdge)
    assert d[1]._edge_query == 'zzz'
    assert d[1]._name == 'abc'
    o2 = chain_to_json(d)
    assert o == o2
