from graphistry.compute.ast import (
    from_json, ASTNode, ASTEdge, ASTLet, ASTRemoteGraph, ASTChainRef,
    n, e, e_forward, e_reverse, e_undirected
)

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


def test_serialization_let_empty():
    """Test Let with empty bindings"""
    dag = ASTLet({})
    o = dag.to_json()
    assert o == {'type': 'Let', 'bindings': {}}
    dag2 = from_json(o)
    assert isinstance(dag2, ASTLet)
    assert dag2.bindings == {}
    o2 = dag2.to_json()
    assert o == o2


def test_serialization_let_single():
    """Test Let with single binding"""
    """Test QueryDAG with single binding"""
>>>>>>> docs(gfql): add comprehensive docstrings to PR1 AST classes
    dag = ASTLet({'a': n()})
    o = dag.to_json()
    assert o['type'] == 'Let'
    assert 'a' in o['bindings']
    dag2 = from_json(o)
    assert isinstance(dag2, ASTLet)
    assert 'a' in dag2.bindings
    assert isinstance(dag2.bindings['a'], ASTNode)


def test_serialization_let_multi():
    """Test Let with multiple bindings"""
    dag = ASTLet({
        'nodes': n({'type': 'person'}),
        'edges': e_forward(),
        'remote': ASTRemoteGraph('dataset123')
    })
    o = dag.to_json()
    dag2 = from_json(o)
    assert isinstance(dag2, ASTLet)
    assert len(dag2.bindings) == 3
    assert isinstance(dag2.bindings['nodes'], ASTNode)
    assert isinstance(dag2.bindings['edges'], ASTEdge)
    assert isinstance(dag2.bindings['remote'], ASTRemoteGraph)


def test_serialization_remoteGraph():
    """Test RemoteGraph serialization"""
    rg = ASTRemoteGraph('my-dataset-id')
    o = rg.to_json()
    assert o == {'type': 'RemoteGraph', 'dataset_id': 'my-dataset-id'}
    rg2 = from_json(o)
    assert isinstance(rg2, ASTRemoteGraph)
    assert rg2.dataset_id == 'my-dataset-id'
    assert rg2.token is None


def test_serialization_remoteGraph_with_token():
    """Test RemoteGraph with auth token"""
    rg = ASTRemoteGraph('my-dataset-id', token='secret-token')
    o = rg.to_json()
    assert o == {
        'type': 'RemoteGraph',
        'dataset_id': 'my-dataset-id',
        'token': 'secret-token'
    }
    rg2 = from_json(o)
    assert isinstance(rg2, ASTRemoteGraph)
    assert rg2.dataset_id == 'my-dataset-id'
    assert rg2.token == 'secret-token'


def test_serialization_chainRef_empty():
    """Test ChainRef with empty chain"""
    cr = ASTChainRef('mydata', [])
    o = cr.to_json()
    assert o == {'type': 'ChainRef', 'ref': 'mydata', 'chain': []}
    cr2 = from_json(o)
    assert isinstance(cr2, ASTChainRef)
    assert cr2.ref == 'mydata'
    assert cr2.chain == []


def test_serialization_chainRef_with_ops():
    """Test ChainRef with operations"""
    cr = ASTChainRef('data1', [n({'type': 'person'}), e_forward()])
    o = cr.to_json()
    assert o['type'] == 'ChainRef'
    assert o['ref'] == 'data1'
    assert len(o['chain']) == 2
    cr2 = from_json(o)
    assert isinstance(cr2, ASTChainRef)
    assert cr2.ref == 'data1'
    assert len(cr2.chain) == 2
    assert isinstance(cr2.chain[0], ASTNode)
    assert isinstance(cr2.chain[1], ASTEdge)
