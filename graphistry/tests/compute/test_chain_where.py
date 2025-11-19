import pandas as pd

from graphistry.compute import n, e_forward
from graphistry.compute.chain import Chain
from graphistry.gfql.same_path_types import col, compare
from graphistry.tests.test_compute import CGFull


def test_chain_where_roundtrip():
    chain = Chain([n({'type': 'account'}, name='a'), e_forward(), n(name='c')], where=[
        compare(col('a', 'owner_id'), '==', col('c', 'owner_id'))
    ])
    json_data = chain.to_json()
    assert 'where' in json_data
    restored = Chain.from_json(json_data)
    assert len(restored.where) == 1


def test_chain_from_json_literal():
    json_chain = {
        'chain': [
            n({'type': 'account'}, name='a').to_json(),
            e_forward().to_json(),
            n({'type': 'user'}, name='c').to_json(),
        ],
        'where': [
            {'eq': {'left': 'a.owner_id', 'right': 'c.owner_id'}}
        ],
    }
    chain = Chain.from_json(json_chain)
    assert len(chain.where) == 1


def test_gfql_chain_dict_with_where_executes():
    nodes_df = n({'type': 'account'}, name='a').to_json()
    edge_json = e_forward().to_json()
    user_json = n({'type': 'user'}, name='c').to_json()
    json_chain = {
        'chain': [nodes_df, edge_json, user_json],
        'where': [{'eq': {'left': 'a.owner_id', 'right': 'c.owner_id'}}],
    }
    nodes_df = pd.DataFrame([
        {'id': 'acct1', 'type': 'account', 'owner_id': 'user1'},
        {'id': 'user1', 'type': 'user'},
    ])
    edges_df = pd.DataFrame([{'src': 'acct1', 'dst': 'user1'}])
    g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
    res = g.gfql(json_chain)
    assert res._nodes is not None
