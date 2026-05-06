import pandas as pd
import pytest

from graphistry.compute import n, e_forward
from graphistry.compute.chain import Chain
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.tests.test_compute import CGFull


def _sample_graph():
    nodes_df = pd.DataFrame([
        {'id': 'acct1', 'type': 'account', 'owner_id': 'user1'},
        {'id': 'user1', 'type': 'user'},
    ])
    edges_df = pd.DataFrame([{'src': 'acct1', 'dst': 'user1'}])
    return CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')


def _sample_ops():
    return [n({'type': 'account'}, name='a'), e_forward(), n(name='c')]


def _sample_where():
    return [compare(col('a', 'owner_id'), '==', col('c', 'owner_id'))]


def test_chain_where_roundtrip():
    chain = Chain(_sample_ops(), where=_sample_where())
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
    g = _sample_graph()
    res = g.gfql(json_chain)
    assert res._nodes is not None


def test_gfql_list_with_where_executes():
    g = _sample_graph()
    res = g.gfql(_sample_ops(), where=_sample_where())
    assert res._nodes is not None


def test_gfql_list_empty_with_where_raises():
    g = _sample_graph()
    with pytest.raises(ValueError, match="empty chains have no aliases"):
        g.gfql([], where=_sample_where())


def test_gfql_list_where_rejects_unsupported_entry_class():
    g = _sample_graph()

    with pytest.raises(ValueError, match=r"where\[0\].*WhereComparison"):
        g.gfql(_sample_ops(), where=[123])


def test_chain_constructor_where_rejects_unsupported_entry_class():
    with pytest.raises(ValueError, match=r"where\[0\].*WhereComparison"):
        Chain(_sample_ops(), where=[object()])


def test_gfql_list_where_mixed_entries_reject_unsupported_entry_class():
    g = _sample_graph()

    valid = compare(col('a', 'owner_id'), '==', col('c', 'owner_id'))
    with pytest.raises(ValueError, match=r"where\[1\].*WhereComparison"):
        g.gfql(_sample_ops(), where=[valid, ('bad', 'entry')])
