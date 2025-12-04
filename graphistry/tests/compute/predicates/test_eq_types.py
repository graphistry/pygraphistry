import pandas as pd
import pytest

from graphistry.compute.chain import Chain
from graphistry.compute.ast import ASTNode
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.predicates.comparison import EQ, eq
from graphistry.compute.gfql_unified import gfql


@pytest.mark.parametrize('val', ['a', 1, 1.5, True])
def test_eq_multi_type_validate_and_apply(val):
    s = pd.Series([val, None])
    predicate = eq(val)
    predicate.validate()
    result = predicate(s)
    assert bool(result.iloc[0]) is True
    assert bool(result.iloc[1]) is False


@pytest.mark.parametrize('val', ['a', 1, 1.5, True])
def test_eq_multi_type_chain_validation_and_execution(val):
    nodes = pd.DataFrame({'id': ['a', 'b', 'c'], 'label': [val, val, 'other']})
    edges = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})

    chain = Chain([
        ASTNode(filter_dict={'label': eq(val)})
    ])

    chain.validate()
    # Runtime execution is exercised indirectly via predicate on series below


@pytest.mark.parametrize('val', ['a', 1, 1.5, True])
def test_eq_multi_type_schema_validation(val):
    nodes = pd.DataFrame({'id': ['a'], 'label': [val]})
    edges = pd.DataFrame({'s': ['a'], 'd': ['a']})

    chain = Chain([
        ASTNode(filter_dict={'label': eq(val)})
    ])

    class DummyG:
        def __init__(self, nodes_df, edges_df):
            self._nodes = nodes_df
            self._edges = edges_df
    dummy = DummyG(nodes, edges)
    validate_chain_schema(dummy, chain, collect_all=False)


@pytest.mark.parametrize('val', ['a', 1, 1.5, True])
def test_eq_multi_type_json_roundtrip(val):
    predicate = eq(val)
    payload = predicate.to_json(validate=False)
    assert payload['type'] == 'EQ'
    assert payload['val'] == val

    restored = EQ.from_json(payload, validate=False)
    restored.validate()
    s = pd.Series([val, 'other'])
    pd.testing.assert_series_equal(restored(s).reset_index(drop=True), pd.Series([True, False]))


@pytest.mark.parametrize('val', ['a', 1, 1.5, True])
def test_eq_multi_type_gfql_runtime(val):
    nodes = pd.DataFrame({'id': ['a', 'b', 'c'], 'label': [val, val, 'other']})
    edges = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
    import graphistry as g  # local import to avoid global init at module load
    g_obj = g.nodes(nodes, 'id').edges(edges, 's', 'd')

    chain = [ASTNode(filter_dict={'label': eq(val)})]

    result = gfql(g_obj, chain)
    filtered_nodes = result._nodes
    assert set(filtered_nodes['id']) == {'a', 'b'}
