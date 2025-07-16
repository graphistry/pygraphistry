"""Test integration of pre-validation with chain() function."""

import pytest
import pandas as pd
from graphistry import edges, nodes
from graphistry.compute.ast import n, e_forward
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError


def test_chain_with_validation_enabled():
    """chain() with validate_schema=True catches errors early."""
    edges_df = pd.DataFrame({
        'src': ['a', 'b'],
        'dst': ['b', 'c']
    })
    nodes_df = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'type': ['person', 'person', 'company']
    })
    
    g = edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')
    
    # Should catch error before execution
    with pytest.raises(GFQLSchemaError) as exc_info:
        g.chain([n({'missing': 'value'})], validate_schema=True)
    
    assert exc_info.value.code == ErrorCode.E301
    assert 'missing' in str(exc_info.value)


def test_chain_without_validation():
    """chain() without validation still works (runtime error)."""
    edges_df = pd.DataFrame({
        'src': ['a', 'b'],
        'dst': ['b', 'c']
    })
    nodes_df = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'type': ['person', 'person', 'company']
    })
    
    g = edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')
    
    # Should raise during execution, not pre-validation
    with pytest.raises(GFQLSchemaError) as exc_info:
        g.chain([n({'missing': 'value'})])  # validate_schema=False by default
    
    assert exc_info.value.code == ErrorCode.E301
    # Error happens during filter_by_dict execution
