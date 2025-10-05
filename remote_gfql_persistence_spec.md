# Remote GFQL Persistence Specification

## Overview

This specification defines minimal changes to enable server-side persistence for remote GFQL operations, allowing users to get dataset_id immediately without client-side upload round-trips.

## Phase 1: persist=True Parameter (Immediate Implementation)

### Client-Side Changes

#### 1. Enhanced gfql_remote() Method Signature

**File**: `graphistry/compute/ComputeMixin.py`

```python
def gfql_remote(self, query,
                api_token=None,
                dataset_id=None,
                output_type="all",
                format=None,
                df_export_args=None,
                node_col_subset=None,
                edge_col_subset=None,
                engine=None,
                validate=True,
                persist=False) -> Union[Plottable, str]:  # 🎯 NEW parameter
    """
    Enhanced gfql_remote with optional server-side persistence.

    :param persist: If True, persist dataset on server and return dataset_id
    :type persist: bool
    :returns: Plottable with _dataset_id set if persist=True
    :rtype: Union[Plottable, str]
    """
```

#### 2. Enhanced chain_remote_generic() Implementation

**File**: `graphistry/compute/chain_remote.py`

```python
def chain_remote_generic(
    self: Plottable,
    chain: Union[Chain, Dict[str, JSONVal], List[Any]],
    api_token: Optional[str] = None,
    dataset_id: Optional[str] = None,
    output_type: OutputTypeGraph = "all",
    format: Optional[FormatType] = None,
    df_export_args: Optional[Dict[str, Any]] = None,
    node_col_subset: Optional[List[str]] = None,
    edge_col_subset: Optional[List[str]] = None,
    engine: Optional[Literal["pandas", "cudf"]] = None,
    validate: bool = True,
    persist: bool = False  # 🎯 NEW parameter
) -> Union[Plottable, pd.DataFrame]:
    """Enhanced with persistence support."""

    # Add persist flag to request body
    request_body = {
        "gfql_operations": chain_json['chain'],
        "format": format,
        "persist": persist  # 🎯 NEW field
    }

    # Handle enhanced response format
    if persist:
        # Parse response with dataset_id
        response_data = response.json()
        result_plottable = self.edges(edges_df).nodes(nodes_df)
        result_plottable._dataset_id = response_data.get('dataset_id')  # 🎯 NEW
        return result_plottable
    else:
        # Current behavior unchanged
        return current_processing_logic()
```

#### 3. URL Generation Method

**File**: `graphistry/PlotterBase.py` or `graphistry/Plottable.py`

```python
def url(self, **url_params) -> Optional[str]:
    """Generate visualization URL from dataset_id.

    :param url_params: Optional URL parameters to include
    :returns: Visualization URL if dataset_id is set
    :rtype: Optional[str]
    """
    if not hasattr(self, '_dataset_id') or not self._dataset_id:
        return None

    # Reuse existing _viz_url logic
    info = {
        'name': self._dataset_id,
        'type': 'arrow',  # or detect from metadata
        'viztoken': str(uuid.uuid4())
    }

    return self._pygraphistry._viz_url(info, url_params)
```

### Server-Side Changes (Minimal)

#### 1. API Endpoint Enhancement

**Endpoint**: `POST /api/v2/etl/datasets/{dataset_id}/gfql/{output_type}`

**Enhanced Request Body**:
```json
{
  "gfql_operations": [...],
  "format": "parquet",
  "persist": true  // 🎯 NEW field
}
```

**Enhanced Response Format** (when persist=true):
```json
{
  "nodes": { ... },
  "edges": { ... },
  "dataset_id": "abc123def456",  // 🎯 NEW field
  "dataset_url": "https://hub.graphistry.com/graph/graph.html?dataset=abc123def456",  // 🎯 OPTIONAL
  "metadata": {
    "name": "...",
    "description": "..."
  }
}
```

#### 2. Server-Side Logic Changes

```python
# Pseudo-code for server-side implementation
def handle_gfql_request(request_body):
    # Execute GFQL operations (existing logic)
    result_graph = execute_gfql_operations(request_body['gfql_operations'])

    if request_body.get('persist', False):
        # 🎯 NEW: Persist dataset
        dataset_id = store_dataset(result_graph)

        # Return enhanced response
        return {
            'nodes': result_graph.nodes,
            'edges': result_graph.edges,
            'dataset_id': dataset_id,  # 🎯 NEW
            'metadata': extract_metadata(result_graph)
        }
    else:
        # Current behavior unchanged
        return {
            'nodes': result_graph.nodes,
            'edges': result_graph.edges
        }
```

## Phase 2: call('save') Operations (Future Enhancement)

### 1. GFQL Safelist Addition

**File**: `graphistry/compute/gfql/call_safelist.py`

```python
SAFELIST_V1['save'] = {
    'allowed_params': {'name', 'description'},
    'required_params': set(),
    'param_validators': {
        'name': is_string_or_none,
        'description': is_string_or_none
    },
    'description': 'Persist dataset on server for immediate access'
}
```

### 2. Save Operation Executor

**File**: `graphistry/compute/gfql/call_executor.py`

```python
def execute_save_operation(g: Plottable, params: Dict[str, Any], engine: Engine) -> Plottable:
    """Execute save operation - mark for server-side persistence."""
    # Set persistence marker on Plottable
    result = g.bind()
    result._persist_requested = True
    result._persist_metadata = params
    return result
```

## Testing Strategy

### 1. Unit Tests

**File**: `graphistry/tests/test_chain_remote_persistence.py`

```python
class TestRemoteGFQLPersistence:

    def test_persist_parameter_false_default(self):
        """Test that persist=False maintains current behavior."""
        # Mock server response without dataset_id
        with mock_server_response(standard_response):
            result = g.gfql_remote([n()], persist=False)
            assert not hasattr(result, '_dataset_id')

    def test_persist_parameter_true(self):
        """Test that persist=True returns dataset_id."""
        # Mock server response with dataset_id
        with mock_server_response(persistence_response):
            result = g.gfql_remote([n()], persist=True)
            assert hasattr(result, '_dataset_id')
            assert result._dataset_id == 'test_dataset_123'

    def test_url_generation_with_dataset_id(self):
        """Test URL generation from dataset_id."""
        result = create_mock_plottable_with_dataset_id('test_123')
        url = result.url()
        assert 'dataset=test_123' in url
        assert 'graph.html' in url

    def test_url_generation_without_dataset_id(self):
        """Test URL generation fails gracefully without dataset_id."""
        result = create_mock_plottable()
        url = result.url()
        assert url is None
```

### 2. Integration Tests (Server Conformance)

**File**: `graphistry/tests/test_server_conformance_persistence.py`

```python
class TestServerPersistenceConformance:
    """Tests that work with both mocked and live servers."""

    @pytest.mark.parametrize("server_type", ["mock", "live"])
    def test_persistence_end_to_end(self, server_type):
        """Test complete persistence workflow."""
        if server_type == "live" and not has_live_server():
            pytest.skip("Live server not available")

        with server_context(server_type):
            # Test complete workflow
            result = g.gfql_remote([call('materialize_nodes')], persist=True)

            # Verify persistence response
            assert hasattr(result, '_dataset_id')
            assert result._dataset_id is not None

            # Verify URL generation
            url = result.url()
            assert url is not None
            assert result._dataset_id in url

    def test_persistence_with_metadata(self):
        """Test that metadata is preserved in persistence."""
        result = g.gfql_remote([
            call('materialize_nodes'),
            call('name', {'name': 'Test Dataset'}),
            call('description', {'description': 'Test description'})
        ], persist=True)

        assert result._name == 'Test Dataset'
        assert result._description == 'Test description'
        assert hasattr(result, '_dataset_id')
```

### 3. Mock Server Responses

**File**: `graphistry/tests/mocks/mock_persistence_responses.py`

```python
PERSISTENCE_RESPONSE = {
    "nodes": mock_nodes_data,
    "edges": mock_edges_data,
    "dataset_id": "mock_dataset_123",
    "dataset_url": "https://hub.graphistry.com/graph/graph.html?dataset=mock_dataset_123",
    "metadata": {
        "name": "Test Dataset",
        "description": "Mock dataset for testing"
    }
}

def mock_persistence_server():
    """Mock server that returns persistence responses."""
    return MockServer({
        'POST /api/v2/etl/datasets/*/gfql/all': {
            'response': PERSISTENCE_RESPONSE,
            'condition': lambda req: req.json().get('persist', False)
        }
    })
```

## Backward Compatibility

### 1. Default Behavior Unchanged
- `gfql_remote()` without `persist=True` works exactly as before
- No breaking changes to existing API
- Server handles missing `persist` field gracefully

### 2. Migration Path
```python
# Current code continues to work
result = g.gfql_remote([call('umap')])  # No changes needed

# New functionality is opt-in
result = g.gfql_remote([call('umap')], persist=True)  # New feature
```

## Performance Considerations

### 1. Server-Side Storage
- Reuse existing dataset storage mechanisms
- No new storage infrastructure required
- Leverage existing dataset lifecycle management

### 2. Response Size
- When persist=True, response includes dataset_id (minimal overhead)
- Data payload unchanged (same nodes/edges data)
- Optional dataset_url reduces client-side URL generation load

## Error Handling

### 1. Persistence Failures
```python
try:
    result = g.gfql_remote([...], persist=True)
except PersistenceError as e:
    # Graceful fallback
    logger.warning(f"Persistence failed: {e}")
    result = g.gfql_remote([...], persist=False)  # Fallback
```

### 2. Server Compatibility
- Graceful degradation if server doesn't support persistence
- Clear error messages for unsupported features
- Version detection for feature availability

## Implementation Timeline

### Phase 1 (Immediate): persist=True Parameter
- **Week 1**: Client-side implementation
- **Week 2**: Testing and mocking infrastructure
- **Week 3**: Server conformance specifications
- **Week 4**: Integration testing and documentation

### Phase 2 (Future): call('save') Operations
- **Month 2**: GFQL safelist additions
- **Month 2**: Save operation execution logic
- **Month 3**: Advanced persistence features
- **Month 3**: Performance optimization

## Success Metrics

1. **User Experience**: Single API call enables dataset persistence
2. **Performance**: Eliminate client upload round-trips
3. **Adoption**: Clear migration path from current patterns
4. **Reliability**: Comprehensive test coverage for persistence scenarios