# GFQL Remote Metadata Hydration

**Branch**: `feat/gfql-remote-metadata-hydration`
**Created**: 2025-10-17
**Status**: ✅ COMPLETED

## Problem

`gfql_remote()` returns a new Plottable but loses metadata that the server computed during GFQL operations. When remote GFQL includes operations like `call('umap')` that modify bindings (src/dst) or encodings (colors, sizes), the client has no way to know what changed.

**Example scenario:**
```python
g1 = graphistry.nodes(df, 'id')
g2 = g1.gfql_remote(call('umap', {'X': ['x', 'y']}))
# Server ran UMAP, changed bindings, added color encodings
# But g2 has none of this metadata!
```

## Solution

The server's `.metadata` response (JSON) mirrors what arrow uploader sends. We should hydrate this back into the returned Plottable.

**Metadata to hydrate:**
- Bindings: `_node`, `_source`, `_destination`, `_edge`
- Simple encodings: `_point_color`, `_point_size`, `_edge_color`, etc.
- Complex encodings: `_complex_encodings`
- Dataset metadata: `_name`, `_description`
- Style: `_style`

## TDD Plan

### Phase 1: Canvas Expected Behavior (Testing)

**1.1 Integration Test Scenarios**

Create `graphistry/tests/test_gfql_remote_metadata.py`:

```python
class TestGFQLRemoteMetadataHydration:
    """Test that gfql_remote() hydrates server metadata into returned Plottable."""

    def test_umap_bindings_hydrated(self):
        """UMAP changes src/dst - verify bindings transfer back."""
        # Server returns metadata with updated bindings

    def test_umap_encodings_hydrated(self):
        """UMAP adds color encoding - verify complex_encodings transfer back."""

    def test_name_description_hydrated(self):
        """call('name') - verify metadata transfers back."""

    def test_style_hydrated(self):
        """call('style') - verify style transfers back."""

    def test_empty_metadata_doesnt_break(self):
        """No metadata or partial metadata - should not error."""
```

**1.2 Mock Server Responses**

Define expected JSON structures based on arrow uploader format:
```python
MOCK_METADATA = {
    'bindings': {
        'node': 'umap_node_id',
        'source': 'umap_src',
        'destination': 'umap_dst'
    },
    'encodings': {
        'point_color': 'umap_cluster',
        'complex_encodings': {...}
    },
    'metadata': {
        'name': 'UMAP Analysis',
        'description': 'After UMAP'
    },
    'style': {...}
}
```

### Phase 2: Implementation

**2.1 Locate Response Handling**

Files to modify:
- `graphistry/PlotterBase.py::gfql_remote()` - response hydration
- `graphistry/pygraphistry.py` - if needed for deserialization

**2.2 Hydration Logic**

```python
def _hydrate_metadata_from_response(self, metadata: dict) -> 'Plottable':
    """Hydrate Plottable from server metadata response."""
    res = self.bind()

    # Bindings
    if 'bindings' in metadata:
        bindings = metadata['bindings']
        res = res.bind(
            node=bindings.get('node'),
            source=bindings.get('source'),
            destination=bindings.get('destination'),
            edge=bindings.get('edge')
        )

    # Simple encodings
    if 'encodings' in metadata:
        encodings = metadata['encodings']
        res = res.bind(
            point_color=encodings.get('point_color'),
            point_size=encodings.get('point_size'),
            # ... etc
        )

    # Complex encodings
    if 'complex_encodings' in metadata.get('encodings', {}):
        res._complex_encodings = metadata['encodings']['complex_encodings']

    # Metadata
    if 'metadata' in metadata:
        if 'name' in metadata['metadata']:
            res = res.name(metadata['metadata']['name'])
        if 'description' in metadata['metadata']:
            res = res.description(metadata['metadata']['description'])

    # Style
    if 'style' in metadata:
        res = res.style(**metadata['style'])

    return res
```

**2.3 Integration into gfql_remote()**

```python
def gfql_remote(self, query, ...):
    # ... existing code ...
    response_json = response.json()

    # Existing: extract nodes/edges DataFrames
    g = self.edges(...).nodes(...)

    # NEW: Hydrate metadata if present
    if 'metadata' in response_json:
        g = g._hydrate_metadata_from_response(response_json['metadata'])

    return g
```

### Phase 3: Live Testing

**3.1 Local Server Test**
```python
# Start local Graphistry with GFQL support
# Run: pytest graphistry/tests/test_gfql_remote_metadata.py -v
```

**3.2 Manual Integration Test**
```python
g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
g2 = g.gfql_remote(call('umap', {'X': ['x', 'y']}))

# Verify hydrated metadata
assert g2._source == 'umap_src'
assert g2._destination == 'umap_dst'
assert g2._point_color == 'umap_cluster'
print(g2._complex_encodings)
```

**3.3 Real Server Test**
- Deploy to test server
- Run against production GFQL endpoint
- Verify all metadata transfers correctly

## Implementation Checklist

### Phase 1: Testing (TDD)
- [ ] Create `test_gfql_remote_metadata.py`
- [ ] Write `test_umap_bindings_hydrated` (FAILING)
- [ ] Write `test_umap_encodings_hydrated` (FAILING)
- [ ] Write `test_name_description_hydrated` (FAILING)
- [ ] Write `test_style_hydrated` (FAILING)
- [ ] Write `test_empty_metadata_doesnt_break` (FAILING)
- [ ] Define mock server response structures
- [ ] Run tests - confirm all FAIL as expected

### Phase 2: Implementation
- [ ] Locate `gfql_remote()` in PlotterBase.py
- [ ] Implement `_hydrate_metadata_from_response()` helper
- [ ] Integrate hydration into `gfql_remote()` response handling
- [ ] Handle missing/partial metadata gracefully
- [ ] Add logging for metadata hydration
- [ ] Run tests - aim for PASSING

### Phase 3: Validation
- [ ] All unit tests passing
- [ ] Manual integration test with local server
- [ ] Type checking clean (mypy)
- [ ] Linting clean (flake8)
- [ ] Test with real server endpoint
- [ ] Document new behavior in docstrings

## Files to Modify

1. **graphistry/PlotterBase.py**
   - Add `_hydrate_metadata_from_response()` method
   - Update `gfql_remote()` to call hydration

2. **graphistry/tests/test_gfql_remote_metadata.py** (NEW)
   - Comprehensive test suite

3. **docs/source/gfql/remote.rst** (if exists)
   - Document metadata hydration behavior

## Success Criteria

✅ Tests pass showing metadata hydration works
✅ Remote GFQL operations preserve server-computed metadata
✅ Bindings, encodings, name, description, style all transfer
✅ Graceful handling of missing metadata
✅ No regressions in existing gfql_remote() behavior

## Implementation Summary

### What Was Implemented

**Phase 1: TDD - Tests First (COMPLETED)**
- ✅ Created `graphistry/tests/test_gfql_remote_metadata.py` with 12 comprehensive tests
- ✅ All tests initially failed as expected (TDD red phase)
- ✅ Mock server responses defined for UMAP, name/description, style, and full metadata scenarios
- ✅ Edge case tests for malformed metadata, None values, and overriding existing bindings

**Phase 2: Core Implementation (COMPLETED)**
- ✅ Created new `graphistry/io/` module for metadata serialization/deserialization
- ✅ Implemented `graphistry/io/metadata.py` with:
  - `serialize_plottable_metadata()` - extract metadata from Plottable for upload
  - `deserialize_plottable_metadata()` - hydrate metadata from server response
  - Helper functions: `serialize_node_bindings()`, `serialize_edge_bindings()`, etc.
- ✅ Updated `PlotterBase._hydrate_metadata_from_response()` to use new io module (thin wrapper)
- ✅ Integrated hydration into `chain_remote.py` for both JSON and zip formats:
  - JSON format: `if 'metadata' in response: result._hydrate_metadata_from_response(response['metadata'])`
  - Zip format: `if 'gfql_metadata' in metadata.json: result._hydrate_metadata_from_response(...)`

**Phase 3: Validation (COMPLETED)**
- ✅ All 12 new tests passing (100% pass rate)
- ✅ All 13 existing persistence tests passing (zero regressions)
- ✅ Graceful error handling with warnings for malformed metadata
- ✅ Comprehensive docstrings in `io/metadata.py` module

### Key Design Decisions

1. **Unified io/ module**: Created `graphistry/io/metadata.py` to house both serialization (for upload) and deserialization (from response) in one place, ensuring format consistency.

2. **Separation of concerns**: GFQL metadata (`gfql_metadata`) is separate from persistence metadata (`dataset_id`, `privacy`) in zip responses.

3. **Backward compatibility**: Thin wrappers in PlotterBase preserve existing API while using new io module internally.

4. **Graceful degradation**: Missing or partial metadata doesn't break the request - warnings are logged but execution continues.

### Files Created/Modified

**Created:**
- `graphistry/io/__init__.py` - Module exports
- `graphistry/io/metadata.py` - Serialization/deserialization logic (370 lines)
- `graphistry/tests/test_gfql_remote_metadata.py` - Test suite (436 lines, 12 tests)

**Modified:**
- `graphistry/PlotterBase.py` - Refactored `_hydrate_metadata_from_response()` to use io module
- `graphistry/compute/chain_remote.py` - Added metadata hydration to JSON and zip response handlers

### Test Coverage

**New Tests (12 total):**
- ✅ `test_umap_bindings_hydrated` - Bindings transfer from server
- ✅ `test_umap_encodings_hydrated` - Encodings (simple + complex) transfer
- ✅ `test_name_description_hydrated` - Metadata fields transfer
- ✅ `test_style_hydrated` - Style configuration transfers
- ✅ `test_empty_metadata_doesnt_break` - Graceful handling of missing metadata
- ✅ `test_partial_metadata_hydrated` - Partial metadata works
- ✅ `test_full_metadata_hydrated` - Complete metadata scenario
- ✅ `test_metadata_preserves_existing_dataframes` - DataFrames not modified
- ✅ `test_zip_format_metadata_hydrated` - Zip format support
- ✅ `test_malformed_metadata_graceful_handling` - Error resilience
- ✅ `test_metadata_with_none_values` - None value handling
- ✅ `test_metadata_overrides_existing_bindings` - Server metadata takes precedence

**Existing Tests (13 total):**
- ✅ All persistence tests still passing (zero regressions)
