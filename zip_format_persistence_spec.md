# Zip Format Persistence Specification for Backend Team

## Overview

This specification extends the existing remote GFQL persistence feature to support zip format responses (parquet/CSV with `output_type="all"`), eliminating the current JSON-only limitation.

## Current Limitation

**Problem**: When `persist=True` is used with `format="parquet"/"csv"` and `output_type="all"`, the server returns a zip file containing nodes and edges data, but no mechanism exists to return the `dataset_id` needed for client-side URL generation.

**Current Behavior**:
- ✅ JSON format: Returns `{"nodes": [...], "edges": [...], "dataset_id": "abc123"}`
- ❌ Zip format: Returns zip with `nodes.parquet` and `edges.parquet` only

## Solution: metadata.json in Zip

### Enhanced Server Response Format

When `persist=True` is included in the request body alongside zip format requests, the server should include an additional `metadata.json` file in the zip response.

#### Request Format (Unchanged)
```json
POST /api/v2/etl/datasets/{dataset_id}/gfql/all
{
  "gfql_operations": [...],
  "format": "parquet",
  "persist": true,
  "privacy": {
    "mode": "organization",
    "notify": true,
    "invited_users": ["user@example.com"]
  }
}
```

#### Enhanced Zip Response Structure
```
response.zip
├── nodes.parquet          # Existing: nodes DataFrame
├── edges.parquet          # Existing: edges DataFrame
└── metadata.json          # NEW: persistence metadata
```

#### metadata.json Format
```json
{
  "dataset_id": "abc123def456",
  "persist": true,
  "created_at": "2025-10-05T12:34:56.789Z",
  "format": "parquet",
  "privacy": {
    "mode": "organization",
    "notify": true,
    "invited_users": ["user@example.com"]
  }
}
```

### Implementation Requirements

#### Required Fields in metadata.json
- `dataset_id` (string): The persisted dataset identifier for URL generation
- `persist` (boolean): Always `true` when this file is present
- `created_at` (string): ISO 8601 timestamp of persistence creation

#### Optional Fields in metadata.json
- `format` (string): Original requested format ("parquet", "csv")
- `privacy` (object): Privacy/share settings applied to the dataset
- `name` (string): Dataset name if set via GFQL operations
- `description` (string): Dataset description if set via GFQL operations

#### Server Logic Changes

```python
# Pseudo-code for server implementation
def create_gfql_zip_response(nodes_df, edges_df, format, persist=False, **kwargs):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_ref:
        # Existing logic: add data files
        if format == "parquet":
            zip_ref.writestr('nodes.parquet', nodes_df.to_parquet())
            zip_ref.writestr('edges.parquet', edges_df.to_parquet())
        elif format == "csv":
            zip_ref.writestr('nodes.csv', nodes_df.to_csv())
            zip_ref.writestr('edges.csv', edges_df.to_csv())

        # NEW: Add metadata when persist=True
        if persist:
            dataset_id = create_persistent_dataset(nodes_df, edges_df, **kwargs)
            metadata = {
                'dataset_id': dataset_id,
                'persist': True,
                'created_at': datetime.utcnow().isoformat() + 'Z',
                'format': format
            }

            # Include privacy settings if provided
            if 'privacy' in kwargs:
                metadata['privacy'] = kwargs['privacy']

            zip_ref.writestr('metadata.json', json.dumps(metadata))

    return zip_buffer.getvalue()
```

### Backward Compatibility

#### Client Behavior
- **New servers** (with metadata.json): Extract dataset_id for URL generation
- **Old servers** (without metadata.json): Graceful degradation, no URL generation
- **No breaking changes**: Existing zip parsing continues to work

#### Server Behavior
- **persist=True + zip format**: Include metadata.json in response
- **persist=False + zip format**: No metadata.json (existing behavior)
- **persist=True + JSON format**: Use existing JSON response format (unchanged)

### Testing Requirements

#### Server-Side Tests
1. **Zip with metadata**: Verify metadata.json is included when persist=True
2. **Zip without metadata**: Verify metadata.json is absent when persist=False
3. **Metadata content**: Verify all required fields are present and correct
4. **Privacy integration**: Verify privacy settings are included in metadata
5. **Multiple formats**: Test with both parquet and CSV formats

#### Client Integration Tests
1. **Metadata present**: Client extracts dataset_id successfully
2. **Metadata absent**: Client handles gracefully (no URL generation)
3. **Malformed metadata**: Client handles JSON parsing errors gracefully
4. **Privacy preservation**: Privacy settings are restored on client Plottable

## Implementation Timeline

### Phase 1: Core Implementation (1-2 sprints)
- Server zip response enhancement with metadata.json
- Client parsing logic for metadata extraction
- Basic unit tests

### Phase 2: Enhanced Features (1 sprint)
- Privacy settings integration in metadata
- Extended metadata fields (name, description)
- Comprehensive integration testing

### Phase 3: Production Readiness (1 sprint)
- Error handling and edge cases
- Performance optimization
- Documentation updates

## Error Handling

### Server-Side Errors
- **Persistence failure**: Return standard zip without metadata.json
- **Metadata serialization error**: Log warning, continue with data-only zip
- **Storage system unavailable**: Fallback to non-persistent response

### Client-Side Errors
- **Missing metadata.json**: Continue with existing behavior (no URL generation)
- **Malformed JSON**: Log warning, continue without dataset_id
- **Missing dataset_id field**: Handle gracefully, no URL generation

## Security Considerations

- **Privacy settings**: Ensure metadata privacy matches actual dataset privacy
- **Dataset_id exposure**: Validate that returned dataset_id is properly scoped to user
- **Metadata tampering**: Client should validate dataset_id format/structure

## Success Metrics

1. **Feature Completeness**: persist=True works with all format combinations
2. **Backward Compatibility**: No regressions with existing servers
3. **Performance**: Metadata addition adds <1% to response time
4. **Adoption**: Users can generate URLs immediately with zip format responses

---

**Questions/Clarifications**: Please reach out for any implementation details or edge cases not covered in this specification.