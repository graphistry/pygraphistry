# User Scenario 1: Remote UMAP + Immediate Visualization

## Current User Pain Point

**Goal**: Run UMAP dimensionality reduction on large dataset remotely and immediately share visualization

**Current Workflow (Inefficient)**:
```python
import graphistry
import pandas as pd

# Large dataset - 50k+ nodes
df = pd.read_csv('large_graph_data.csv')
g = graphistry.edges(df, 'src', 'dst')

# Step 1: Remote UMAP transformation
result = g.gfql_remote([
    call('umap', {'n_components': 2, 'n_neighbors': 15}),
    call('name', {'name': 'Customer Journey UMAP'}),
    call('description', {'description': 'UMAP visualization of customer interactions'})
])

# Step 2: Client round-trip to create visualization
url = result.plot()  # ❌ UPLOADS DATA AGAIN! Inefficient for large datasets

# Step 3: Share URL
print(f"Share this visualization: {url}")
```

**Problems**:
1. **Double data transfer**: Data goes client → server → client → server
2. **Slow feedback**: Large datasets take time for round-trip
3. **Resource waste**: Server computes UMAP, client downloads, then uploads again

## Desired Workflow (With Persistence)

**Option A: call('save') in GFQL chain**
```python
# Single server-side operation with persistence
result = g.gfql_remote([
    call('umap', {'n_components': 2, 'n_neighbors': 15}),
    call('name', {'name': 'Customer Journey UMAP'}),
    call('description', {'description': 'UMAP visualization of customer interactions'}),
    call('save')  # 🎯 NEW: Persist dataset on server
])

# Result includes both data AND dataset_id for immediate URL generation
print(f"Dataset ID: {result._dataset_id}")
print(f"Visualization URL: {result.url()}")  # 🎯 NEW: Generate URL from dataset_id
```

**Option B: persist parameter**
```python
# Alternative syntax with parameter
result = g.gfql_remote([
    call('umap', {'n_components': 2, 'n_neighbors': 15}),
    call('name', {'name': 'Customer Journey UMAP'}),
    call('description', {'description': 'UMAP visualization of customer interactions'})
], persist=True)  # 🎯 NEW: Enable server-side persistence

print(f"Dataset ID: {result._dataset_id}")
print(f"Visualization URL: {result.url()}")
```

## Benefits

1. **Single server operation**: Transform + persist in one call
2. **Immediate sharing**: URL available without client upload
3. **Efficient**: No unnecessary data transfers
4. **Familiar**: Mirrors existing `g.upload()` semantics

## Technical Requirements

1. **Server persistence**: Store transformed dataset with generated dataset_id
2. **Enhanced return format**: Include dataset_id in response
3. **URL generation**: Client method to create visualization URL from dataset_id
4. **Backward compatibility**: Non-persist calls work as before

## Expected User Experience

- **Before**: 3 steps, multiple round-trips, slow for large data
- **After**: 1 step, server-side efficiency, immediate URL sharing
- **Intuitive**: Follows established upload() patterns
- **Flexible**: Data still available for further local processing