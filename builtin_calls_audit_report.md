# GFQL Built-in Calls Documentation Audit Report

## Executive Summary

The `builtin_calls.rst` documentation contains significant inaccuracies and is missing critical methods. Of the 12 methods documented, 4 are not in the safelist (33% error rate), and 16 important methods from the safelist are completely missing from the documentation.

## 1. Accuracy Issues - Methods NOT in Safelist

### ❌ pagerank
- **Status**: NOT IN SAFELIST
- **Issue**: Documented as `call('pagerank')` but this is incorrect
- **Correct Usage**: Should be `call('compute_cugraph', {'alg': 'pagerank'})`
- **Location**: Lines 38-44
- **Examples affected**: Lines 20, 26, 42-44, 179, 189, 199, 221, 226

### ❌ umap
- **Status**: NOT IN SAFELIST
- **Issue**: Documented as a direct call method
- **Reality**: UMAP is a feature/capability, not a direct call method
- **Location**: Lines 87-93

### ❌ add_graph
- **Status**: NOT IN SAFELIST
- **Issue**: Method does not exist in safelist
- **Location**: Lines 114-119

### ❌ sample
- **Status**: NOT IN SAFELIST
- **Issue**: Method does not exist in safelist
- **Location**: Lines 138-143

## 2. Correctly Documented Methods (IN SAFELIST ✓)

- ✓ get_degrees (lines 46-53)
- ✓ filter_nodes_by_dict (lines 58-63)
- ✓ filter_edges_by_dict (lines 65-70)
- ✓ hop (lines 75-82)
- ✓ fa2_layout (lines 95-101)
- ✓ materialize_nodes (lines 106-112)
- ✓ prune_self_edges (lines 121-126)
- ✓ name (lines 131-136)

## 3. Missing Critical Methods from Safelist

### Graph Algorithm Methods
- **compute_cugraph** - Critical for GPU algorithms (pagerank, louvain, etc.)
- **compute_igraph** - For igraph-based algorithms

### Layout Methods
- **layout_cugraph** - GPU-accelerated layouts
- **layout_igraph** - igraph-based layouts
- **layout_graphviz** - Graphviz layouts (dot, neato, etc.)

### Degree Analysis Methods
- **get_indegrees** - Calculate in-degrees only
- **get_outdegrees** - Calculate out-degrees only

### Visual Encoding Methods
- **encode_point_color** - Map node values to colors
- **encode_edge_color** - Map edge values to colors
- **encode_point_size** - Map node values to sizes
- **encode_point_icon** - Map node values to icons

### Graph Transformation Methods
- **collapse** - Collapse nodes by shared attributes
- **drop_nodes** - Remove specified nodes
- **keep_nodes** - Keep only specified nodes

### Analysis Methods
- **get_topological_levels** - DAG analysis
- **description** - Set visualization description

## 4. Documentation Quality Issues

### Missing Parameter Tables
The documentation lacks structured parameter tables showing:
- Required vs optional parameters
- Parameter types
- Default values
- Valid value ranges/options

### Incorrect Examples
All examples using `call('pagerank')` are wrong and should use:
```python
call('compute_cugraph', {'alg': 'pagerank'})
```

### Missing Safelist Validation Details
No mention of:
- How validation works
- Error codes (E303, E105, E201)
- What happens with invalid parameters

### Misleading GPU Acceleration Section
Lists methods as GPU-accelerated that either:
- Don't exist in safelist (pagerank)
- May not actually be GPU-accelerated (filter operations)

## 5. Recommendations

1. **Immediate Fixes Required**:
   - Replace all `pagerank` references with `compute_cugraph` examples
   - Remove non-existent methods (umap, add_graph, sample)
   - Add all missing methods from safelist

2. **Add Parameter Documentation**:
   - Create tables for each method showing parameters
   - Include type information and validation rules
   - Show which parameters are required vs optional

3. **Improve Examples**:
   - Fix all incorrect pagerank examples
   - Add examples for compute_cugraph with different algorithms
   - Show proper error handling

4. **Add Validation Section**:
   - Explain safelist validation
   - Document error codes and their meanings
   - Show how to handle validation errors

5. **Clarify GPU Acceleration**:
   - Only list methods that actually support GPU
   - Explain which algorithms are available through compute_cugraph
   - Remove misleading claims about filter operations

## 6. Example of Corrected Documentation

Instead of:
```python
call('pagerank')
call('pagerank', {'damping': 0.85, 'iterations': 20})
```

Should be:
```python
call('compute_cugraph', {'alg': 'pagerank'})
call('compute_cugraph', {'alg': 'pagerank', 'params': {'damping': 0.85, 'max_iter': 20}})
```

## 7. Statistics

- **Total methods documented**: 12
- **Methods not in safelist**: 4 (33%)
- **Methods correctly documented**: 8 (67%)
- **Methods missing from docs**: 16
- **Total methods in safelist**: 24
- **Documentation coverage**: 8/24 (33%)