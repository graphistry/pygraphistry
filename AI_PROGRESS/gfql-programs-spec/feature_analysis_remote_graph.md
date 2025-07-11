# Feature Analysis: Remote Graph Loading (RemoteGraph)

## Executive Summary

The RemoteGraph feature proposed in sketch.md introduces the ability to load and query remote graphs directly within pure GFQL expressions, without requiring Python code. This analysis examines its relationship to existing functionality, security implications, and critical performance considerations.

## 1.3.3.1: Relationship to Current Dataset Loading Mechanisms

### Current bind(dataset_id=...) Functionality

The existing PyGraphistry system supports loading remote datasets through the `bind()` method:

```python
# Current approach - requires Python
graphistry.bind(dataset_id="abc123").chain(...)
```

**Current Implementation Details:**
- Located in `PlotterBase.bind()` (line 973)
- Stores dataset_id in the Plottable object
- Requires Python client to orchestrate the loading
- Uses authenticated API calls through `ClientSession`
- Leverages existing upload/download infrastructure

### RemoteGraph vs Current Remote Execution

**Key Differences:**

1. **Execution Context**
   - Current: Python client orchestrates remote operations
   - RemoteGraph: Server-side GFQL runtime loads graphs
   
2. **Composition Model**
   - Current: Sequential chain operations with intermediate Python steps
   - RemoteGraph: Pure GFQL DAG execution without client roundtrips

3. **Authentication Flow**
   - Current: Token passed from Python client on each request
   - RemoteGraph: Token must be embedded or referenced in GFQL context

4. **Data Flow**
   ```
   Current:
   Python → Upload → Server → Query → Download → Python → Next Operation
   
   RemoteGraph:
   GFQL Program → Server loads multiple graphs → Server executes DAG → Single Result
   ```

### Integration with Existing Infrastructure

**Reusable Components:**
- `chain_remote.py` - Remote execution infrastructure
- `ClientSession` - Authentication and session management
- `ArrowUploader` - Data serialization mechanisms
- Dataset permission checks

**New Requirements:**
- Server-side graph loading within GFQL runtime
- DAG execution engine
- Graph reference resolution
- Cross-dataset permission validation

## 1.3.3.2: Security and Authentication Considerations

### Access Control for Remote Graphs

**Current Security Model:**
- Each dataset has owner/organization permissions
- API tokens authenticate requests
- Dataset IDs are opaque identifiers
- Permissions checked on each API call

**RemoteGraph Security Challenges:**

1. **Token Propagation**
   ```json
   {
     "type": "RemoteGraph",
     "graph_id": "abc123"
     // No token field - how to authenticate?
   }
   ```
   
2. **Permission Elevation Risks**
   - User A creates GFQL program referencing their graphs
   - User B executes program - which permissions apply?
   - Need clear execution context boundaries

3. **Dataset Discovery**
   - Remote graphs could probe for dataset existence
   - Information leakage through error messages
   - Timing attacks on permission checks

### Cross-Tenant Data Isolation

**Critical Concerns:**

1. **Memory Isolation**
   - Multiple graphs loaded in same execution context
   - Potential for data leakage between tenants
   - Need strong process isolation

2. **Reference Injection**
   ```json
   {
     "ref": "../../other_tenant/graph"  // Path traversal?
   }
   ```

3. **Compute Resource Isolation**
   - Tenant A's RemoteGraph could consume resources affecting Tenant B
   - Need resource quotas per execution context

### Authentication Token Handling

**Design Options:**

1. **Implicit Token (Current User Context)**
   - Pro: Simple, uses existing auth
   - Con: Limits sharing of GFQL programs
   
2. **Embedded Tokens**
   - Pro: Self-contained programs
   - Con: Security risk, token leakage
   
3. **Token References**
   ```json
   {
     "type": "RemoteGraph",
     "graph_id": "abc123",
     "auth_ref": "@current_user"  // Or "@token:xyz"
   }
   ```

4. **Capability-Based Security**
   - Generate limited-scope tokens for specific graphs
   - Time-bound access grants
   - Audit trail for graph access

### Data Exfiltration Risks

**Attack Vectors:**

1. **Large Graph Loading**
   - Load many graphs to exceed memory
   - Force data to disk, potential inspection
   
2. **Graph Combinators**
   - Union operations could merge unauthorized data
   - Intersection could reveal membership information
   
3. **Error Messages**
   - Graph schema details in errors
   - Row counts, column names exposure

**Mitigations:**
- Strict output sanitization
- Rate limiting on RemoteGraph operations
- Audit logging of all graph access
- Output size limits

## 1.3.3.3: Critical Review - Network/Caching/Error Handling

### Network Latency and Timeout Handling

**Current State:**
- No explicit timeout handling in `chain_remote.py`
- Uses `requests.post()` with default timeouts
- No retry logic for transient failures

**RemoteGraph Challenges:**

1. **Cascading Timeouts**
   - DAG with multiple RemoteGraphs
   - Serial loading could exceed total timeout
   - Need per-operation and total timeouts

2. **Partial Failures**
   ```
   Graph A: Loaded ✓
   Graph B: Timeout ✗
   Graph C: Not attempted
   ```
   - How to handle partially loaded state?
   - Rollback or partial execution?

3. **Network Optimization**
   - Parallel graph loading where possible
   - Connection pooling for same server
   - HTTP/2 multiplexing benefits

**Recommendations:**
```python
{
  "type": "RemoteGraph",
  "graph_id": "abc123",
  "timeout_ms": 30000,  # Per-graph timeout
  "retry_policy": {
    "max_attempts": 3,
    "backoff": "exponential"
  }
}
```

### Caching Strategies

**Cache Levels:**

1. **Graph Metadata Cache**
   - Schema, row counts, column types
   - TTL: Hours to days
   - Key: (user_id, dataset_id, version)

2. **Graph Data Cache**
   - Full graph data in memory/disk
   - TTL: Minutes to hours
   - Key: (user_id, dataset_id, version, filters)

3. **Computation Cache**
   - Results of GFQL operations
   - TTL: Based on data volatility
   - Key: Full GFQL program hash

**Cache Invalidation:**
- Version-based invalidation
- Time-based expiry
- Explicit invalidation API
- Memory pressure eviction

**Implementation Considerations:**
```python
{
  "type": "RemoteGraph",
  "graph_id": "abc123",
  "cache_policy": {
    "mode": "aggressive",  # or "conservative", "none"
    "ttl_seconds": 3600,
    "validate": true      # Check version before use
  }
}
```

### Error Handling

**Error Categories:**

1. **Authentication Errors**
   - Invalid token
   - Expired token
   - Insufficient permissions
   
2. **Network Errors**
   - Connection timeout
   - DNS resolution
   - SSL/TLS failures
   
3. **Data Errors**
   - Graph not found
   - Schema mismatch
   - Corrupted data
   
4. **Resource Errors**
   - Memory exhaustion
   - Disk space
   - Compute quotas

**Error Response Design:**
```json
{
  "error": {
    "type": "RemoteGraphError",
    "code": "GRAPH_NOT_FOUND",
    "graph_ref": "abc123",
    "message": "Dataset not found or access denied",
    "details": {
      "attempted_at": "2024-01-15T10:30:00Z",
      "retry_possible": true
    }
  },
  "partial_results": null  // Or partial data if available
}
```

### Resource Limits and Quotas

**Per-User Limits:**
- Max RemoteGraphs per program: 10
- Max total graph size: 10GB
- Max execution time: 5 minutes
- Max memory per execution: 16GB

**Per-Graph Limits:**
- Max graph size: 2GB
- Max load time: 60 seconds
- Max cache size: 1GB

**Quota Management:**
```python
{
  "type": "QueryDAG",
  "resource_limits": {
    "max_memory_gb": 8,
    "max_time_seconds": 300,
    "max_graphs": 5
  },
  "graph": { ... }
}
```

## Implementation Recommendations

### Phase 1: Minimal Viable Feature
1. Single RemoteGraph support (no DAG)
2. Implicit authentication (current user)
3. No caching
4. Basic timeout handling

### Phase 2: Production Readiness
1. Full DAG support
2. Caching infrastructure
3. Comprehensive error handling
4. Resource quotas

### Phase 3: Advanced Features
1. Cross-tenant graph sharing
2. Capability-based security
3. Advanced cache strategies
4. Graph versioning

## Security Checklist

- [ ] Token validation per graph access
- [ ] Input sanitization for graph IDs
- [ ] Output sanitization for errors
- [ ] Rate limiting implementation
- [ ] Audit logging
- [ ] Resource quota enforcement
- [ ] Memory isolation between tenants
- [ ] Timeout handling at all levels
- [ ] Cache security (no cross-tenant leaks)
- [ ] Permission inheritance model

## Performance Considerations

1. **Baseline Metrics Needed:**
   - Single graph load time
   - Memory usage per graph size
   - Network bandwidth requirements
   - Cache hit rates

2. **Optimization Opportunities:**
   - Parallel graph loading
   - Incremental graph updates
   - Columnar data transfer
   - Compression strategies

3. **Monitoring Requirements:**
   - Graph load latencies
   - Cache performance
   - Resource usage per tenant
   - Error rates by category

## Conclusion

RemoteGraph represents a significant evolution from the current dataset loading mechanism, enabling pure GFQL programs to compose multiple graphs without Python orchestration. However, this power comes with substantial security and performance challenges that must be carefully addressed. The phased implementation approach allows for iterative refinement while maintaining system stability and security.