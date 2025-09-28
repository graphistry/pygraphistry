# Policy Implementation Plan

## Current Status
- Branch: `feat/gfql-policy-hooks`
- **71 tests passing** (3 skipped for cudf)
- **PR #754**: Updated and ready for review
- **Core Policy Implementation**: ✅ COMPLETE - Simplified to Accept/Deny only
- **Remote/JWT Investigation**: 🔄 IN PROGRESS

## ✅ Investigation Phase COMPLETE
- Found 3 modification types (engine, query, params)
- Identified ~20 tests using modifications
- Documented all impact areas

## ✅ Implementation Phase COMPLETE

### 3. Remove Modification Logic ✅
- ✅ Removed PolicyModification TypedDict
- ✅ Updated PolicyFunction type to return None only
- ✅ Deleted validation.py file entirely
- ✅ Removed modification handling from:
  - gfql_unified.py (removed lines 205-219)
  - chain.py (removed lines 524-542)
  - chain_let.py (removed lines 434-456)
  - call_executor.py (removed lines 58-73)
- ✅ Updated __init__.py exports

### 4. Simplify PolicyContext ✅
- ✅ Kept all fields for inspection (still useful)

### 5. Update Exception Handling ✅
- ✅ PolicyException is now the only control mechanism
- ✅ Enrichment logic intact and working

## ✅ Testing Phase COMPLETE

### 6. Update Tests ✅
- ✅ test_policy_exceptions.py - Updated signatures
- ✅ test_policy_hooks.py - Removed modifications
- ✅ test_policy_validation.py - Completely rewritten for exceptions
- ✅ test_policy_behavior_modification.py - Rewritten for accept/deny
- ✅ test_policy_integration.py - Fixed imports and signatures
- ✅ test_policy_recursion.py - Fixed imports and signatures
- ✅ test_policy_closure_state.py - Fixed imports and signatures

### 7. New Test Patterns ✅
- ✅ Accept/Deny patterns implemented
- ✅ State tracking without modifications tested
- ✅ Clean error messages verified

## ✅ Documentation Phase COMPLETE

### 8. Documentation Updates ✅
- ✅ Updated PR #754 description with simplified design
- ✅ Updated all example code to show Accept/Deny pattern
- ✅ Updated docstrings in types.py
- ✅ Created FINAL_SUMMARY.md with complete documentation

## ✅ Rollout Phase COMPLETE

### 9. Final Validation ✅
- ✅ All 71 tests passing (including related tests)
- ✅ CI green
- ✅ No breaking changes - policy parameter is optional

## Key Achievements

### Removed ~200 Lines of Complex Code:
- PolicyModification TypedDict
- validation.py (entire file)
- All modification handling logic
- Complex edge case handling

### Simplified to Clean Pattern:
```python
def policy(context: PolicyContext) -> None:
    if not_allowed:
        raise PolicyException(phase, reason, code)
    # Implicit accept
```

### Test Results:
- 46 policy tests passing
- 25 related tests passing
- 3 skipped (require cudf)
- Total: 71 passing tests

## Commits Made:
1. `4080b2d7` - Simplify policy system to Accept/Deny only
2. `5732ae26` - Update first 4 test files for Accept/Deny pattern
3. `b20a83be` - Complete policy simplification

## PR Status:
- **PR #754**: https://github.com/graphistry/pygraphistry/pull/754
- **Status**: Ready for review
- **Description**: Updated with simplified design
- **CI**: Green ✅

## 🔄 Remote Data Loading & JWT Investigation

### Dynamic Data Loading Methods Found:
1. **ASTRemoteGraph**: Loads remote datasets by dataset_id
   - Accepts optional JWT token parameter
   - Used in chain_let.py for remote dataset references

2. **gfql_remote()**: Remote execution of GFQL queries
   - JWT handled via api_token parameter
   - Falls back to session.api_token if not provided
   - Uploads data if no dataset_id present

3. **ASTRef**: References to DAG bindings (not remote)
   - Used for local DAG references only

### JWT Token Flow:
1. **Current Implementation**:
   - `gfql_remote()` accepts `api_token` parameter
   - Falls back to `self.session.api_token` if not provided
   - Token passed as Bearer token in Authorization header
   - PyGraphistry client can be initialized with JWT

2. **Hub Integration Options**:
   - **Option A**: Hub pre-configures PyGraphistry client with JWT
     ```python
     import graphistry
     graphistry.register(api_token=user_jwt)
     g.gfql_remote(...)  # Uses registered JWT
     ```

   - **Option B**: Hub passes JWT per-call
     ```python
     g.gfql_remote(query, api_token=user_jwt)
     ```

### Policy Hook Considerations for Remote Data:

1. **Current State**:
   - Policy hooks work with local gfql() execution
   - Remote calls (gfql_remote) bypass local policies entirely
   - ASTRemoteGraph loads happen during chain execution:
     - JWT taken from AST object: `api_token=ast_obj.token`
     - Calls chain_remote() internally to fetch data
     - No policy hooks invoked for this remote fetch

2. **Important Finding**: ASTRemoteGraph Loading Gap
   - When Hub uses `g.gfql(ASTRemoteGraph(...))` locally
   - The remote dataset is fetched WITHOUT policy hooks
   - This is a **potential security gap** if Hub needs to control remote loads

3. **Options for Hub**:

   **Option A - Current Approach (No Changes Needed)**:
   - Hub uses server-side policies for all remote operations
   - Client-side policies only for local data transformations
   - JWT auth provides user context and access control

   **Option B - Add Remote Load Hooks (Future Enhancement)**:
   - Could add policy hooks before/after ASTRemoteGraph fetch
   - Would allow Hub to control which remote datasets can be loaded
   - Not currently implemented but could be added if needed

4. **Recommendation**:
   - For now, use Option A - server-side control is sufficient
   - Hub should pre-configure PyGraphistry client with user JWT
   - If Hub needs client-side control of remote loads, we can add hooks later

## 🔄 Remote/Network Operation Policy Support

### Network Operations Survey:
1. **ASTRemoteGraph** - Main concern
   - Fetches remote datasets by dataset_id
   - Makes network call via chain_remote()
   - Can load arbitrarily large graphs
   - NO policy hooks currently

2. **gfql_remote()** - Server-side execution
   - Entire query runs remotely
   - Already has server-side control
   - Less critical for client-side policies

3. **upload()** - Data upload
   - Called implicitly if no dataset_id
   - Network operation but user-initiated
   - Less critical as it's pushing, not pulling

### Design Decision: Generic vs Specific Hooks

**Option A: Specific Remote Hooks**
```python
policy = {
    'preload': ...,      # Before local data load
    'postload': ...,     # After local data load
    'remote_preload': ...,  # Before remote fetch
    'remote_postload': ..., # After remote fetch
    'call': ...          # Before method calls
}
```

**Option B: Generic Network Hooks**
```python
policy = {
    'preload': ...,     # Before ANY data load (local or remote)
    'postload': ...,    # After ANY data load (local or remote)
    'call': ...         # Before method calls
}
# Context indicates if it's remote via 'is_remote': True
```

**Recommendation: Option B - Generic Hooks**
- More future-proof
- Simpler API surface
- Context can differentiate: `context['is_remote'] = True`
- Can add `context['remote_dataset_id']` for remote loads

### Implementation Plan:

1. **Update chain_let.py for ASTRemoteGraph**:
   - Add preload hook BEFORE chain_remote() call
   - Add postload hook AFTER chain_remote() returns
   - Include remote-specific context fields

2. **Context Extensions for Remote**:
```python
# Remote preload context
{
    'phase': 'preload',
    'is_remote': True,
    'remote_dataset_id': 'xyz-123',
    'remote_token': 'jwt...' if provided else None,
    'operation': 'ASTRemoteGraph'
}

# Remote postload context
{
    'phase': 'postload',
    'is_remote': True,
    'remote_dataset_id': 'xyz-123',
    'graph_stats': {...},  # Stats from fetched data
    'operation': 'ASTRemoteGraph'
}
```

3. **Testing Strategy**:
   - Mock chain_remote() to avoid real network calls
   - Test policy accepts/denies remote loads
   - Test context fields are properly set
   - Test both with and without JWT tokens

4. **Questions for User**:
   - ✅ Should we use generic hooks (recommended) or specific remote hooks?
   - ✅ Should upload() operations also trigger policies? (recommend: no)
   - Any other network operations to consider?

### Next Steps:
1. ✅ Survey complete - found ASTRemoteGraph as main gap
2. ⏳ Implement generic preload/postload for ASTRemoteGraph
3. ⏳ Add is_remote and related context fields
4. ⏳ Write tests with mocked network calls
5. ⏳ Update documentation

## Summary:
✅ Core policy system complete and simplified to Accept/Deny only
✅ JWT handling already supported via api_token parameter
🔄 Remote data loading policy support in progress
⏳ Implementing generic hooks for ASTRemoteGraph

The core implementation is complete. Remote policy support is being added.