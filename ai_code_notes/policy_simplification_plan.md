# Policy Simplification Plan: Remove Non-None Returns

## Current Status
- Branch: `feat/gfql-policy-hooks`
- All 49 tests passing
- PR #754 open
- **Issue**: Non-None returns (modifications) are overly complex and have shaky semantics

## Goal
Simplify policy system to Accept/Deny only (no modifications)

## Investigation Phase

### 1. Audit Current Modification Usage
- [ ] Find all places that handle non-None returns
- [ ] Document what each modification type does
- [ ] Identify which tests rely on modifications

### 2. Impact Analysis
- [ ] Which files need changes
- [ ] Which tests need updates
- [ ] Breaking changes assessment

## Implementation Phase

### 3. Remove Modification Logic
- [ ] Remove PolicyModification TypedDict
- [ ] Update PolicyFunction type to return None only
- [ ] Remove validation.py (no longer needed)
- [ ] Remove modification handling from:
  - gfql_unified.py (preload query/engine mods)
  - chain.py (postload engine conversion)
  - chain_let.py (postload handling)
  - call_executor.py (call param mods)

### 4. Simplify PolicyContext
- [ ] Remove fields only needed for modifications
- [ ] Keep only inspection fields

### 5. Update Exception Handling
- [ ] Ensure PolicyException is the only control mechanism
- [ ] Keep enrichment logic

## Testing Phase

### 6. Update Tests
- [ ] Remove modification tests from test_policy_behavior_modification.py
- [ ] Update test_policy_validation.py (remove modification validation)
- [ ] Fix test_policy_integration.py scenarios
- [ ] Ensure all tests still pass

### 7. Add New Tests
- [ ] Test Accept/Deny patterns
- [ ] Test state tracking without modifications
- [ ] Test clean error messages

## Documentation Phase

### 8. Update Documentation
- [ ] Update PR description
- [ ] Update example code
- [ ] Update docstrings

## Rollout Phase

### 9. Final Validation
- [ ] All tests pass
- [ ] CI green
- [ ] No breaking changes for basic usage

## Important Notes
- **HALT if Sonnet 4 downgrade occurs**
- Keep changes minimal and focused
- Preserve Accept/Deny functionality
- Maintain thread safety
- Keep closure-based state management

## Current TODO: Implementation Phase - Removing PolicyModification

### Status After Memory Compact:
- ✅ Investigation complete (see policy_modification_audit.md)
- 🚧 Currently editing types.py to remove PolicyModification
- Need to update ~20 tests after core changes

### Next Steps:
1. Change PolicyFunction to return None only (line 54 in types.py)
2. Remove PolicyModification TypedDict (lines 39-50 in types.py)
3. Delete validation.py entirely
4. Remove modification handling from 4 core files
5. Update 20 tests to use Accept/Deny pattern