# Policy Modification Audit

## Modification Types Currently Implemented

### 1. Engine Modifications
**Where Used:**
- `gfql_unified.py`: Line 211-213 - Preload phase can change engine
- `chain.py`: Line 530-542 - Postload phase converts DataFrames to new engine
- `call_executor.py`: Line 65-67 - Call phase can override engine

**Test Coverage:**
- `test_policy_behavior_modification.py`: 8 tests use engine modifications
- `test_policy_integration.py`: 4 tests use engine modifications
- `test_policy_recursion.py`: 1 test uses engine modification
- `test_policy_closure_state.py`: 1 test uses engine modification

### 2. Query Modifications
**Where Used:**
- `gfql_unified.py`: Line 216-217 - Preload phase can replace entire query

**Test Coverage:**
- `test_policy_recursion.py`: 3 tests modify query
- `test_policy_behavior_modification.py`: 1 test modifies query

### 3. Parameter Modifications
**Where Used:**
- `call_executor.py`: Line 72-73 - Call phase can modify method parameters

**Test Coverage:**
- `test_policy_behavior_modification.py`: 2 tests modify params

## Files That Need Changes

### Core Implementation Files
1. **graphistry/compute/gfql/policy/types.py**
   - Remove PolicyModification TypedDict
   - Change PolicyFunction to return None only

2. **graphistry/compute/gfql/policy/validation.py**
   - DELETE ENTIRELY (no modifications to validate)

3. **graphistry/compute/gfql_unified.py**
   - Remove lines 205-219 (modification handling)
   - Keep lines 221-227 (exception handling)

4. **graphistry/compute/chain.py**
   - Remove lines 524-542 (modification handling)
   - Keep lines 544-550 (exception handling)

5. **graphistry/compute/chain_let.py**
   - Remove lines 434-456 (modification handling)
   - Keep lines 458-463 (exception handling)

6. **graphistry/compute/gfql/call_executor.py**
   - Remove lines 58-73 (modification handling)
   - Keep lines 75-81 (exception handling)

### Test Files That Need Updates
1. **test_policy_behavior_modification.py**
   - 8 tests need rewriting or removal
   - Focus on testing accept/deny instead

2. **test_policy_validation.py**
   - Remove all modification validation tests
   - Keep exception validation tests

3. **test_policy_integration.py**
   - Update 4 tests that use engine modifications
   - Convert to accept/deny patterns

4. **test_policy_recursion.py**
   - Update 4 tests that modify query/engine
   - Test recursion with exceptions only

5. **test_policy_closure_state.py**
   - Update 1 test using engine modification
   - Use state tracking with exceptions

## Impact Analysis

### Breaking Changes
- Policies can no longer return dictionaries
- No more engine switching in policies
- No more query rewriting in policies
- No more parameter modification in policies

### What Stays
- PolicyException for denying operations
- PolicyContext for inspection
- Closure-based state management
- Thread-safe recursion prevention
- All three hook phases (preload, postload, call)

### Migration Path
Before:
```python
def policy(context):
    if context['phase'] == 'preload':
        return {'engine': 'pandas'}  # Force CPU
    return None
```

After:
```python
def policy(context):
    if context['phase'] == 'preload':
        if context.get('engine') == 'cudf':
            raise PolicyException('preload', 'GPU not allowed')
    return None
```

## Test Impact Summary
- **14 tests** use engine modifications
- **4 tests** use query modifications
- **2 tests** use parameter modifications
- **Total: ~20 tests need updates**

## Recommendation
1. Start by removing modification logic from core files
2. Update PolicyFunction type signature
3. Fix tests one by one to use accept/deny patterns
4. Ensure CI stays green throughout