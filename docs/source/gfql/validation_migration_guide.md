# GFQL Validation Migration Guide

This guide helps you migrate from the external validation system to the new built-in validation.

## What Changed

The GFQL validation system has been integrated directly into the AST classes, providing:
- Automatic validation during construction
- Structured error codes for programmatic handling
- Better performance with pre-execution validation
- More helpful error messages with suggestions

## Migration Overview

### Old System (External Validation)
```python
from graphistry.compute.gfql.validate import validate_syntax, validate_schema

# Manual validation
issues = validate_syntax(query)
if issues:
    for issue in issues:
        print(f"{issue.level}: {issue.message}")
```

### New System (Built-in Validation)
```python
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import GFQLValidationError

# Automatic validation
try:
    chain = Chain(query)  # Validates automatically
except GFQLValidationError as e:
    print(f"[{e.code}] {e.message}")
```

## Key Differences

### 1. Automatic vs Manual Validation

**Before:**
```python
# Create query
query = [{"type": "n"}, {"type": "e_forward", "hops": -1}]

# Manually validate
issues = validate_syntax(query)
if issues:
    # Handle errors
```

**After:**
```python
# Validation happens automatically
try:
    chain = Chain([n(), e_forward(hops=-1)])
except GFQLTypeError as e:
    print(f"Error: {e.message}")  # "hops must be a positive integer"
```

### 2. Error Structure

**Before:**
```python
class ValidationIssue:
    level: str  # 'error' or 'warning'
    message: str
    operation_index: Optional[int]
    field: Optional[str]
    suggestion: Optional[str]
```

**After:**
```python
class GFQLValidationError(Exception):
    code: str  # e.g., "E301"
    message: str
    context: dict  # Contains field, value, suggestion, etc.
```

### 3. Error Types

**Before:** Single `ValidationIssue` class with `level` field

**After:** Specific exception types:
- `GFQLSyntaxError` (E1xx): Structural issues
- `GFQLTypeError` (E2xx): Type mismatches  
- `GFQLSchemaError` (E3xx): Data-related issues

### 4. Schema Validation

**Before:**
```python
schema = extract_schema_from_dataframes(nodes_df, edges_df)
issues = validate_schema(query, schema)
```

**After:**
```python
# Runtime validation (automatic)
result = g.chain(query)  # Raises GFQLSchemaError if invalid

# Pre-execution validation (optional)
result = g.chain(query, validate_schema=True)
```

## Migration Steps

### Step 1: Update Imports

Replace old imports:
```python
# Remove these
from graphistry.compute.gfql.validate import (
    validate_syntax,
    validate_schema,
    validate_query,
    ValidationIssue
)

# Add these
from graphistry.compute.exceptions import (
    GFQLValidationError,
    GFQLSyntaxError,
    GFQLTypeError,
    GFQLSchemaError,
    ErrorCode
)
```

### Step 2: Remove Manual Validation Calls

Old pattern:
```python
def process_query(query):
    # Validate first
    issues = validate_syntax(query)
    if issues:
        return None, issues
    
    # Then execute
    chain = Chain(query)
    return chain, None
```

New pattern:
```python
def process_query(query):
    try:
        chain = Chain(query)  # Validation included
        return chain
    except GFQLValidationError as e:
        # Handle error
        raise
```

### Step 3: Update Error Handling

Old pattern:
```python
issues = validate_query(query, nodes_df, edges_df)
for issue in issues:
    if issue.level == 'error':
        logger.error(f"{issue.message}")
    else:
        logger.warning(f"{issue.message}")
```

New pattern:
```python
try:
    result = g.chain(query)
except GFQLSyntaxError as e:
    logger.error(f"Syntax error [{e.code}]: {e.message}")
except GFQLSchemaError as e:
    logger.error(f"Schema error [{e.code}]: {e.message}")
    if e.code == ErrorCode.E301:
        logger.info(f"Available columns: {e.context.get('suggestion')}")
```

### Step 4: Use Error Codes

Error codes enable programmatic handling:

```python
try:
    result = g.chain(query)
except GFQLSchemaError as e:
    if e.code == ErrorCode.E301:  # Column not found
        # Suggest available columns
        print(e.context.get('suggestion'))
    elif e.code == ErrorCode.E302:  # Type mismatch
        # Show type information
        print(f"Column type: {e.context.get('column_type')}")
```

### Step 5: Leverage Collect-All Mode

New feature for getting all errors at once:

```python
# Get all validation errors
chain = Chain(query)
errors = chain.validate(collect_all=True)

for error in errors:
    print(f"[{error.code}] {error.message}")
```

## Common Patterns

### Pattern 1: Query Builder with Validation

```python
class QueryBuilder:
    def __init__(self):
        self.operations = []
    
    def add_operation(self, op):
        # Test validates immediately
        test_chain = Chain(self.operations + [op])
        self.operations.append(op)
        return self
    
    def build(self):
        return Chain(self.operations)
```

### Pattern 2: Pre-execution Validation

```python
def safe_execute(g, operations):
    # Validate before expensive execution
    chain = Chain(operations)
    
    # Pre-validate schema
    if hasattr(chain, 'validate_schema'):
        errors = chain.validate_schema(g, collect_all=True)
        if errors:
            logger.warning(f"Found {len(errors)} schema issues")
    
    # Execute
    return g.chain(operations)
```

### Pattern 3: Error Recovery

```python
def execute_with_fallback(g, operations):
    try:
        return g.chain(operations)
    except GFQLSchemaError as e:
        if e.code == ErrorCode.E301:
            # Try without the problematic filter
            field = e.context.get('field')
            logger.warning(f"Removing filter on missing column: {field}")
            # ... modify operations ...
            return g.chain(modified_operations)
        raise
```

## Backward Compatibility Notes

1. **Empty chains remain valid** - No breaking change
2. **Old validation module still exists** - But deprecated
3. **Error handling is stricter** - Errors that were warnings may now raise

## Benefits of Migration

1. **Performance**: No separate validation pass needed
2. **Developer Experience**: Errors caught immediately during construction
3. **Better Messages**: Structured errors with suggestions
4. **Type Safety**: Specific exception types for different error categories
5. **Flexibility**: Choose between fail-fast and collect-all modes

## Need Help?

- Check the [updated validation notebook](../demos/gfql/gfql_validation_fundamentals_updated.ipynb)
- See [Python embedding docs](spec/python_embedding.md#validation) for API details
- Review [error code reference](../../compute/exceptions.py) for all codes