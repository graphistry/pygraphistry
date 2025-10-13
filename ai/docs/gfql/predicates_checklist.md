# GFQL Predicates: Complete Implementation Checklist

When adding or modifying GFQL predicates, operators are **cross-cutting** - they touch multiple systems. This guide ensures all integration points are updated.

## 📋 Quick Checklist

When adding/modifying a predicate:

- [ ] **1. Implementation** - Define class and function in `predicates/*.py`
- [ ] **2. Validation** - Add `_validate_fields()` method
- [ ] **3. JSON Registry** - Register in `predicates/from_json.py`
- [ ] **4. Schema Validators** - Add to type checking in validation files
- [ ] **5. Language Spec** - Update GFQL grammar in `docs/source/gfql/spec/language.md`
- [ ] **6. Quick Reference** - Add to operator table in `docs/source/gfql/predicates/quick.rst`
- [ ] **7. Tests** - Add comprehensive tests (pandas + cuDF when applicable)
- [ ] **8. Docstrings** - Add function docstring with examples

---

## 1️⃣ Implementation

**Files**: `graphistry/compute/predicates/str.py`, `numeric.py`, `temporal.py`, etc.

### Class Definition

```python
from typing import Optional, Union
from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


class Startswith(ASTPredicate):
    def __init__(
        self,
        pat: Union[str, tuple],
        case: bool = True,
        na: Optional[bool] = None
    ) -> None:
        # Convert list to tuple for JSON deserialization compatibility
        self.pat = tuple(pat) if isinstance(pat, list) else pat
        self.case = case
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        # Implementation here - handle both pandas and cuDF
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__
        if is_cudf:
            # cuDF workaround logic
            pass
        else:
            # pandas native logic
            return s.str.startswith(self.pat, self.na)
```

**Key points**:
- Inherit from `ASTPredicate`
- Store all parameters as instance variables
- Handle list→tuple conversion for JSON deserialization
- Implement `__call__` for both pandas and cuDF
- Handle engine-specific workarounds (case-insensitive, na parameters)

### Factory Function

```python
def startswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Startswith:
    """
    Return whether a given pattern or tuple of patterns is at the
    start of a string

    Args:
        pat: Pattern (str) or tuple of patterns to match at start of
             string. When tuple, returns True if string starts with ANY
             pattern (OR logic)
        case: If True, case-sensitive matching (default: True)
        na: Fill value for missing values (default: None)

    Returns:
        Startswith predicate

    Examples:
        >>> # Single pattern, case-sensitive (default)
        >>> n({"name": startswith("John")})
        >>> # Single pattern, case-insensitive
        >>> n({"name": startswith("john", case=False)})
        >>> # Multiple patterns (OR logic)
        >>> n({"filename": startswith(("test_", "demo_"))})
    """
    return Startswith(pat, case, na)
```

**Key points**:
- Lowercase function name (matches Python conventions)
- Full docstring with Args, Returns, Examples
- All parameters have defaults where applicable

---

## 2️⃣ Validation

**File**: Same as implementation file

### Add `_validate_fields()` Method

```python
class Startswith(ASTPredicate):
    # ... __init__ and __call__ ...

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        # Validate pat type
        if not isinstance(self.pat, (str, tuple)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "pat must be string or tuple of strings",
                field="pat",
                value=type(self.pat).__name__
            )

        # If tuple, validate all elements are strings
        if isinstance(self.pat, tuple):
            for i, p in enumerate(self.pat):
                if not isinstance(p, str):
                    raise GFQLTypeError(
                        ErrorCode.E201,
                        f"pat tuple element {i} must be string",
                        field="pat",
                        value=type(p).__name__
                    )

        # Validate case parameter
        if not isinstance(self.case, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "case must be boolean",
                field="case",
                value=type(self.case).__name__
            )

        # Validate na parameter
        if not isinstance(self.na, (bool, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be boolean or None",
                field="na",
                value=type(self.na).__name__
            )
```

**Key points**:
- Validate all parameters (type, value constraints)
- Use `GFQLTypeError` with appropriate `ErrorCode`
- Provide clear error messages with field name and actual value
- This is called automatically during `from_json()` and `to_json()`

---

## 3️⃣ JSON Registry

**File**: `graphistry/compute/predicates/from_json.py`

### Add Import

```python
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add new predicate
    IsNumeric, IsAlpha, IsDecimal, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsTitle, IsNull, NotNull
)
```

### Add to Registry

```python
predicates : List[Type[ASTPredicate]] = [
    Duplicated,
    IsIn,
    GT, LT, GE, LE, EQ, NE, Between, IsNA, NotNA,
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add new predicate
    IsNumeric, IsAlpha, IsDecimal, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsDecimal, IsTitle, IsNull, NotNull,
    IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
    IsYearStart, IsYearEnd, IsLeapYear
]
```

**Key points**:
- Import the class (not the function)
- Add to `predicates` list
- The `type_to_predicate` dict is auto-generated from class names
- This enables `from_json({'type': 'Fullmatch', ...})`

### JSON Serialization Support

If your predicate uses tuples or other non-JSON types, ensure they're handled in `graphistry/utils/json.py`:

```python
def serialize_to_json_val(obj: Any) -> JSONVal:
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, tuple):
        # Convert tuples to lists for JSON serialization
        return [serialize_to_json_val(item) for item in obj]
    elif isinstance(obj, list):
        return [serialize_to_json_val(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: serialize_to_json_val(value) for key, value in obj.items()}
    else:
        raise TypeError(f"Unsupported type for to_json: {type(obj)}")
```

---

## 4️⃣ Schema Validators

Schema validators perform static analysis of GFQL chains to catch type mismatches before execution.

### File 1: `graphistry/compute/gfql/validate.py`

**Add Import**:
```python
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add new predicate
    IsNumeric, IsAlpha, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsDecimal, IsTitle
)
```

**Add to STRING_PREDICATES**:
```python
def _validate_predicate_type(predicate: ASTPredicate, column: str,
                             column_type: str, op_index: int,
                             field_prefix: str = "") -> List[ValidationIssue]:
    """Validate predicate is appropriate for column type."""
    # ...

    # Define string predicate types
    STRING_PREDICATES = (
        Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add new predicate
        IsNumeric, IsAlpha, IsDigit, IsLower, IsUpper,
        IsSpace, IsAlnum, IsDecimal, IsTitle
    )
```

### File 2: `graphistry/compute/validate/validate_schema.py`

**Add Import**:
```python
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch  # ← Add new predicate
)
```

**Add to isinstance Check**:
```python
# Check predicate type matches column type
if isinstance(val, (NumericASTPredicate, Between)) and not pd.api.types.is_numeric_dtype(col_dtype):
    # ... numeric type error ...

if isinstance(val, (Contains, Startswith, Endswith, Match, Fullmatch)) and not pd.api.types.is_string_dtype(col_dtype):
    # ... string type error ...
```

**Key points**:
- Add to appropriate predicate tuple (STRING_PREDICATES, TEMPORAL_PREDICATES, etc.)
- This enables type checking like: "Error: fullmatch() used on numeric column"

---

## 5️⃣ Language Spec

**File**: `docs/source/gfql/spec/language.md`

Update the grammar and examples:

### Grammar Section

```markdown
string_pred ::= string_match | string_check
string_match ::= "contains(" string ("," "case=" boolean)? ("," "regex=" boolean)? ")"
              | "match(" string ("," "case=" boolean)? ("," "flags=" integer)? ")"
              | "fullmatch(" string ("," "case=" boolean)? ("," "flags=" integer)? ")"
              | ("startswith" | "endswith") "(" string ("," "case=" boolean)? ")"
```

### Operator Reference Section

```markdown
Pattern matching predicates:
```python
contains(pat, case=True, regex=True)     # Contains pattern (substring or regex)
startswith(prefix, case=True)            # Starts with prefix
endswith(suffix, case=True)              # Ends with suffix
match(pat, case=True, flags=0)           # Matches regex from start of string
fullmatch(pat, case=True, flags=0)       # Matches regex against entire string
```
```

**Key points**:
- Update grammar with all parameters
- Add operator to appropriate section (string/numeric/temporal)
- Include parameter defaults

---

## 6️⃣ Quick Reference

**File**: `docs/source/gfql/predicates/quick.rst`

Add row to operator table:

```rst
   * - ``fullmatch(pattern, case=True)``
     - String matches regex ``pattern`` entirely. Case-insensitive if ``case=False``.
     - ``n({ "code": fullmatch(r"\d{3}", case=False) })``
```

**Key points**:
- Show signature with key parameters
- Brief description (1 sentence)
- Concise example showing typical usage
- Use RST table format (3 columns: Operator | Description | Example)

---

## 7️⃣ Tests

**File**: `graphistry/tests/compute/predicates/test_str.py` (or appropriate test file)

### Test Structure

```python
class TestStartswith:
    """Test startswith predicate with various parameter combinations."""

    # Test 1: Basic functionality
    def test_startswith_basic(self, engine):
        s = make_series(['apple', 'banana', 'apricot'], engine)
        result = startswith('app')(s)
        expected = make_series([True, False, True], engine)
        assert_series_equal(result, expected)

    # Test 2: Case-insensitive
    def test_startswith_case_insensitive(self, engine):
        s = make_series(['Apple', 'BANANA', 'apricot'], engine)
        result = startswith('app', case=False)(s)
        expected = make_series([True, False, True], engine)
        assert_series_equal(result, expected)

    # Test 3: Tuple patterns
    def test_startswith_tuple(self, engine):
        s = make_series(['apple', 'banana', 'cherry'], engine)
        result = startswith(('app', 'ban'), case=True)(s)
        expected = make_series([True, True, False], engine)
        assert_series_equal(result, expected)

    # Test 4: NA handling
    def test_startswith_na_fill(self, engine):
        s = make_series(['apple', None, 'apricot'], engine)
        result = startswith('app', na=False)(s)
        expected = make_series([True, False, True], engine)
        assert_series_equal(result, expected)

    # Test 5: Empty tuple edge case
    def test_startswith_empty_tuple(self, engine):
        s = make_series(['apple', 'banana'], engine)
        result = startswith((), case=True)(s)
        expected = make_series([False, False], engine)
        assert_series_equal(result, expected)

    # Test 6: Validation errors
    def test_startswith_invalid_pat_type(self):
        with pytest.raises(GFQLTypeError) as exc:
            startswith(123)  # Invalid: numeric instead of string
        assert exc.value.code == ErrorCode.E201
        assert "pat must be string or tuple" in str(exc.value)

    # Test 7: JSON round-trip
    def test_startswith_json_serialization(self):
        pred = startswith(('test', 'demo'), case=False, na=True)
        json_data = pred.to_json()
        restored = Startswith.from_json(json_data)
        assert restored.pat == ('test', 'demo')
        assert restored.case == False
        assert restored.na == True
```

**Test Coverage Requirements**:
1. ✅ Basic functionality (happy path)
2. ✅ All parameter combinations
3. ✅ Edge cases (empty inputs, None/NA values, empty tuples)
4. ✅ Validation errors (invalid parameter types/values)
5. ✅ JSON serialization/deserialization
6. ✅ Both pandas and cuDF (using `engine` fixture)
7. ✅ Parity tests (pandas vs cuDF produce same results)

---

## 8️⃣ Documentation & Examples

### Docstring Template

```python
def startswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Startswith:
    """
    Return whether a given pattern or tuple of patterns is at the
    start of a string

    Args:
        pat: Pattern (str) or tuple of patterns to match at start of
             string. When tuple, returns True if string starts with ANY
             pattern (OR logic)
        case: If True, case-sensitive matching (default: True)
        na: Fill value for missing values (default: None)

    Returns:
        Startswith predicate

    Examples:
        >>> # Single pattern, case-sensitive (default)
        >>> n({"name": startswith("John")})
        >>> # Single pattern, case-insensitive
        >>> n({"name": startswith("john", case=False)})
        >>> # Multiple patterns (OR logic)
        >>> n({"filename": startswith(("test_", "demo_"))})
        >>> # Multiple patterns, case-insensitive
        >>> n({"filename": startswith(("TEST", "DEMO"), case=False)})
    """
    return Startswith(pat, case, na)
```

**Key points**:
- Args section lists all parameters with types and descriptions
- Returns section describes return type
- Examples section shows 3-5 real-world usage patterns
- Examples progress from simple to complex
- Use `>>>` prefix for code examples (doctest style)

---

## 🔍 Common Patterns

### Pattern 1: Case-Insensitive Workaround

Both pandas and cuDF often don't support `case` parameter natively:

```python
def __call__(self, s: SeriesT) -> SeriesT:
    if not self.case:
        # Workaround: lowercase both string and pattern
        s_modified = s.str.lower()
        pat_modified = self.pat.lower()
        result = s_modified.str.startswith(pat_modified)
    else:
        result = s.str.startswith(self.pat)
```

### Pattern 2: NA Parameter Workaround

cuDF doesn't support `na` parameter - use `fillna()`:

```python
def __call__(self, s: SeriesT) -> SeriesT:
    is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

    if is_cudf:
        result = s.str.startswith(self.pat)
        if self.na is not None:
            return result.fillna(self.na)
        return result
    else:
        # pandas supports na parameter
        return s.str.startswith(self.pat, self.na)
```

### Pattern 3: Tuple Pattern Workaround

cuDF docs claim tuple support but implementation fails (see [cuDF#20237](https://github.com/rapidsai/cudf/issues/20237)):

```python
if isinstance(self.pat, tuple):
    if not is_cudf and self.case:
        # pandas native tuple support
        return s.str.startswith(self.pat)
    else:
        # cuDF workaround: manual OR logic
        results = [s.str.startswith(p) for p in self.pat]
        result = results[0]
        for r in results[1:]:
            result = result | r
        return result
```

### Pattern 4: List→Tuple Conversion for JSON

JSON doesn't have tuples, so arrays deserialize as lists. Convert in `__init__`:

```python
def __init__(self, pat: Union[str, tuple], ...):
    # Convert list to tuple for JSON deserialization compatibility
    self.pat = tuple(pat) if isinstance(pat, list) else pat
```

---

## 📦 Summary: Files to Update

### Core Implementation (Required)

| File | Purpose | What to Add |
|------|---------|-------------|
| `predicates/str.py` | Implementation | Class + function + validation |
| `predicates/from_json.py` | Deserialization | Import + registry entry |
| `gfql/validate.py` | Schema validation | Import + predicate tuple |
| `validate/validate_schema.py` | Type checking | Import + isinstance check |
| `tests/.../test_str.py` | Tests | Comprehensive test class |
| `utils/json.py` | JSON support | Handle special types (tuples, etc.) |

### Documentation (Required)

| File | Purpose | What to Add |
|------|---------|-------------|
| `docs/source/gfql/spec/language.md` | Grammar spec | Grammar + operator reference |
| `docs/source/gfql/predicates/quick.rst` | Quick reference | Operator table row |

### Extended Documentation (If Applicable)

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/source/gfql/overview.rst` | Tutorial examples | If predicate useful for common patterns |
| `docs/source/gfql/quick.rst` | Quick start guide | If predicate commonly used |
| `docs/source/gfql/spec/cypher_mapping.md` | Cypher translation | If predicate has Cypher equivalent |
| `docs/source/gfql/spec/wire_protocol.md` | JSON wire format | If predicate has complex serialization |
| `docs/source/gfql/wire_protocol_examples.md` | Wire examples | If showing JSON format helpful |
| `docs/source/gfql/datetime_filtering.md` | Temporal guide | If temporal predicate |
| `docs/source/gfql/translate.rst` | Translation guide | If predicate helps translations |
| `docs/source/gfql/about.rst` | About page | If predicate is a key feature |

---

## ✅ Verification Checklist

After implementing a new predicate, verify:

```bash
# 1. Tests pass (pandas + cuDF)
./bin/pytest.sh graphistry/tests/compute/predicates/test_str.py -v

# 2. Type checking passes
./bin/mypy.sh graphistry/compute/predicates/str.py

# 3. JSON serialization works
python -c "
from graphistry.compute.predicates.str import startswith
from graphistry.compute.predicates.from_json import from_json

pred = startswith(('a', 'b'), case=False, na=True)
json = pred.to_json()
restored = from_json(json)
assert restored.pat == ('a', 'b')
assert restored.case == False
print('✅ JSON serialization OK')
"

# 4. Documentation builds
./docs/validate-docs.sh docs/source/gfql/*.rst

# 5. Full CI validation
cd docker && WITH_BUILD=0 ./test-cpu-local.sh
```

---

## 🚨 Common Mistakes

1. **Forgetting JSON registry** - Predicate serializes but can't deserialize
2. **Missing schema validators** - Type mismatches not caught until runtime
3. **Skipping list→tuple conversion** - JSON round-trip fails
4. **No cuDF workarounds** - Works in pandas but breaks in GPU mode
5. **Missing validation** - Invalid parameters accepted, fail later
6. **Incomplete tests** - Edge cases not covered, bugs slip through
7. **Outdated docs** - Users can't discover new features

---

## 📚 Reference Examples

### Complete Real-World Example: IsIn Predicate

To verify our checklist is complete, here's every file `IsIn` appears in:

**1. Implementation** (`graphistry/compute/predicates/is_in.py`):
```python
class IsIn(ASTPredicate):
    def __init__(self, options: list) -> None:
        self.options = options

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isin(self.options)

    def _validate_fields(self) -> None:
        if not isinstance(self.options, list):
            raise GFQLTypeError(...)

def is_in(options: list) -> IsIn:
    return IsIn(options)
```

**2. JSON Registry** (`graphistry/compute/predicates/from_json.py`):
```python
from graphistry.compute.predicates.is_in import IsIn

predicates : List[Type[ASTPredicate]] = [
    Duplicated,
    IsIn,  # ← Registered here
    GT, LT, ...
]
```

**3. Documentation - Language Spec** (`docs/source/gfql/spec/language.md`):
```markdown
membership ::= "is_in(" "[" value ("," value)* "]" ")"

is_in([value1, value2, ...])  # Value in list
```

**4. Documentation - Quick Reference** (`docs/source/gfql/predicates/quick.rst`):
```rst
   * - ``is_in(values)``
     - Value in list ``values``.
     - ``n({ "type": is_in(["person", "company"]) })``
```

**5. Documentation - Overview** (`docs/source/gfql/overview.rst`):
```python
from graphistry import n, e_forward, is_in

g.chain([
    n({"type": is_in(["person", "company"])}),
    ...
])
```

**6. Documentation - Wire Protocol** (`docs/source/gfql/spec/wire_protocol.md`):
```json
{
    "type": "IsIn",
    "options": ["value1", "value2"]
}
```

**7. Documentation - Cypher Mapping** (`docs/source/gfql/spec/cypher_mapping.md`):
```markdown
| `n.id IN [1,2,3]` | `is_in([1,2,3])` | `{"type": "IsIn", "options": [1,2,3]}` |
```

**8. Tests** (`graphistry/tests/compute/predicates/test_is_in.py`):
```python
class TestIsIn:
    def test_is_in_basic(self, engine):
        s = make_series([1, 2, 3], engine)
        result = is_in([1, 3])(s)
        expected = make_series([True, False, True], engine)
        assert_series_equal(result, expected)
```

**Files NOT requiring updates for IsIn** (but might for other predicates):
- ❌ `gfql/validate.py` - IsIn is not in STRING_PREDICATES (it's general-purpose)
- ❌ `validate/validate_schema.py` - No special type checking needed
- ❌ `utils/json.py` - Lists already supported

### Recent PR Examples

See recent PRs for complete examples:
- **PR #774**: Added `fullmatch()`, updated `startswith()`/`endswith()` with `case` and tuple support
- **PR #697**: Initial case-insensitive support for string predicates

Key commits:
- Implementation: `graphistry/compute/predicates/str.py`
- JSON/validation: `1458b8b8` - "fix(gfql): add Fullmatch to schema validators and JSON registry"
- Tests: `graphistry/tests/compute/predicates/test_str.py`
