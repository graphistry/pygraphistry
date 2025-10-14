# GFQL Predicates: Complete Implementation Checklist

When adding or modifying GFQL predicates, operators are **cross-cutting** - they touch multiple systems. This guide ensures all integration points are updated.

## 📋 9-Step Implementation Checklist (Minimum Requirements)

**Note**: Steps 0-8 are MINIMUM requirements. See "Extended Documentation" section for optional docs (add based on criteria).

| # | Step | File(s) | Required |
|---|------|---------|----------|
| 0 | **Module Exports** | `compute/__init__.py`, `__init__.py`, `ast.py`, `conf.py` | ✅ 🚨 CRITICAL - Users can't import without |
| 1 | **Implementation** | `predicates/*.py` | ✅ Class + function + `__call__` |
| 2 | **Validation** | Same file | ✅ Add `_validate_fields()` |
| 3 | **JSON Registry** | `from_json.py` | ✅ Import + registry entry |
| 4 | **Schema Validators** | `gfql/validate.py`, `validate/validate_schema.py` | ✅ Add to type tuples |
| 5 | **Language Spec** | `docs/.../language.md` | ✅ Grammar + operator ref |
| 6 | **Quick Reference** | `docs/.../quick.rst` | ✅ Operator table row |
| 7 | **Tests** | `tests/.../test_*.py` | ✅ Comprehensive tests |
| 8 | **Docstrings** | Implementation file | ✅ Full docstring + examples |

---

## 0️⃣ Module Exports 🚨 CRITICAL

**🚨 WITHOUT THIS, PREDICATE IS UNUSABLE!** Tests pass importing directly from module file, but users get ImportError.

**Add to 4 files** (pattern: `fullmatch, Fullmatch` after similar predicates):
1. `graphistry/compute/__init__.py` - import from `.predicates.str` + add to `__all__` list
2. `graphistry/__init__.py` - import from `graphistry.compute`
3. `graphistry/compute/ast.py` - import from `.predicates.str`
4. `docs/source/conf.py` - add `('py:class', 'graphistry.compute.predicates.str.Fullmatch')` to `nitpick_ignore`

**Verify**: `python -c "from graphistry import fullmatch; print('✅')"`

---

## 1️⃣ Implementation

**Files**: `predicates/str.py`, `numeric.py`, `temporal.py`, etc.

```python
from typing import Optional, Union
from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT

class Startswith(ASTPredicate):  # Inherit from ASTPredicate
    def __init__(self, pat: Union[str, tuple], case: bool = True, na: Optional[bool] = None):
        self.pat = tuple(pat) if isinstance(pat, list) else pat  # Convert list→tuple for JSON
        self.case = case
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        # Handle both pandas and cuDF (engine-specific workarounds as needed)
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__
        if is_cudf:
            # ... cuDF workaround logic
        else:
            return s.str.startswith(self.pat, self.na)  # pandas native

def startswith(pat: Union[str, tuple], case: bool = True, na: Optional[bool] = None) -> Startswith:
    """Factory function (lowercase) - see Docstring Template section for full example"""
    return Startswith(pat, case, na)
```

**Must include**: Class inheriting `ASTPredicate`, `__init__` storing params, `__call__(s: SeriesT) -> SeriesT`, factory function with docstring

---

## 2️⃣ Validation

**File**: Same as implementation file

```python
def _validate_fields(self) -> None:
    """Validate all parameters - called automatically by from_json/to_json"""
    from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

    # Validate each parameter type/value
    if not isinstance(self.pat, (str, tuple)):
        raise GFQLTypeError(ErrorCode.E201, "pat must be string or tuple", field="pat", value=type(self.pat).__name__)

    if isinstance(self.pat, tuple):
        for i, p in enumerate(self.pat):
            if not isinstance(p, str):
                raise GFQLTypeError(ErrorCode.E201, f"pat tuple element {i} must be string", field="pat", value=type(p).__name__)

    if not isinstance(self.case, bool):
        raise GFQLTypeError(ErrorCode.E201, "case must be boolean", field="case", value=type(self.case).__name__)

    if not isinstance(self.na, (bool, type(None))):
        raise GFQLTypeError(ErrorCode.E201, "na must be boolean or None", field="na", value=type(self.na).__name__)
```

**Pattern**: Check types/values, raise `GFQLTypeError(ErrorCode, message, field=, value=)` with clear error messages

---

## 3️⃣ JSON Registry

**File**: `graphistry/compute/predicates/from_json.py`

```python
# 1. Add import (class, not function)
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add here
    ...
)

# 2. Add to registry list
predicates : List[Type[ASTPredicate]] = [
    Duplicated, IsIn, GT, LT, GE, LE, EQ, NE, Between, IsNA, NotNA,
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add here
    ...
]
```

**Enables**: `from_json({'type': 'Fullmatch', ...})` deserialization. `type_to_predicate` dict auto-generated from class names.

**Non-JSON types**: If predicate uses tuples/custom types, add to `utils/json.py` serializer:
```python
elif isinstance(obj, tuple):
    return [serialize_to_json_val(item) for item in obj]  # Tuple → list for JSON
```

---

## 4️⃣ Schema Validators

**Purpose**: Static analysis catches type mismatches before execution (e.g., "Error: fullmatch() used on numeric column")

**File 1**: `gfql/validate.py`
```python
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add import
    ...
)

# In _validate_predicate_type():
STRING_PREDICATES = (
    Contains, Startswith, Endswith, Match, Fullmatch,  # ← Add to tuple
    IsNumeric, IsAlpha, ...
)
```

**File 2**: `validate/validate_schema.py`
```python
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch  # ← Add import
)

# In _validate_filter_dict():
if isinstance(val, (Contains, Startswith, Endswith, Match, Fullmatch)) and not pd.api.types.is_string_dtype(col_dtype):
    # ... string type error ...
```

**Pattern**: Import class, add to appropriate tuple (STRING_PREDICATES, TEMPORAL_PREDICATES, etc.) or isinstance check

---

## 5️⃣ Language Spec

**File**: `docs/source/gfql/spec/language.md`

```markdown
# Grammar section - add production rule
string_match ::= "contains(" string ... ")"
              | "fullmatch(" string ("," "case=" boolean)? ("," "flags=" integer)? ")"  # ← Add
              | ("startswith" | "endswith") "(" string ("," "case=" boolean)? ")"

# Operator reference section - add signature + brief description
fullmatch(pat, case=True, flags=0)  # Matches regex against entire string
```

## 6️⃣ Quick Reference

**File**: `docs/source/gfql/predicates/quick.rst`

```rst
# Add 3-column table row: Operator | Description | Example
   * - ``fullmatch(pattern, case=True)``
     - String matches regex ``pattern`` entirely. Case-insensitive if ``case=False``.
     - ``n({ "code": fullmatch(r"\d{3}", case=False) })``
```

---

## 7️⃣ Tests

**File**: `tests/compute/predicates/test_str.py` (or appropriate test file)

```python
class TestStartswith:
    """Test predicate with various parameter combinations"""

    def test_startswith_basic(self, engine):  # Happy path
        result = startswith('app')(make_series(['apple', 'banana', 'apricot'], engine))
        assert_series_equal(result, make_series([True, False, True], engine))

    def test_startswith_case_insensitive(self, engine):  # Parameter combos
        ...

    def test_startswith_tuple(self, engine):  # Tuple patterns
        ...

    def test_startswith_na_fill(self, engine):  # NA handling edge case
        ...

    def test_startswith_invalid_pat_type(self):  # Validation error
        with pytest.raises(GFQLTypeError) as exc:
            startswith(123)  # Invalid: numeric instead of string
        assert "pat must be string or tuple" in str(exc.value)

    def test_startswith_json_serialization(self):  # JSON round-trip
        pred = startswith(('test', 'demo'), case=False, na=True)
        restored = Startswith.from_json(pred.to_json())
        assert restored.pat == ('test', 'demo') and restored.case == False
```

**Coverage**: ✅ Basic ✅ All params ✅ Edge cases (empty, NA) ✅ Validation errors ✅ JSON ✅ Both engines (fixture) ✅ Parity

---

## 8️⃣ Documentation & Examples

**Docstring template**: Args (params + descriptions), Returns, Examples (3-5 progressive examples)

```python
def startswith(pat: Union[str, tuple], case: bool = True, na: Optional[bool] = None) -> Startswith:
    """Return whether pattern(s) match at start of string

    Args:
        pat: Pattern or tuple. Tuple uses OR logic (any match = True)
        case: Case-sensitive if True (default: True)
        na: Fill value for missing (default: None)

    Returns:
        Startswith predicate

    Examples:
        >>> n({"name": startswith("John")})                           # Basic
        >>> n({"name": startswith("john", case=False)})                # Case-insensitive
        >>> n({"filename": startswith(("test_", "demo_"))})            # Multiple patterns
    """
    return Startswith(pat, case, na)
```

---

## 🔍 Common Patterns

| Pattern | Workaround | Code Snippet |
|---------|-----------|--------------|
| **Case-insensitive** | Lowercase both | `s_work = s.str.lower() if not self.case else s; pat_work = pat.lower() if not self.case else pat` |
| **NA parameter** | cuDF lacks `na`, use fillna | `result = s.str.startswith(pat); return result.fillna(self.na) if is_cudf and self.na else result` |
| **Tuple patterns** | cuDF fails ([#20237](https://github.com/rapidsai/cudf/issues/20237)), manual OR | `results = [s.str.startswith(p) for p in pat]; result = results[0]; for r in results[1:]: result \|= r` |
| **JSON tuples** | JSON→list, convert back | `self.pat = tuple(pat) if isinstance(pat, list) else pat` (in `__init__`) |

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
| `docs/source/gfql/spec/wire_protocol.md` | JSON wire format (EXHAUSTIVE) | Complete JSON serialization with all parameters |

### Extended Documentation (If Applicable)

| File | Purpose | When to Update |
|------|---------|----------------|
| `docs/source/gfql/overview.rst` | Tutorial examples | If predicate useful for common patterns |
| `docs/source/gfql/quick.rst` | Quick start guide | If predicate commonly used |
| `docs/source/gfql/spec/cypher_mapping.md` | Cypher translation | If predicate has Cypher equivalent |
| `docs/source/gfql/wire_protocol_examples.md` | Wire examples | If showing JSON format helpful |
| `docs/source/gfql/datetime_filtering.md` | Temporal guide | If temporal predicate |
| `docs/source/gfql/translate.rst` | Translation guide | If predicate helps translations |
| `docs/source/gfql/about.rst` | About page | If predicate is a key feature |

---

## ✅ Verification

```bash
./bin/pytest.sh graphistry/tests/compute/predicates/test_str.py -v  # Tests pass
./bin/mypy.sh graphistry/compute/predicates/str.py                   # Type checking
python -c "from graphistry import fullmatch; print('✅ Import OK')"  # Exports work
python -c "from ...from_json import from_json; pred = fullmatch('x'); assert from_json(pred.to_json()).pat == 'x'"  # JSON
cd docker && WITH_BUILD=0 ./test-cpu-local.sh                        # Full CI
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

## 📚 Reference: IsIn - Complete Documentation Map

IsIn covers **all 9 required steps** plus **3 required docs** and **7 optional docs** (shown below for completeness):

### Required Steps (0-8)

| Step | File | What's Added |
|------|------|--------------|
| 0️⃣ | `compute/__init__.py`, `__init__.py`, `ast.py`, `conf.py` | Import + __all__ exports + nitpick_ignore |
| 1️⃣ | `predicates/is_in.py` | `class IsIn(ASTPredicate)` + `def is_in(options)` |
| 2️⃣ | Same | `_validate_fields()` checks `isinstance(options, list)` |
| 3️⃣ | `from_json.py` | Import + registry: `predicates = [Duplicated, IsIn, ...]` |
| 4️⃣ | N/A | IsIn is general-purpose, no STRING_PREDICATES entry needed |
| 5️⃣ | `spec/language.md` | `membership ::= "is_in(" "[" value ... "]" ")"` |
| 6️⃣ | `predicates/quick.rst` | Table row: `is_in(values)` \| Description \| Example |
| 7️⃣ | `test_is_in.py` | `class TestIsIn` with comprehensive tests |
| 8️⃣ | `is_in.py` | Factory function docstring |

### Required Docs (3 files)

| File | What's Added |
|------|--------------|
| `spec/language.md` | Grammar + operator signature |
| `predicates/quick.rst` | Operator table row |
| `spec/wire_protocol.md` | Complete JSON format with all parameters |

### Optional Docs (IsIn appears in 7 of these)

| File | Why IsIn Appears Here | Criteria Met |
|------|-----------------------|--------------|
| `overview.rst` | Tutorial example: "Filter by Multiple Node Types" | ✅ Common pattern for filtering |
| `quick.rst` | Quick start guide example | ✅ Frequently used predicate |
| `wire_protocol_examples.md` | JSON examples showing is_in serialization | ✅ Illustrates JSON format |
| `spec/cypher_mapping.md` | Maps to Cypher `IN` operator | ✅ Has Cypher equivalent |
| `translate.rst` | Translation examples using is_in | ✅ Helps with translations |
| `about.rst` | Mentioned in feature overview | ✅ Key/common feature |
| `datetime_filtering.md` | Examples with temporal values in is_in | ✅ Works with temporal predicates |

**Summary**: IsIn appears in **10 docs files** (3 required + 7 optional) because it's a foundational, frequently-used predicate. New predicates **must update the 3 required docs**, and may add optional docs based on criteria.

### Example: startswith/endswith/fullmatch (mid-level predicates)

**Updated 3 required docs only:**
- ✅ `spec/language.md` - Grammar rules
- ✅ `predicates/quick.rst` - Operator table rows
- ✅ `spec/wire_protocol.md` - JSON format with all parameters

**Did NOT update optional docs - why:**
- `overview.rst` / `quick.rst` - Not common enough for tutorials yet
- `wire_protocol_examples.md` - Spec sufficient, no special examples needed
- `spec/cypher_mapping.md` - Already had startswith/endswith (Cypher `STARTS WITH`/`ENDS WITH`)
- `translate.rst` / `about.rst` / `datetime_filtering.md` - Not applicable

**Takeaway**: Mid-level predicates → 3 required docs only. Foundational predicates like IsIn → 10 docs (3 required + 7 optional).

**Reference**: #774 (fullmatch + case/tuple support), #697 (case-insensitive predicates)

---

## 🔧 Type-Safe call() Operations (Step 9)

When adding new methods to `call()` safelist, update 3 files:

1. **`call_types.py`** - Add `TypedDict`, update `CallMethodName` Literal, add to `CallParams` Union
2. **`ast.py`** - Import TypedDict, add `@overload` signature
3. **`__init__.py`** - Export types if needed

**Example**: Adding `new_method`
```python
# call_types.py
class NewMethodParams(TypedDict, total=False):
    param1: str  # Required in safelist comment
    engine: Literal['pandas', 'cudf']

CallMethodName = Literal['hop', 'new_method', 'umap']  # Add alphabetically
CallParams = Union[HopParams, NewMethodParams, UmapParams]  # Add here

# ast.py - TYPE_CHECKING block
from graphistry.compute.call_types import NewMethodParams  # Import

@overload
def call(function: Literal['new_method'], params: 'NewMethodParams' = ...) -> ASTCall: ...
```

**Verify**: `./bin/mypy.sh graphistry/compute/ast.py graphistry/compute/call_types.py`
