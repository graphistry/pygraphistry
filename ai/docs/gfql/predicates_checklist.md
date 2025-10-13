# GFQL Predicates: Complete Implementation Checklist

When adding or modifying GFQL predicates, operators are **cross-cutting** - they touch multiple systems. This guide ensures all integration points are updated.

## 📋 9-Step Implementation Checklist

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

**Purpose**: Make predicate accessible to users via `from graphistry import predicate`

**🚨 WITHOUT THIS STEP, THE PREDICATE IS UNUSABLE!** Tests pass because they import directly from the module file.

**File 1**: `graphistry/compute/__init__.py`
```python
# Import
from .predicates.str import (
    contains, Contains,
    startswith, Startswith,
    endswith, Endswith,
    match, Match,
    fullmatch, Fullmatch,  # ← Add here
    ...
)

# __all__ list
__all__ = [
    ...
    'contains', 'Contains', 'startswith', 'Startswith',
    'endswith', 'Endswith', 'match', 'Match',
    'fullmatch', 'Fullmatch',  # ← Add here
    ...
]
```

**File 2**: `graphistry/__init__.py`
```python
from graphistry.compute import (
    n, e, ...,
    contains, Contains,
    startswith, Startswith,
    endswith, Endswith,
    match, Match,
    fullmatch, Fullmatch,  # ← Add here
    ...
)
```

**File 3**: `graphistry/compute/ast.py`
```python
from .predicates.str import (
    contains, Contains,
    startswith, Startswith,
    endswith, Endswith,
    match, Match,
    fullmatch, Fullmatch,  # ← Add here
    ...
)
```

**File 4**: `docs/source/conf.py` (nitpick_ignore for Sphinx)
```python
nitpick_ignore = [
    ...
    ('py:class', 'graphistry.compute.predicates.str.Match'),
    ('py:class', 'graphistry.compute.predicates.str.Fullmatch'),  # ← Add here
    ...
]
```

**Test exports work**:
```bash
python -c "from graphistry import fullmatch; print('✅ OK')"
python -c "from graphistry.compute import fullmatch; print('✅ OK')"
```

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

| Pattern | Workaround | Code |
|---------|-----------|------|
| **Case-insensitive** | Lowercase both | `s_lower = s.str.lower(); result = s_lower.str.startswith(pat.lower())` |
| **NA parameter** | cuDF lacks `na`, use fillna | `result = s.str.startswith(pat); return result.fillna(na) if na else result` |
| **Tuple patterns** | cuDF fails ([#20237](https://github.com/rapidsai/cudf/issues/20237)), manual OR | `results = [s.str.startswith(p) for p in pat]; result = results[0]; for r in results[1:]: result \|= r` |
| **JSON tuples** | JSON→list, convert back | `self.pat = tuple(pat) if isinstance(pat, list) else pat` (in `__init__`) |

**Full case/NA/tuple example**:
```python
def __call__(self, s: SeriesT) -> SeriesT:
    is_cudf = 'cudf' in s.__module__
    pat = self.pat.lower() if not self.case else self.pat
    s_work = s.str.lower() if not self.case else s

    if isinstance(self.pat, tuple):
        results = [s_work.str.startswith(p) for p in pat]
        result = results[0]; [result := result | r for r in results[1:]]
    else:
        result = s_work.str.startswith(pat)

    return result.fillna(self.na) if is_cudf and self.na else result
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

## 📚 Reference: IsIn Cross-Check

IsIn appears in all 9 steps (verified for checklist completeness):

| Step | File | What's Added |
|------|------|--------------|
| 0️⃣ | `compute/__init__.py`, `__init__.py`, `ast.py` | Import + __all__ exports |
| 1️⃣ | `predicates/is_in.py` | `class IsIn(ASTPredicate)` + `def is_in(options)` |
| 2️⃣ | Same | `_validate_fields()` checks `isinstance(options, list)` |
| 3️⃣ | `from_json.py` | Import + registry: `predicates = [Duplicated, IsIn, ...]` |
| 4️⃣ | N/A | IsIn is general-purpose, no STRING_PREDICATES entry needed |
| 5️⃣ | `spec/language.md` | `membership ::= "is_in(" "[" value ... "]" ")"` |
| 6️⃣ | `predicates/quick.rst` | Table row: `is_in(values)` \| Description \| Example |
| 7️⃣ | `test_is_in.py` | `class TestIsIn` with comprehensive tests |
| 8️⃣ | `is_in.py` | Factory function docstring |

**Also appears in** (optional): `overview.rst` (examples), `wire_protocol.md` (JSON format), `cypher_mapping.md` (translation)

**Reference PRs**: #774 (fullmatch + case/tuple support), #697 (case-insensitive predicates)
