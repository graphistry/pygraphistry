# TODO: Strict Type Checking & Code Quality Improvements

This file tracks issues discovered during code review that should be addressed when enabling stricter linting/type checking.

## High Priority

### 1. ✅ FIXED: UnboundLocalError in plot() method (PlotterBase.py:1778)
**Status**: Fixed in commit 5e0fc482

**Issue**: `uploader` variable only assigned when `api_version==3` but referenced unconditionally.

**Root Cause**:
- Line 1745: `uploader` only defined in API v3 branch
- Line 1778: checks `if uploader is not None:` without initialization

**Fix Applied**: Initialize `uploader = None` before the conditional (line 1735)

**Test Gap**: No tests for `api_version=1` + `render='g'` combination

---

### 2. ✅ FIXED: Missing error handling for unsupported API versions
**Status**: Fixed (pending commit)

**Issue**: Code only handled api_version 1 and 3, would cause UnboundLocalError for other values

**Fix Applied**: Added else clause raising ValueError with clear error message (line 1758-1763)

---

## Medium Priority

### 3. Missing test coverage for API v1 code paths
**File**: `graphistry/tests/`

**Issue**: No tests found for `api_version=1` functionality

**Impact**: Critical bugs like the `uploader` UnboundLocalError go undetected

**Recommendation**:
- Add test suite for API v1 endpoints
- Test all render modes ('g', 'url', 'ipython', 'databricks', 'browser') with api_version=1
- Use mocking to avoid needing actual API v1 server

**Example test**:
```python
def test_plot_api_v1_render_g():
    """Test that api_version=1 with render='g' doesn't raise UnboundLocalError"""
    g = graphistry.edges(pd.DataFrame({'src': [1], 'dst': [2]}))
    g.session.api_version = 1
    result = g.plot(render='g', skip_upload=True)
    assert result._dataset_id is None  # Should not crash
```

---

### 4. Stricter MyPy configuration
**File**: `mypy.ini` or `setup.cfg`

**Current**: MyPy doesn't catch unbound local variable errors

**Recommendation**: Enable stricter checks:
```ini
[mypy]
warn_untyped_defs = True
warn_return_any = True
warn_unused_ignores = True
check_untyped_defs = True
no_implicit_optional = True
warn_redundant_casts = True
```

**Why it missed the bug**:
- Python's dynamic typing - no compile-time variable declarations
- MyPy needs strict settings to catch these
- May not analyze all code paths without explicit type annotations

---

### 5. Enable pylint undefined-variable check
**File**: `.pylintrc` or command line flags

**Issue**: pylint would catch `uploader` UnboundLocalError if configured

**Recommendation**:
```bash
pylint --enable=undefined-variable graphistry/PlotterBase.py
```

---

## Low Priority

### 6. Existing TODO comments to review
**File**: `graphistry/PlotterBase.py`

Found TODOs that may warrant attention:
- Line 864: `#TODO check set to api=3?`
- Line 891: `#TODO ensure that it is a color?`
- Line 2185: `#TODO: should we hash just in case...`
- Line 2247: `# TODO: per-gdf hashing?`
- Line 2263: `#TODO push the hash check to Spark`

**Recommendation**: Review each TODO and either:
- Implement the improvement
- Document why it's not needed
- Create GitHub issue for future work

---

### 7. ✅ FIXED: API version 2 support?
**File**: `graphistry/PlotterBase.py:1736-1741`
**Status**: Fixed (pending commit)

**Observation**: Code only handled `api_version == 1` or `== 3`, no handling for version 2

**Question**: Is API v2 intentionally skipped, or is this a gap?

**Fix Applied**: Added else clause that raises clear ValueError for unsupported API versions

---

## Tools to Consider

### Static Analysis
- **ruff** - Fast Python linter (already in use?)
- **bandit** - Security issue detection
- **vulture** - Find dead code
- **radon** - Cyclomatic complexity analysis

### Runtime Analysis
- **coverage.py** - Measure test coverage
- **pytest-cov** - Coverage integration with pytest

### Pre-commit Hooks
```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    hooks:
      - id: check-added-large-files
      - id: check-yaml
      - id: end-of-file-fixer

  - repo: https://github.com/astral-sh/ruff-pre-commit
    hooks:
      - id: ruff
        args: [--fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    hooks:
      - id: mypy
        args: [--strict]
```

---

## Action Items

**Immediate** (Easy wins):
- [x] Fix uploader UnboundLocalError (commit 5e0fc482)
- [x] Add else clause with error for unsupported API versions (pending commit)
- [ ] Add test for api_version=1 + render='g'

**Short term** (Next sprint):
- [ ] Enable stricter mypy settings incrementally
- [ ] Add pylint to CI/CD with undefined-variable check
- [ ] Review and address/document all TODO comments

**Long term** (Future improvement):
- [ ] Comprehensive test coverage for all API versions
- [ ] Pre-commit hooks for automatic linting
- [ ] Regular security audits with bandit

---

## Notes

- This file created as part of GFQL validation & hypergraph engine improvements
- See commits: 9addd59b, 83dc8b88, 05ec1aad, 64fd8b03, 5e0fc482
- Branch: `fix/gfql-remote-hypergraph`
