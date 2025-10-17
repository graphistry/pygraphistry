# Pyre Analysis for PyGraphistry

Pyre-check is a performant type checker for Python that can also be used for advanced code analysis and refactoring. This guide shows how to use pyre on PyGraphistry for tasks beyond what grep/LLM can handle.

## When to Use Pyre vs Grep vs AST Scripts

- **Grep/Ripgrep**: Simple text search, file location, pattern matching
- **AST Scripts**: Custom analysis (attribute tracking, method modification patterns)
- **Pyre**: Type-aware analysis, find-all-references, call graphs, dependency analysis

**Use Pyre When:**
- Finding all callers of a method (call graph analysis)
- Finding all implementations of an interface
- Type-aware refactoring (rename with type constraints)
- Analyzing complex dependency chains
- Finding unused code based on type information

**Use AST Scripts When:**
- Custom pattern detection (e.g., "methods that modify attribute X")
- Pyre is too slow or times out
- Need fast iteration during development

**Use Grep When:**
- Simple string/pattern search
- Quick file location
- Initial exploration

## Installation

```bash
# System-level install (recommended)
uv tool install --python python3.12 pyre-check

# Verify installation
which pyre  # Should show: /home/lmeyerov/.local/bin/pyre
pyre --version
```

## Configuration

PyGraphistry has a `.pyre_configuration` file:

```json
{
  "site_package_search_strategy": "pep561",
  "source_directories": ["graphistry"],
  "exclude": [".*/tests/.*", ".*/demos/.*", ".*/docs/.*"],
  "strict": false
}
```

## Running Pyre

### Option 1: Docker (Recommended - Avoids Library Issues)

```bash
# Using python:3.12-slim image (has required glibc/libstdc++ versions)
docker run --rm -it \
    -v /home/lmeyerov/Work/pygraphistry2:/workspace:ro \
    -w /workspace \
    python:3.12-slim \
    bash -c "pip install pyre-check && pyre check"
```

**Note**: First run takes 5-10 minutes to analyze ~1700 functions. Use incremental mode for faster subsequent runs.

### Option 2: Host System (May Have Library Issues)

```bash
# If you have compatible glibc (2.33+) and libstdc++ (GLIBCXX_3.4.29+)
pyre check

# Incremental mode (faster after initial run)
pyre incremental
```

**Known Issue**: On older systems, you may see:
```
pyre.bin: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.33' not found
```
→ Use Docker approach instead.

## Top 5 Pyre Commands for Analysis

### 1. Find All References to a Function/Class

```bash
# Find everywhere a method is called
pyre query "find_references('graphistry.PlotterBase.nodes')"

# Find everywhere a class is used
pyre query "find_references('graphistry.Plottable.Plottable')"
```

**Use Case**: Before refactoring a method, find all call sites to assess impact.

### 2. Get Type Information

```bash
# Get the type of an expression at a specific location
pyre query "types('graphistry/PlotterBase.py')"

# Get the type at a specific line and column
pyre query "type_at_position('graphistry/PlotterBase.py', 100, 10)"
```

**Use Case**: Understanding complex type flows, verifying inferred types.

### 3. Find All Attributes

```bash
# Find all attributes of a class
pyre query "attributes('graphistry.PlotterBase.PlotterBase')"

# Find where an attribute is defined
pyre query "defines('graphistry.PlotterBase.PlotterBase', '_dataset_id')"
```

**Use Case**: Attribute tracking, finding all dataset-relevant attributes (as in our ID invalidation bug).

### 4. Call Graph Analysis

```bash
# Find all callees of a function (what it calls)
pyre query "callees('graphistry.PlotterBase.nodes')"

# Find all callers of a function (what calls it)
pyre query "callers('graphistry.PlotterBase.nodes')"
```

**Use Case**: Understanding data flow, finding indirect modification paths.

### 5. Validate Types

```bash
# Check types without query mode
pyre check

# Check with detailed output
pyre check --output=json

# Check specific file
pyre check graphistry/PlotterBase.py
```

**Use Case**: Pre-commit validation, finding type errors.

## Practical Example: Finding Dataset ID Invalidation Issues

**Problem**: Find all methods that modify `_dataset_id` attribute.

### Attempt 1: Pyre Query (Ideal but slow)
```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.12-slim bash -c "
    pip install pyre-check && \
    pyre query \"find_references('graphistry.PlotterBase._dataset_id')\"
"
```

**Issue**: May timeout on large codebase (5-10 minutes).

### Attempt 2: AST Script (Practical alternative)
```python
# plans/fix_dataset_file_id_invalidation/analyze_attribute_modifications.py
import ast

class MethodAnalyzer(ast.NodeVisitor):
    def visit_Assign(self, node):
        # Track all assignments to _dataset_id
        if isinstance(target, ast.Attribute) and target.attr == '_dataset_id':
            # Record method name and line number
            ...
```

**Result**: Completes in < 1 second, found all 5 buggy methods.

**Lesson**: For custom analysis patterns, AST scripts are often more practical than pyre queries.

## Troubleshooting

### Pyre Times Out

**Symptom**: Hangs at "Processed 1631 of 1709 functions" for 5+ minutes.

**Solutions**:
1. Use incremental mode after first full run: `pyre incremental`
2. Analyze smaller subsets: `pyre check graphistry/PlotterBase.py`
3. Use AST scripts for custom analysis instead
4. Increase timeout: `pyre check --timeout 600`

### Library Incompatibility

**Symptom**: `GLIBC_2.33 not found` or `GLIBCXX_3.4.29 not found`

**Solutions**:
1. Use Docker with python:3.12-slim image (has newer libraries)
2. Upgrade system libraries (may not be feasible)
3. Use alternative analysis tools (AST scripts, mypy)

### Out of Memory

**Symptom**: Pyre crashes or system becomes unresponsive.

**Solutions**:
1. Close other applications
2. Analyze smaller file subsets
3. Use pyre in Docker with memory limits: `docker run --memory=4g ...`

## Integration with Development Workflow

### Pre-commit Type Checking

```bash
# Run mypy instead of pyre for faster type checking
./bin/mypy.sh

# Or in Docker
cd docker && WITH_BUILD=0 WITH_TEST=0 WITH_LINT=0 WITH_TYPECHECK=1 ./test-cpu-local.sh
```

**Note**: PyGraphistry uses mypy for regular type checking. Pyre is for advanced analysis only.

### Refactoring Workflow

1. **Identify**: Use pyre/grep to find all occurrences
2. **Analyze**: Use AST scripts for custom patterns
3. **Refactor**: Make changes with type awareness
4. **Validate**: Run mypy + tests to verify

## Pyre vs Mypy

| Feature | Pyre | Mypy |
|---------|------|------|
| Type checking | ✅ Yes | ✅ Yes |
| Speed | Slower (5-10min) | Faster (< 1min) |
| Query API | ✅ Yes | ❌ No |
| Call graphs | ✅ Yes | ❌ No |
| Find references | ✅ Yes | ❌ No |
| PyGraphistry default | ❌ No | ✅ Yes |

**Recommendation**: Use mypy for regular type checking, pyre for advanced refactoring analysis.

## Future Work

- **Performance**: Investigate pyre incremental mode for faster analysis
- **Integration**: Add pyre query scripts for common refactoring tasks
- **Documentation**: Document useful query patterns as they're discovered
- **CI/CD**: Consider pyre for static analysis in CI (if performance improves)

## Resources

- [Pyre Documentation](https://pyre-check.org/)
- [Pyre Query API](https://pyre-check.org/docs/querying-pyre/)
- PyGraphistry AST Scripts: `plans/*/analyze_*.py`
