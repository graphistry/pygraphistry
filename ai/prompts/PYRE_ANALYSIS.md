# Pyre/Pysa Analysis for PyGraphistry

**Status**: ✅ Pysa found 2 bugs AST missed (2025-10-16, Pyre v0.9.25)

**Recommendation**: Use AST (< 1s) for direct patterns, then Pysa (~60s) for call chains

## Quick Commands

**Pysa call graph analysis** (finds indirect bugs via call chains):
```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.12-slim \
  bash -c "pip install -q pyre-check && pyre analyze --save-results-to ./pysa_results"
# Output: pysa_results/call-graph.json (16MB, 37626 functions)
```

**Extract callers** (see `ai/assets/pysa_extract_callers.py`):
```bash
python3 ai/assets/pysa_extract_callers.py pysa_results/call-graph.json \
  PlotterBase.PlotterBase.bind \
  PlotterBase.PlotterBase.nodes
```

**Config**: `.pyre_configuration` (excludes hop.py to prevent hang)

## Pysa Workflow

1. **Run Pysa** (60s, one-time per codebase state):
   - Generates call-graph.json covering entire codebase
   - Query it multiple times without re-running
   - Regenerate after significant code changes

2. **Extract callers** using `ai/assets/pysa_extract_callers.py`

3. **LLM filter** to real bugs:
   ```
   For each caller:
   1. Modifies dataset state? (Y/N)
   2. Invalidates _dataset_id after? (Y/N)
   3. If Y and N → BUG
   ```

**Results**: Found `style()` and `infer_labels()` - both use indirect `bind()` calls that AST missed

## Timeout Debugging

If Pysa hangs or times out:

```bash
# Run with debug mode to identify problematic method
docker run --rm -v $(pwd):/workspace -w /workspace python:3.12-slim \
  bash -c "pip install -q pyre-check && timeout 120 pyre --debug check 2>&1 | tail -200"
```

**Look for**: `"The type check of <module>.<function> is taking more than 60 seconds"`

**Solutions**:
- Add problematic file to `.pyre_configuration` exclude list
- Example: `hop.py` (384 lines, complex generics) causes hang at function 1631/1709

## Key Gotchas

- **Newline-delimited JSON**: Parse call-graph.json line-by-line, not as single JSON object
- **Module naming**: `PlotterBase.PlotterBase.method` not `graphistry.PlotterBase.method`
- **LLM filtering critical**: Raw output has false positives (test methods, sinks like `plot()`)
- **call-graph.json reusable**: One 60s run → unlimited queries until code changes
