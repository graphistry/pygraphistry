# Pyre/Pysa Analysis for PyGraphistry

**Recommendation**: Use AST (< 1s) for direct patterns, then Pysa (~60s) for call chains

## Quick Commands

**1. Generate call graph** (60s, finds indirect bugs via call chains):
```bash
docker run --rm -v $(pwd):/workspace -w /workspace python:3.12-slim \
  bash -c "pip install -q pyre-check && pyre analyze --save-results-to ./pysa_results"
```

**2. Verify call graph generated**:
```bash
ls -lh pysa_results/call-graph.json  # Should be ~16MB, 37626 functions
```

**3. Extract callers** (see `ai/assets/pysa_extract_callers.py`):
```bash
# Single method
python3 ai/assets/pysa_extract_callers.py pysa_results/call-graph.json \
  PlotterBase.PlotterBase.bind

# Multiple methods
python3 ai/assets/pysa_extract_callers.py pysa_results/call-graph.json \
  PlotterBase.PlotterBase.bind \
  PlotterBase.PlotterBase.nodes \
  PlotterBase.PlotterBase.edges
```

**Config**: `.pyre_configuration` (excludes hop.py to prevent hang)

**Optional - Taint models**: To use taint models, add to your local `.pyre_configuration`:
```json
"taint_models_path": ["path/to/your/taint_models"]
```
Note: Taint models are experimental and not committed to the repo.

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
