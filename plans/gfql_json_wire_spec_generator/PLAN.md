# GFQL JSON Wire Spec Generator - Research & Documentation Plan

## Objective
Research and document how to generate cogent JSON AST wire specifications for the GFQL language, including examples for the most common flows. This is an exploratory research task where we'll **record EVERY command, step, and source** as we go, then distill them into a simplified recipe.

## Approach
1. **Research Phase**: Explore the codebase to understand existing wire spec generation
2. **Documentation Phase**: Record sources, methods, and examples as we discover them
3. **Synthesis Phase**: Create a simplified recipe from our findings

**CRITICAL**: Log EVERY command, file read, grep, glob, etc. to the Command Log section below!

## Research Questions
- How do GFQL AST nodes convert to JSON wire protocol?
- What are the most common GFQL usage patterns that need JSON representation?
- What helper methods/utilities exist for generating wire specs?
- What validation is performed on wire specs?

## Common GFQL Flows to Document
Based on initial review, these are the most important patterns:

1. **Chain-based graph traversal**
   - Node matcher → Edge matcher → Node matcher sequences
   - Example: Finding friends-of-friends

2. **Hypergraph transformations**
   - Event data → entity relationships
   - Multiple entity types and categorical columns

3. **UMAP layout**
   - Dimensionality reduction for graph visualization
   - Engine selection and parameters

4. **Let bindings**
   - Intermediate result storage
   - Multi-step queries

5. **Call operations**
   - Method invocations with parameters
   - Safelisted operations

6. **Predicates**
   - Filtering (gt, lt, is_in, contains, etc.)
   - Combining predicates with and_/or_

7. **Encodings**
   - Visual property mappings
   - Color, size, icon bindings

## Methodology
1. Start with simple examples and build complexity
2. **LOG EVERY COMMAND AND STEP** in the Command Log section
3. Record every source file, method, and insight in RESEARCH_NOTES.md
4. Test generation of JSON for each pattern
5. Document any gotchas or edge cases
6. Create a step-by-step recipe at the end

## Deliverables
- [ ] Documented sources for wire spec generation
- [ ] JSON examples for each common flow
- [ ] Step-by-step recipe for generating wire specs
- [ ] Test/validation approach
- [ ] Usage guide for future developers

## Recording Template
For each discovery, we'll record:
- **Source**: File path and line numbers
- **Method/Function**: What generates the JSON
- **Example Input**: Python GFQL code
- **Example Output**: JSON wire spec
- **Notes**: Any important details or gotchas

---

## Command Log

### Step 1: Setup Branch and Plan
**Command:**
```bash
git checkout -b feature/gfql-json-wire-spec-generator
```
**Result:** Created new branch
**Files Created:**
- `plans/gfql_json_wire_spec_generator/PLAN.md`

---

### Step 2: Discover Wire Spec Files
**Commands:**
```bash
# Search for files with "wire" in name
glob **/*wire*.py

# Find test files for GFQL
glob **/test*gfql*.py

# Find files with to_json methods
grep "def to_json" --path graphistry/compute --output_mode files_with_matches

# Find files with from_json methods
grep "from_json" --path graphistry/compute --output_mode files_with_matches
```

**Results:**
- No files with "wire" in name
- Found 7 GFQL test files
- Found 6 files with `to_json` methods:
  - `graphistry/compute/ast.py`
  - `graphistry/compute/chain.py`
  - `graphistry/compute/ast_temporal.py`
  - `graphistry/compute/ASTSerializable.py`
  - `graphistry/compute/predicates/comparison.py`
  - `graphistry/compute/predicates/is_in.py`
- Found 8 files with `from_json` methods

---

### Step 3: Read Core Serialization Infrastructure
**Commands:**
```bash
# Read base serialization class
read graphistry/compute/ASTSerializable.py

# Read predicate deserialization registry
read graphistry/compute/predicates/from_json.py

# Read basic test examples
read graphistry/tests/compute/test_gfql.py --limit 100
```

**Key Findings:**
- `ASTSerializable` provides automatic JSON serialization via `to_json()`
- Discriminated union pattern: `type` field indicates AST node class
- Predicate registry maps type names to classes
- See `RESEARCH_NOTES.md` for detailed findings

---

### Step 4: Examine AST Node Examples
**Commands:**
```bash
# Find all to_json usage in tests
grep "\.to_json\(" --path graphistry/tests/compute --output_mode content -n --head_limit 30

# Read basic AST serialization tests
read graphistry/tests/compute/test_ast.py

# Read call operation tests
read graphistry/tests/compute/test_call_operations.py --limit 150
```

**Key Findings:**
- Basic node/edge serialization examples
- Call operation JSON structure
- Parameter validation examples

---

### Step 5: Examine Chain Serialization
**Commands:**
```bash
# Read chain serialization tests
read graphistry/tests/compute/test_chain.py --offset 250 --limit 50
```

**Key Findings:**
- Chain wraps array of AST nodes/edges
- Predicates serialize within filter_dict
- Multi-hop and fixed-point examples

---

### Step 6: Create Example Generator Script
**Files Created:**
- `plans/gfql_json_wire_spec_generator/RESEARCH_NOTES.md` - Detailed research findings
- `plans/gfql_json_wire_spec_generator/generate_examples.py` - Script to generate JSON examples

**Purpose:** Programmatically generate JSON wire protocol examples for all common GFQL patterns

---

### Step 7: Run Example Generator (Iterative Fixes)
**Commands:**
```bash
# Attempt 1 - validation error on UMAP params
uv run python3.12 plans/gfql_json_wire_spec_generator/generate_examples.py > plans/gfql_json_wire_spec_generator/EXAMPLES_OUTPUT.md 2>&1
# Error: Unknown parameter 'feature_cols' for umap

# Fix 1: Change feature_cols -> n_components in generate_examples.py
# edit plans/gfql_json_wire_spec_generator/generate_examples.py

# Attempt 2 - TypeError with ASTRef
uv run python3.12 plans/gfql_json_wire_spec_generator/generate_examples.py > plans/gfql_json_wire_spec_generator/EXAMPLES_OUTPUT.md 2>&1
# Error: ASTRef.__init__() missing required argument 'chain'

# Fix 2: Correct ASTRef usage - needs both ref name and chain list
# edit plans/gfql_json_wire_spec_generator/generate_examples.py
# Changed: ASTRef('users') -> ASTRef('users', [operations...])

# Attempt 3 - SUCCESS!
uv run python3.12 plans/gfql_json_wire_spec_generator/generate_examples.py > plans/gfql_json_wire_spec_generator/EXAMPLES_OUTPUT.md 2>&1
```

**Result:** ✅ Successfully generated 13 comprehensive examples covering all major GFQL patterns

**Files Generated:**
- `plans/gfql_json_wire_spec_generator/EXAMPLES_OUTPUT.md` - Complete JSON wire protocol examples

---

## Progress Log

**2025-10-18 - Research Phase**
- ✅ Created branch and plan
- ✅ Discovered core serialization infrastructure
- ✅ Documented AST node serialization
- ✅ Documented predicate serialization
- ✅ Documented chain serialization
- ✅ Created example generator script
- 🔄 Generating comprehensive JSON examples (in progress)
- ⏳ Creating simplified recipe
- ⏳ Creating usage guide

**Next Steps:**
1. ✅ Complete JSON example generation
2. ✅ Document all examples in EXAMPLES_OUTPUT.md
3. ✅ Create simplified step-by-step recipe
4. ⏳ Commit all documentation
5. ⏳ Create PR with findings

---

### Step 8: Create Simplified Recipe
**File Created:**
- `plans/gfql_json_wire_spec_generator/RECIPE.md` - Step-by-step guide for generating JSON wire specs

**Contents:**
- Quick start guide
- Core concepts (ASTSerializable, discriminated unions, automatic field serialization)
- Step-by-step recipe
- Common patterns (chains, calls, DAGs, predicates, references)
- Validation and deserialization
- Key files reference
- Troubleshooting common issues

---

## Summary of Deliverables

✅ **PLAN.md** - Complete research plan with granular command logging
✅ **RESEARCH_NOTES.md** - Detailed findings from code exploration
✅ **generate_examples.py** - Script to generate JSON examples
✅ **EXAMPLES_OUTPUT.md** - 13 comprehensive JSON wire protocol examples
✅ **RECIPE.md** - Simple, actionable guide for generating JSON specs

---
Created: 2025-10-18
Completed: 2025-10-18
Status: ✅ Research Complete - Ready for PR
