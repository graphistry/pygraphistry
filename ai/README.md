# ai

Specialized documentation for AI assistants working on PyGraphistry. These guides supplement the main CLAUDE.md with detailed, topic-specific information.

## 🎯 Quick Reference

### Critical Development Rules
- **Functional Programming**: Always return new objects, never modify in-place
- **No `copy()` on DataFrames**: Operations already return new objects  
- **Use `df.assign()`**: Never use `df[col] = val` syntax
- **Preserve Git History**: Avoid unnecessary rewrites
- **No Claude Comments**: Remove explanatory comments before committing

### Essential Commands
```bash
# Before any work - establish baseline (containerized)
cd docker && WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Quick Docker test (from docker/ directory)
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Run specific tests fast
WITH_LINT=0 WITH_TYPECHECK=0 WITH_BUILD=0 ./test-cpu-local.sh graphistry/tests/test_file.py

# GPU tests - FAST (reuse base image, no rebuild)
IMAGE="graphistry/graphistry-nvidia:${APP_BUILD_TAG:-latest}-${CUDA_SHORT_VERSION:-12.8}"
docker run --rm --gpus all -v "$(pwd)/graphistry:/opt/pygraphistry/graphistry:ro" \
    $IMAGE pytest /opt/pygraphistry/graphistry/tests/test_file.py -v

# GPU tests - SLOW (full rebuild, use before merge)
cd docker && ./test-gpu-local.sh

# Validate RST documentation syntax
./docs/validate-docs.sh                           # All docs
./docs/validate-docs.sh docs/source/gfql/*.rst   # Specific files
git diff --name-only HEAD -- '*.rst' | xargs ./docs/validate-docs.sh  # Changed files

# Note: Direct script execution requires local environment setup
# ./bin/lint.sh && ./bin/mypy.sh && ./bin/pytest.sh
```

### Performance Must-Haves
- Never call `str()` repeatedly on same value
- Use vectorized operations, not loops
- Select only needed columns: `df[['col1', 'col2']]`
- Use `logger.debug('msg %s', var)` not f-strings in loggers
- Respect engine abstractions (`df_concat`, `resolve_engine`)

## 📋 Documentation Hierarchy

```
CLAUDE.md                    # General guide (< 500 lines)
├── ai/                    # Specialized guides
│   ├── README.md           # This file - overview & quick ref
│   ├── docs/              # Documentation guides
│   │   ├── gfql/          # GFQL patterns & optimization
│   │   ├── gpu/           # GPU/RAPIDS best practices  
│   │   └── connectors/    # Database-specific patterns
│   └── prompts/           # Reusable workflow templates
└── plans/                 # Task tracking (gitignored)
```

### When to Use Each Level
- **CLAUDE.md**: Start here for general PyGraphistry development
- **ai/**: Load specific guides only when working on that topic
- **plans/**: Track multi-session work and complex implementations

## 🚀 Status Tracking Conventions

### Priority System (P0-P5)
- **P0 🚨**: Critical - Breaking functionality, must fix immediately
- **P1 🔴**: High - Type safety, imports, security issues  
- **P2 🟡**: Medium - Code style consistency, best practices
- **P3 🟢**: Low - Minor improvements, nice-to-haves
- **P4 ⚪**: Minimal - Cosmetic, already suppressed
- **P5 ⬜**: Skip - Won't fix, intentional patterns

### Progress Indicators
- ✅ Complete
- 🔄 In Progress  
- 📝 Planned
- ❌ Blocked
- ⏭️ Skipped

## 📁 Directory Structure

```
ai/
├── docs/                    # Documentation guides
│   ├── gfql/               # GFQL-specific patterns and guidelines
│   ├── gpu/                # GPU/CUDA development notes
│   └── connectors/         # Database connector patterns
└── prompts/                # Reusable workflow templates
    ├── PLAN.md                   # Task planning template with strict execution protocol
    ├── LINT_TYPES_CHECK.md       # Code quality enforcement (with P0-P5)
    ├── CONVENTIONAL_COMMITS.md   # Git commit workflow with PyGraphistry conventions
    ├── PYRE_ANALYSIS.md          # Advanced code analysis with pyre-check
    ├── IMPLEMENTATION_PLAN.md    # [TODO] Feature implementation tracking
    └── USER_TESTING_PLAYBOOK.md  # [TODO] AI-driven testing workflows
```

## 📖 Usage Guidelines

### Loading Documentation
1. **Start with CLAUDE.md** for general PyGraphistry work
2. **Load specific guides** only when working on that topic
3. **Don't load everything** - each guide is self-contained
4. **Check file size** - guides should be < 500 lines

### File Size Guidelines
- **Documentation**: Max 500 lines per file
- **Code files**: Max 300 lines ideal, 500 lines acceptable
- **Functions**: Max 50 lines per function
- **Classes**: Split large classes into mixins

## ✏️ Creating New Guides

When adding a new guide:
1. Place in appropriate subdirectory
2. Use descriptive names (e.g., `neo4j_patterns.md`)
3. Add header explaining when to use it
4. Focus on patterns, not API details
5. Include practical examples
6. Add to directory structure above

## 📚 Current Guides

### GFQL (Graph Frame Query Language)
- Query patterns and optimization
- Column naming conventions  
- Performance considerations
- Engine abstraction patterns
- **Load when**: Working on graph queries, chain/hop operations

### GPU/RAPIDS
- RAPIDS integration patterns
- Memory management strategies
- CPU/GPU fallback handling
- cuDF vs pandas compatibility
- **Load when**: Implementing GPU features, optimizing performance

### Connectors
- Database-specific patterns
- Connection management
- Error handling strategies
- Testing with databases
- **Load when**: Adding/fixing database integrations

### Prompt Templates
- **PLAN.md**: Task planning template with strict execution protocol for multi-step work
- **LINT_TYPES_CHECK.md**: Code quality enforcement with P0-P5 priorities
- **CONVENTIONAL_COMMITS.md**: Git commit workflow following PyGraphistry conventions
- **PYRE_ANALYSIS.md**: Advanced code analysis with pyre-check for refactoring and type-aware searching
- **IMPLEMENTATION_PLAN.md** [TODO]: Systematic feature implementation
- **USER_TESTING_PLAYBOOK.md** [TODO]: AI-driven testing workflows
- **Load when**: Starting new tasks, creating commits, fixing code quality issues, planning complex work, refactoring code

## 🧪 Testing Quick Reference

### Docker Commands (Recommended)
```bash
cd docker

# Fast iteration - skip slow parts
WITH_BUILD=0 ./test-cpu-local-minimal.sh

# Only lint and typecheck (no tests or build)
WITH_BUILD=0 WITH_TEST=0 ./test-cpu-local.sh

# Full validation before commit
./test-cpu-local.sh

# GPU functionality
./test-gpu-local.sh

# Specific features
./test-umap-learn-core.sh  # UMAP embeddings
./test-dgl.sh              # Graph neural networks
./test-embed.sh            # Embedding features
```

### GPU Testing - Fast (Reuse Base Image)

Docker containers include: **pytest, mypy, ruff** (preinstalled)

```bash
# Reuse existing graphistry image (no rebuild)
IMAGE="graphistry/graphistry-nvidia:${APP_BUILD_TAG:-latest}-${CUDA_SHORT_VERSION:-12.8}"

docker run --rm --gpus all \
    -v "$(pwd):/workspace:ro" \
    -w /workspace -e PYTHONPATH=/workspace \
    $IMAGE pytest graphistry/tests/test_file.py -v
```

**Fast iteration**: Use this during development
**Full rebuild**: Use `./docker/test-gpu-local.sh` before merge

### Environment Control
| Variable | Default | Purpose |
|----------|---------|---------|
| `WITH_LINT` | 1 | Run flake8 linting |
| `WITH_TYPECHECK` | 1 | Run mypy type checking |
| `WITH_BUILD` | 0 | Build documentation |
| `WITH_NEO4J` | 0 | Run Neo4j integration tests |
| `PYTHON_VERSION` | - | Override Python version |

## 🔍 Code Analysis & Search Tools

### Tool Selection Guide

**Use Grep/Ripgrep for:**
- Simple text/pattern search
- Quick file location
- Initial exploration
```bash
grep -r "dataset_id" graphistry/*.py
rg "_dataset_id" --type py
```

**Use AST Scripts for:**
- Custom pattern detection (e.g., "methods that modify attribute X")
- Fast iteration during development (< 1 second)
- When pyre is too slow or times out
```bash
python3 plans/task_name/analyze_*.py
```

**Use Pyre for:**
- Type-aware analysis and refactoring
- Find-all-references (call graph analysis)
- Finding all implementations of an interface
- Complex dependency chain analysis
- See [prompts/PYRE_ANALYSIS.md](prompts/PYRE_ANALYSIS.md) for detailed guide

**Recommendation**: Start with grep for exploration, use AST scripts for custom analysis, use pyre only when you need type-aware refactoring or call graphs.

## 🔧 Common Patterns

### DataFrame Operations
```python
# ✅ Good - Functional style
df = df.assign(new_col=values)
df = df[df['col'] > 0]  # Returns new DataFrame

# ❌ Bad - In-place modification  
df['new_col'] = values
df.drop('col', inplace=True)
```

### Engine Abstraction
```python
# ✅ Good - Engine agnostic
from graphistry.compute.typing import DataFrameLike
engine = resolve_engine(df)
result = engine.df_concat([df1, df2])

# ❌ Bad - Engine specific
if isinstance(df, pd.DataFrame):
    result = pd.concat([df1, df2])
```

### Type Annotations
```python
# ✅ Good - Clear types with imports
from typing import Optional, Union, TYPE_CHECKING
if TYPE_CHECKING:
    import cudf

def process(df: Union[pd.DataFrame, 'cudf.DataFrame']) -> Optional[pd.DataFrame]:
    return df if len(df) > 0 else None
```

## 🎯 Development Workflow

### Before Starting
1. Read CLAUDE.md for general context
2. Run baseline: `./bin/lint.sh && ./bin/mypy.sh`
3. Load specific guide if needed

### During Development  
1. Follow functional programming patterns
2. Add type annotations to new code
3. Use appropriate priority (P0-P5) for issues
4. Track complex work in plans/

### Before Committing
1. Run Docker tests: `cd docker && WITH_BUILD=0 ./test-cpu-local.sh`
2. Update CHANGELOG.md under `## [Development]` for user-visible changes:
   - **Added**: New features, predicates, call methods, API additions
   - **Fixed**: Bug fixes, breaking changes resolved
   - **Changed**: Behavior changes, deprecations
   - **Breaking 🔥**: API changes that require user code updates
   - **Docs**: Documentation improvements, examples, tutorials
   - **Infra**: CI/CD, testing infrastructure, build system changes
   - **Security**: Security fixes and improvements
   - **Perf**: Performance improvements with benchmarks
   - Include PR/issue numbers, examples, and impact descriptions
   - Omit: Internal refactorings, test updates, type-only changes
3. Use conventional commit: `fix(scope): description` (see `prompts/CONVENTIONAL_COMMITS.md`)
4. Remove debug code and Claude comments

## 📝 Task Planning & Tracking

For multi-session or complex work:
```
plans/task_name/
├── implementation_plan.md  # Phases and approach
├── progress.md            # Current status (update each session)
├── insights.md            # Learnings and recommendations
└── [task-specific files]  # Test results, benchmarks, etc.
```

### Example Task Names
- `add_gfql_caching`
- `fix_gpu_memory_leak`  
- `implement_new_layout`
- `optimize_umap_performance`