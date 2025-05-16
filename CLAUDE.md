# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

PyGraphistry is a Python library for graph visualization, analytics, and AI with GPU acceleration capabilities. It's designed to work with graph data by:

1. Loading and transforming data from various sources into graph structures
2. Providing visualization tools with GPU acceleration
3. Offering graph analytics and AI capabilities including querying, ML, and clustering

The library follows a client-server model where:
- The Python client prepares data and handles transformations like loading, wrangling, querying, ML, and AI
- Visualization happens through Graphistry servers (cloud or self-hosted)
- Most user interaction follows a functional programming style with immutable state

## Architecture

PyGraphistry has a modular architecture consisting of:

1. Core visualization engine that connects to Graphistry servers
2. GFQL (Graph Frame Query Language) for dataframe-native graph queries
3. Integration with many databases and graph systems (Neo4j, Neptune, TigerGraph, etc.)
4. GPU acceleration through RAPIDS integration
5. AI/ML capabilities including UMAP embeddings and graph neural networks

Most components follow functional-style programming where methods create new copies of objects with updated bindings rather than modifying state.

## Development Commands

### Containers

PyGraphistry uses Docker for development and testing. The `docker` directory contains Dockerfiles and scripts for building and running tests in isolated environments. The bin/*.sh are unaware of the Docker context, so you should run from the docker folder, which calls the appropriate scripts.

### Environment Setup

```bash
# Install PyGraphistry with development dependencies
pip install -e .[dev]

# For GPU-accelerated features
pip install -e .[rapids]

# For AI capabilities
pip install -e .[ai]

# For full development setup
pip install -e .[dev,test,ai]
```

### Testing Commands

Testing is via containerized pytest, with shell scripts for convenient entry points:

```bash
# Run all tests
./bin/test.sh

# Run tests in parallel when many (xdist)
./bin/test.sh -n auto

# Run minimal tests (no external dependencies)
./bin/test-minimal.sh

# Run specific test file or test
python -m pytest -vv graphistry/tests/test_file.py::TestClass::test_function

# Run with Neo4j connectivity tests
WITH_NEO4J=1 ./bin/test.sh

# Docker-based testing (recommended for full testing)
cd docker && ./test-cpu-local-minimal.sh
cd docker && ./test-cpu-local.sh
# For faster, targeted tests (WITH_BUILD=0 skips slow docs build)
WITH_LINT=0 WITH_TYPECHECK=0 WITH_BUILD=0 ./test-cpu-local.sh graphistry/tests/test_file.py::TestClass::test_function
# Ex: GFQL
WITH_BUILD=0 ./test-cpu-local-minimal.sh graphistry/tests/test_compute_chain.py graphistry/tests/compute
```

### Linting and Type Checking

Run before testing:

```bash
# Lint the code
./bin/lint.sh

# Type check with mypy
./bin/typecheck.sh
```

### Building Documentation

Sphinx-based:

```bash
# Build documentation locally
cd docs && ./build.sh
```

### GPU Testing

```bash
# For GPU functionality (if available)
cd docker && ./test-gpu-local.sh
```

## Common Development Workflows

### Adding a New Feature

1. Ensure you understand the functional programming style of PyGraphistry
2. Create new features as standalone modules or methods where possible
3. Implement it following the client-server model respecting immutable state
4. Add appropriate tests in the `graphistry/tests/` directory
5. Run linting and type checking before submitting changes

### Testing Changes

1. Use the appropriate test script for your feature:
   - `test-minimal.sh` for core functionality
   - `test-features.sh` for features functionality
   - `test-umap-learn-core.sh` for UMAP functionality
   - `test-dgl.sh` for graph neural network functionality
   - `test-embed.sh` for embedding functionality
   - Additional specialized tests exist for specific components

2. For database connectors, ensure you have the relevant database running:
   - `WITH_NEO4J=1 ./bin/test.sh` for Neo4j tests

### Building and Publishing

1. Update the changelog in CHANGELOG.md
2. Tag with semantic versioning: `git tag X.Y.Z && git push --tags`
3. Confirm GitHub Actions publishes to PyPI

### Dependencies

* Dependencies are managed in `setup.py`
* The `stubs` list in setup.py contains type stubs for development
* Avoid adding unnecessary dependencies
* If you encounter type checking errors related to missing imports:
  - First check if they're already defined in the `stubs` list in setup.py
  - If not, consider adding them to the ignore list in mypy.ini using format:
    ```
    [mypy-package_name.*]
    ignore_missing_imports = True
    ```

#### Dependency Structure

```python
# Core dependencies - always installed
core_requires = [
  'numpy', 'pandas', 'pyarrow', 'requests', ...
]

# Type stubs for development
stubs = [
  'pandas-stubs', 'types-requests', 'ipython', 'types-tqdm'
]

# Optional dependencies by category
base_extras_light = {...}  # Light integrations (networkx, igraph, etc)
base_extras_heavy = {...}  # Heavy integrations (GPU, AI, etc)
dev_extras = {...}         # Development tools (docs, testing, etc)
```

#### Docker Testing Dependencies

* Docker tests install dependencies via `-e .[test,build]` or `-e .[dev]`
* The PIP_DEPS environment variable controls which dependencies are installed
* If adding new stubs, add them to the `stubs` list in setup.py

## Project Dependencies

PyGraphistry has different dependency sets depending on functionality:

- Core: numpy, pandas, pyarrow, requests
- Optional integrations: networkx, igraph, neo4j, gremlin, etc.
- GPU acceleration: RAPIDS ecosystem (cudf, cugraph)
- AI extensions: umap-learn, dgl, torch, sentence-transformers

## Coding tips

* We're version controlled: Avoid unnecessary rewrites to preserve history
* Occasionally try lint & type checks when editing
* Post-process: remove Claude's explanatory comments

## Performance Guidelines

### Functional & Immutable
* Follow functional programming style - return new objects rather than modifying existing ones
* No explicit `copy()` calls on DataFrames - pandas/cudf operations already return new objects
* Chain operations to minimize intermediate objects

### DataFrame Efficiency
* Never call `str()` repeatedly on the same value - compute once and reuse
* Use `assign()` instead of direct column assignment: `df = df.assign(**{col: val})` not `df[col] = val`
* Select only needed columns: `df[['col1', 'col2']]` not `df` when processing large DataFrames
* Use `concat` and `drop_duplicates` with `subset` parameter when combining DataFrames
* Process collections at once (vectorized) rather than element by element

### GFQL & Engine
* Respect engine abstractions - use `df_concat`, `resolve_engine` etc. to support both pandas/cudf
* Collection-oriented algorithms: Process entire node/edge collections at once
* Be mindful of column name conflicts in graph operations
* Reuse computed temporary columns to avoid unnecessary conversions
* Consider memory implications during graph traversals

## Git tips

* Commits: We use conventional commits for commit messages, where each commit is a semantic change that can be understood in isolation, typically in the form of `type(scope): subject`. For example, `fix(graph): fix a bug in graph loading`. Try to isolate commits to one change at a time, and use the `--amend` flag to modify the last commit if you need to make changes before pushing. Changes should be atomic and self-contained, don't do too many things in one commit.

* CHANGELOG.md: We use a changelog to track changes in the project. We use semvars as git tags, so while deveoping, put in the top (reverse-chronological) section of the changelog `## [Development]`. Organize changes into subsections like `### Feat`,  `### Fixed`, `### Breaking 🔥`, etc.: reuse section names from the rest of the CHANGELOG.md. Be consistent in general.