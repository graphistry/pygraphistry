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

## Project Dependencies

PyGraphistry has different dependency sets depending on functionality:

- Core: numpy, pandas, pyarrow, requests
- Optional integrations: networkx, igraph, neo4j, gremlin, etc.
- GPU acceleration: RAPIDS ecosystem (cudf, cugraph)
- AI extensions: umap-learn, dgl, torch, sentence-transformers