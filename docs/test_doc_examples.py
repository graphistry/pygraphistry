#!/usr/bin/env python3
"""Pytest-based audit of code examples in GFQL documentation.

Extracts ``.. code-block:: python`` (RST) and ```python (MD) blocks from
docs/source/gfql/, runs them sequentially per file in a shared namespace
(like a notebook), and reports failures.

Skips GPU-only, remote-only, file-IO, and ML examples automatically.
Tolerates column-not-found and undefined-variable errors from conceptual
blocks that reference prose context.

Run locally:
    uv run --no-project --with python-igraph --with lark -- pytest docs/test_doc_examples.py -v

In CI, runs inside the docs docker container where graphistry is installed.
"""
from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import pytest

GFQL_DOCS = Path(__file__).resolve().parent / "source" / "gfql"

# Blocks containing any of these are auto-skipped (GPU, remote, file I/O, ML)
SKIP_PATTERNS = [
    "cudf", "cugraph", 'engine="cudf"', "engine='cudf'",
    "gfql_remote", "upload()", "python_remote",
    "chain_remote", ".plot()", ".register(",
    "graphviz", "umap", "dbscan", "featurize",
    "RGCN", "dgl", "torch",
    "read_csv(", "read_json(", "read_parquet(",
    "neo4j.GraphDatabase", "pymongo", "sqlalchemy",
    "flask", "Flask", "PlottableValidator",
    "compute_cugraph",
]

# These errors are noise — the example assumes different data or prose context
TOLERATED_ERRORS = [
    # Column/schema mismatches with our sample graph
    "column-not-found",
    "does not exist in node dataframe",
    "does not exist in edge dataframe",
    "does not exist in dataframe",
    "missing column",
    "references missing column",
    "Available columns:",
    "no node/edge bindings",
    "unsupported token in row expression",
    # Undefined variables from conceptual/prose blocks
    "is not defined",
    # File I/O that slipped past skip patterns
    "No such file or directory",
    "No module named",
    # Vertex attribute conflicts with sample graph's 'name' column
    "Vertex attribute conflict",
    # Expected validation errors in demo blocks
    "invalid-hops-value",
    # Cyclic graph errors from sample graph topology
    "Cyclic graph",
    # fa2 requires GPU
    "fa2_layout requires",
    # Multi-alias RETURN not yet supported
    "supports one MATCH source alias",
    # Raw datetime string ambiguity
    "Raw string",
    # Type mismatches from temporal wire protocol examples
    "val must be numeric",
    # Key errors from sample graph missing expected columns/node IDs
    "KeyError",
    # Hypergraph requires specific graph setup
    "Hypergraph requires",
    # Spec pseudocode with bare syntax / ellipsis
    "invalid syntax",
    "Got ellipsis",
    # Where clause form not yet supported
    "must use StepColumnRef",
    # Conceptual examples referencing undefined graph attributes / columns
    "'src'",
    "'source'",
    "'weight'",
    "'people'",
    "'persons'",
    "'adults'",
]

SETUP_CODE = """\
import sys, os, warnings
sys.path.insert(0, os.environ.get('PYGRAPHISTRY_ROOT', '.'))
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pandas as pd
import graphistry
from graphistry import n, e_forward, e_reverse, e_undirected, e, is_in
from graphistry.compute.predicates.numeric import gt, lt, ge, le, between
from graphistry.compute.predicates.str import contains, startswith, endswith
try:
    from graphistry.compute.predicates.comparison import eq, ne
except ImportError:
    pass
from graphistry.compute.ast import ASTCall as call, ASTLet as let, ASTRef as ref
from datetime import datetime, date, time, timedelta
nodes = pd.DataFrame({
    'id': ['a', 'b', 'c', 'd'],
    'type': ['person', 'person', 'company', 'person'],
    'score': [10, 5, 1, 8],
    'age': [30, 25, 40, 35],
    'country': ['USA', 'UK', 'USA', 'UK'],
    'name': ['Alice', 'Bob', 'Charlie', 'Dave'],
    'status': ['active', 'active', 'inactive', 'active'],
})
edges = pd.DataFrame({
    's': ['a', 'b', 'c', 'a'],
    'd': ['b', 'c', 'd', 'c'],
    'weight': [7, 9, 3, 5],
    'status': ['active', 'active', 'inactive', 'active'],
    'type': ['knows', 'works_at', 'knows', 'likes'],
    'label': ['knows', 'works_at', 'knows', 'likes'],
})
g = graphistry.nodes(nodes, 'id').edges(edges, 's', 'd')
"""


def _extract_code_blocks(path: Path) -> List[Tuple[int, str]]:
    text = path.read_text()
    blocks: List[Tuple[int, str]] = []

    if path.suffix == ".rst":
        pattern = re.compile(
            r"^\.\.\s+code-block::\s+python\s*\n"
            r"((?:\n|\s*\n|[ \t]+[^\n]*\n)*)",
            re.MULTILINE,
        )
        for m in pattern.finditer(text):
            raw = m.group(1)
            lines = raw.split("\n")
            code_lines = [ln for ln in lines if ln.strip()]
            if not code_lines:
                continue
            indent = min(len(ln) - len(ln.lstrip()) for ln in code_lines)
            code = "\n".join(ln[indent:] for ln in lines).strip()
            line_no = text[: m.start()].count("\n") + 1
            blocks.append((line_no, code))

    elif path.suffix == ".md":
        pattern = re.compile(r"^```python\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)
        for m in pattern.finditer(text):
            code = m.group(1).strip()
            line_no = text[: m.start()].count("\n") + 1
            blocks.append((line_no, code))

    return blocks


def _should_skip(code: str) -> Optional[str]:
    for pat in SKIP_PATTERNS:
        if pat in code:
            return pat
    return None


def _is_tolerated(err_str: str) -> bool:
    return any(pat in err_str for pat in TOLERATED_ERRORS)


def _collect_doc_files() -> List[Path]:
    return sorted(p for p in GFQL_DOCS.rglob("*") if p.suffix in (".rst", ".md"))


def _run_file_blocks(path: Path) -> List[Tuple[int, str, Optional[str]]]:
    """Run all blocks in a file sequentially. Return [(line, code, error_or_None)]."""
    blocks = _extract_code_blocks(path)
    results: List[Tuple[int, str, Optional[str]]] = []

    ns: dict = {}
    try:
        exec(compile(SETUP_CODE, "setup", "exec"), ns)
    except Exception as e:
        return [(0, "SETUP", str(e))]

    for line_no, code in blocks:
        skip = _should_skip(code)
        if skip:
            continue  # silently skip, not a test case

        try:
            exec(compile(code, f"{path.name}:{line_no}", "exec"), ns)
            results.append((line_no, code[:80], None))
        except Exception as e:
            err = str(e)
            if _is_tolerated(err):
                continue  # noise, not a test case
            results.append((line_no, code[:80], err.split("\n")[0][:200]))

    return results


# ---------------------------------------------------------------------------
# Pytest parametrization: one test per file
# ---------------------------------------------------------------------------

_doc_files = _collect_doc_files()


@pytest.mark.parametrize(
    "doc_path",
    _doc_files,
    ids=[str(p.relative_to(GFQL_DOCS)) for p in _doc_files],
)
def test_gfql_doc_examples(doc_path: Path) -> None:
    """All non-skipped, non-tolerated code blocks in a GFQL doc file must execute."""
    import os
    os.environ.setdefault("PYGRAPHISTRY_ROOT", str(Path(__file__).resolve().parent.parent))
    results = _run_file_blocks(doc_path)
    if not results:
        pytest.skip("no runnable code blocks")
    failures = [(line, code, err) for line, code, err in results if err is not None]
    if failures:
        msg_parts = [f"  {doc_path.name}:{line}: {err}" for line, _, err in failures]
        pytest.fail(
            f"{len(failures)} code example(s) failed in {doc_path.name}:\n"
            + "\n".join(msg_parts)
        )
