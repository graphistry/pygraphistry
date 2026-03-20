#!/usr/bin/env python3
"""Pytest-based audit of code examples in GFQL documentation.

Extracts ``.. code-block:: python`` (RST) and ```python (MD) blocks from
docs/source/gfql/, runs them sequentially per file in a shared namespace
(like a notebook), and reports failures.

Blocks can be marked in the doc source to control test behavior:

RST (invisible comment before the code-block):

    .. doc-test: skip
    .. doc-test: xfail

    .. code-block:: python

        code_here()

Markdown (HTML comment before the fenced block):

    <!-- doc-test: skip -->
    ```python
    code_here()
    ```

Unmarked blocks are auto-skipped if they contain GPU/remote/file-IO
patterns (see SKIP_PATTERNS). All other unmarked blocks must pass.

Run locally:
    uv run --no-project --with python-igraph --with lark -- pytest docs/test_doc_examples.py -v
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import pytest

GFQL_DOCS = Path(__file__).resolve().parent / "source" / "gfql"

# ---------------------------------------------------------------------------
# Auto-skip: blocks containing these are infrastructure-untestable
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Setup namespace
# ---------------------------------------------------------------------------
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
from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import (
    GFQLValidationError, GFQLSyntaxError,
    GFQLTypeError, GFQLSchemaError,
)
from graphistry import col, compare
from graphistry.compute import (
    rows, where_rows, return_, order_by, limit,
    skip, distinct, with_, select, group_by, unwind,
    remote,
)
from graphistry.compute.validate_schema import validate_chain_schema
from datetime import datetime, date, time, timedelta
nodes = pd.DataFrame({
    'id': ['a', 'b', 'c', 'd'],
    'type': ['person', 'person', 'company', 'person'],
    'score': [10, 5, 1, 8],
    'age': [30, 25, 40, 35],
    'country': ['USA', 'UK', 'USA', 'UK'],
    'name': ['Alice', 'Bob', 'Charlie', 'Dave'],
    'status': ['active', 'active', 'inactive', 'active'],
    'active': [True, True, False, True],
    'degree': [2, 3, 1, 0],
    'owner_id': ['x1', 'x1', 'x2', 'x2'],
    'org_id': ['org1', 'org1', 'org2', 'org2'],
    'department': ['sales', 'eng', 'sales', 'eng'],
    'importance': [0.9, 0.5, 0.1, 0.7],
    'risk_score': [3, 7, 1, 9],
    'created_at': pd.to_datetime(['2023-06-15', '2023-03-01', '2022-11-20', '2024-01-10']),
    'timestamp': pd.to_datetime(['2023-06-15 10:30', '2023-03-01 14:00', '2022-11-20 09:15', '2024-01-10 16:45']),
    'event_date': [date(2023, 6, 15), date(2023, 3, 1), date(2022, 11, 20), date(2024, 1, 10)],
    'date': [date(2023, 6, 15), date(2023, 3, 1), date(2022, 11, 20), date(2024, 1, 10)],
    'event_time': [time(10, 30), time(14, 0), time(9, 15), time(16, 45)],
    'daily_event': [time(10, 30), time(14, 0), time(9, 15), time(16, 45)],
    'start_date': [date(2023, 1, 1), date(2023, 2, 1), date(2023, 3, 1), date(2023, 4, 1)],
    'priority': [1, 3, 2, 1],
    'category': ['A', 'B', 'A', 'C'],
    'infected': [False, True, False, True],
})
edges = pd.DataFrame({
    's': ['a', 'b', 'c', 'a'],
    'd': ['b', 'c', 'd', 'c'],
    'weight': [7, 9, 3, 5],
    'status': ['active', 'active', 'inactive', 'active'],
    'type': ['knows', 'works_at', 'knows', 'likes'],
    'label': ['knows', 'works_at', 'knows', 'likes'],
    'timestamp': pd.to_datetime(['2023-06-15', '2023-07-20', '2023-08-10', '2023-09-05']),
    'relationship': ['friend', 'colleague', 'friend', 'manager'],
    'protocol': ['TCP', 'UDP', 'TCP', 'HTTP'],
    'e_type': ['internal', 'external', 'internal', 'external'],
})
g = graphistry.nodes(nodes, 'id').edges(edges, 's', 'd')
# Aliases for translate.rst pandas examples
nodes_df = nodes
edges_df = edges
# Graph without 'name' column for igraph compatibility
nodes_no_name = nodes.drop(columns=['name'])
g_no_name = graphistry.nodes(nodes_no_name, 'id').edges(edges, 's', 'd')
g_edges_only = graphistry.edges(edges, 's', 'd')
# Aliases for translate.rst examples that use src/dst column names
edges_df['src'] = edges_df['s']
edges_df['dst'] = edges_df['d']
# Add date column to edges for datetime examples
edges['date'] = [date(2023, 6, 15), date(2023, 7, 20), date(2023, 8, 10), date(2023, 9, 5)]
# Extra node columns for remaining examples
nodes['risk1'] = [3, 7, 1, 9]
nodes['risk2'] = [1, 5, 8, 2]
nodes['amount'] = [100, 500, 200, 1000]
nodes['balance'] = [1000, 5000, 200, 50000]
# Dummy policy variables for policy.rst conceptual blocks
query = [n()]
policy_dict = {}
"""

# ---------------------------------------------------------------------------
# Block extraction with doc-test markers
# ---------------------------------------------------------------------------

Marker = Literal["skip", "xfail", None]


def _extract_code_blocks(path: Path) -> List[Tuple[int, str, Marker]]:
    """Extract code blocks with their doc-test marker (if any)."""
    text = path.read_text()
    blocks: List[Tuple[int, str, Marker]] = []

    if path.suffix == ".rst":
        # Look for optional ``.. doc-test: <marker>`` comment before code-block
        pattern = re.compile(
            r"(?:^\.\.\s+doc-test:\s*(skip|xfail)\s*\n\s*\n)?"
            r"^\.\.\s+code-block::\s+python\s*\n"
            r"((?:\n|\s*\n|[ \t]+[^\n]*\n)*)",
            re.MULTILINE,
        )
        for m in pattern.finditer(text):
            marker = m.group(1)  # "skip", "xfail", or None
            raw = m.group(2)
            lines = raw.split("\n")
            code_lines = [ln for ln in lines if ln.strip()]
            if not code_lines:
                continue
            indent = min(len(ln) - len(ln.lstrip()) for ln in code_lines)
            code = "\n".join(ln[indent:] for ln in lines).strip()
            line_no = text[: m.start()].count("\n") + 1
            blocks.append((line_no, code, marker))

    elif path.suffix == ".md":
        # Look for optional <!-- doc-test: <marker> --> before ```python
        pattern = re.compile(
            r"(?:^<!--\s*doc-test:\s*(skip|xfail)\s*-->\s*\n)?"
            r"^```python\s*\n(.*?)^```",
            re.MULTILINE | re.DOTALL,
        )
        for m in pattern.finditer(text):
            marker = m.group(1)
            code = m.group(2).strip()
            line_no = text[: m.start()].count("\n") + 1
            blocks.append((line_no, code, marker))

    return blocks


def _should_skip(code: str) -> Optional[str]:
    for pat in SKIP_PATTERNS:
        if pat in code:
            return pat
    return None


# ---------------------------------------------------------------------------
# Per-file runner
# ---------------------------------------------------------------------------

def _run_file_blocks(path: Path) -> List[Tuple[int, str, Optional[str], Marker]]:
    """Run all blocks in a file. Return [(line, snippet, error_or_None, marker)]."""
    blocks = _extract_code_blocks(path)
    results: List[Tuple[int, str, Optional[str], Marker]] = []

    ns: dict = {}
    try:
        exec(compile(SETUP_CODE, "setup", "exec"), ns)
    except Exception as e:
        return [(0, "SETUP", str(e), None)]

    for line_no, code, marker in blocks:
        # Explicit doc-test: skip
        if marker == "skip":
            continue

        # Auto-skip GPU/remote/etc
        if _should_skip(code):
            continue

        try:
            exec(compile(code, f"{path.name}:{line_no}", "exec"), ns)
            results.append((line_no, code[:80], None, marker))
        except Exception as e:
            err = str(e).split("\n")[0][:200]
            results.append((line_no, code[:80], err, marker))

    return results


# ---------------------------------------------------------------------------
# Pytest
# ---------------------------------------------------------------------------

_doc_files = sorted(
    p for p in GFQL_DOCS.rglob("*") if p.suffix in (".rst", ".md")
)


@pytest.mark.parametrize(
    "doc_path",
    _doc_files,
    ids=[str(p.relative_to(GFQL_DOCS)) for p in _doc_files],
)
def test_gfql_doc_examples(doc_path: Path) -> None:
    """All non-skipped code blocks in a GFQL doc file must execute (or be marked xfail)."""
    os.environ.setdefault("PYGRAPHISTRY_ROOT", str(Path(__file__).resolve().parent.parent))
    results = _run_file_blocks(doc_path)
    if not results:
        pytest.skip("no runnable code blocks")

    # xfail blocks that fail are fine; xfail blocks that pass get noted
    real_failures = []
    for line, code, err, marker in results:
        if err is not None and marker != "xfail":
            real_failures.append((line, err))
        # xfail that passes — not an error, just means we could un-xfail it

    if real_failures:
        msg = [f"  {doc_path.name}:{line}: {err}" for line, err in real_failures]
        pytest.fail(
            f"{len(real_failures)} code example(s) failed in {doc_path.name}:\n"
            + "\n".join(msg)
        )
