"""Guard: a polars-gated test may not be invisible to CI.

WHY THIS EXISTS (measured, not hypothetical). polars is installed in exactly ONE CI lane
-- ``test-polars``, driven by ``bin/test-polars.sh`` -- and that lane runs an explicit file
list. Every other lane collection-skips a module-level ``pytest.importorskip("polars")`` and
per-test ``polars``/``polars-gpu`` parameters for want of the wheel. So a polars-gated test
that the lane's file list omits executes NOWHERE, while both lanes report green.

Measured on GitHub Actions run 30311675717: 280 tests ran locally with polars installed and
appeared in no CI job log at all, including whole modules named ``test_engine_polars_*``.
Two real defects were sitting in that blind spot (a polars-gpu test that fails without the
RAPIDS stack, and a pandas/polars ``toUpper`` divergence).

The rule enforced here: every test module that mentions polars is EITHER in the lane's file
list OR carries a written reason for being out of it. Silence is not a valid answer.
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Union

import pytest


REPO_ROOT = Path(__file__).resolve().parents[4]
LANE_SCRIPT = REPO_ROOT / "bin" / "test-polars.sh"
TEST_ROOTS = ("graphistry/tests", "tests")

# Second lane phase: this module runs under ``-k polars``, so only node ids containing
# "polars" execute. Kept out of the main array on purpose (runtime), but its polars-gated
# tests still have to be selectable -- enforced by
# ``test_lowering_polars_gated_tests_are_named_so_the_k_filter_selects_them``.
K_FILTERED_MODULE = "graphistry/tests/compute/gfql/cypher/test_lowering.py"

# This guard itself: it is pure static analysis over source text, needs no polars wheel, and
# only matches its own scanners because it QUOTES the gate it looks for.
SELF = "graphistry/tests/compute/gfql/test_polars_lane_completeness.py"

# Modules that mention polars but are NOT polars-gated, each with the reason. A new entry
# here is a deliberate, reviewable statement -- which is the point of the guard.
NOT_POLARS_GATED: Dict[str, str] = {
    "graphistry/tests/compute/gfql/cypher/test_row_pushdown.py": (
        "single engine-agnostic to_pandas() branch; every test runs on the pandas lane"
    ),
    "graphistry/tests/test_compute_filter_by_dict.py": (
        "pandas oracle only; the sentence naming polars is a docstring cross-reference"
    ),
    "graphistry/tests/compute/gfql/index/test_index_gpu_edge_match.py": (
        "cudf/GPU-gated (module-level importorskip('cudf') + skipif no GPU), not polars-gated; "
        "belongs to the separate GPU-lane gap, and the polars CPU lane could not run it"
    ),
}

_POLARS_IMPORTORSKIP = re.compile(r"""importorskip\(\s*["']polars["']""")


def _lane_file_list() -> List[str]:
    """Parse ``POLARS_TEST_FILES`` out of bin/test-polars.sh.

    Deliberately strict: if the array cannot be found the guard fails rather than passing
    vacuously, so restructuring the script cannot silently disable this check.
    """
    text = LANE_SCRIPT.read_text(encoding="utf-8")
    match = re.search(r"^POLARS_TEST_FILES=\((.*?)^\)", text, re.MULTILINE | re.DOTALL)
    assert match is not None, (
        f"{LANE_SCRIPT} no longer declares a POLARS_TEST_FILES=( ... ) array; this guard "
        "parses it as the lane's single source of truth"
    )
    files: List[str] = []
    for raw in match.group(1).splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            files.append(line)
    assert files, "POLARS_TEST_FILES parsed as empty"
    return files


FuncDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]


def _node_source(source_lines: List[str], node: Union[ast.stmt, ast.expr]) -> str:
    """py3.8-safe source slice for an AST node (``ast.unparse`` is 3.9+)."""
    start: int = node.lineno
    end: int = node.end_lineno if node.end_lineno is not None else start
    return "\n".join(source_lines[max(start - 1, 0):end])


def _selected_by_k_polars(source_lines: List[str], node: FuncDef) -> bool:
    """True when ``-k polars`` can select the node: either the function name carries it, or a
    decorator does (a ``parametrize`` over an engine id containing 'polars' puts it in the
    node id, which is what ``-k`` matches against)."""
    if "polars" in node.name:
        return True
    return any(
        "polars" in _node_source(source_lines, dec) for dec in node.decorator_list
    )


def _test_modules_mentioning_polars() -> Set[str]:
    found: Set[str] = set()
    for root in TEST_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("test_*.py"):
            if "polars" in path.read_text(encoding="utf-8").lower():
                found.add(path.relative_to(REPO_ROOT).as_posix())
    return found


def test_lane_file_list_paths_all_exist() -> None:
    """A renamed or deleted test file must not silently shrink the lane."""
    missing = [rel for rel in _lane_file_list() if not (REPO_ROOT / rel).is_file()]
    assert missing == [], f"bin/test-polars.sh lists files that do not exist: {missing}"


def test_every_polars_mentioning_test_module_is_in_the_lane_or_justified() -> None:
    """The completeness rule. New polars test file => lane entry or written exemption."""
    lane = set(_lane_file_list()) | {K_FILTERED_MODULE, SELF}
    unclassified = sorted(
        _test_modules_mentioning_polars() - lane - set(NOT_POLARS_GATED)
    )
    assert unclassified == [], (
        "these test modules mention polars but run in NO CI lane (polars is installed only "
        "in test-polars, which runs an explicit file list): "
        f"{unclassified}. Add each to POLARS_TEST_FILES in bin/test-polars.sh, or to "
        "NOT_POLARS_GATED here with the reason it needs no polars lane."
    )


def test_not_polars_gated_entries_are_not_stale() -> None:
    """An exemption for a file that is gone, or that the lane now runs, must be removed."""
    lane = set(_lane_file_list()) | {K_FILTERED_MODULE}
    gone = sorted(rel for rel in NOT_POLARS_GATED if not (REPO_ROOT / rel).is_file())
    assert gone == [], f"NOT_POLARS_GATED names files that no longer exist: {gone}"
    contradictory = sorted(set(NOT_POLARS_GATED) & lane)
    assert contradictory == [], (
        f"NOT_POLARS_GATED claims these need no polars lane, but the lane runs them: "
        f"{contradictory}"
    )


def test_no_module_level_polars_gate_outside_the_lane() -> None:
    """The sharpest form of the hole: a module-level importorskip skips the WHOLE file
    everywhere else, so omission from the lane costs every test in it at once."""
    lane = set(_lane_file_list()) | {K_FILTERED_MODULE, SELF}
    offenders: List[str] = []
    for root in TEST_ROOTS:
        base = REPO_ROOT / root
        if not base.is_dir():
            continue
        for path in base.rglob("test_*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in lane:
                continue
            source = path.read_text(encoding="utf-8")
            source_lines = source.splitlines()
            for node in ast.parse(source).body:  # module level only
                if _POLARS_IMPORTORSKIP.search(_node_source(source_lines, node)):
                    offenders.append(rel)
                    break
    assert offenders == [], (
        "module-level pytest.importorskip('polars') outside the polars lane -- every test in "
        f"these files is collection-skipped in every CI lane: {offenders}"
    )


def test_lowering_polars_gated_tests_are_named_so_the_k_filter_selects_them() -> None:
    """``test_lowering.py`` runs under ``-k polars``. A test that calls
    ``importorskip("polars")`` but has no 'polars' in its function name is deselected in the
    polars lane and skipped in the pandas lanes -- i.e. it runs nowhere."""
    path = REPO_ROOT / K_FILTERED_MODULE
    source = path.read_text(encoding="utf-8")
    source_lines = source.splitlines()
    offenders = [
        node.name
        for node in ast.parse(source).body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
        and not _selected_by_k_polars(source_lines, node)
        and _POLARS_IMPORTORSKIP.search(_node_source(source_lines, node))
    ]
    assert offenders == [], (
        f"{K_FILTERED_MODULE} is run with `-k polars`; these polars-gated tests are not "
        f"selected by that filter and therefore run in no lane: {offenders}. Put 'polars' in "
        "the test name (or parametrize it over a 'polars' engine id)."
    )


@pytest.mark.parametrize("rel", sorted(NOT_POLARS_GATED))
def test_exemption_reasons_are_substantive(rel: str) -> None:
    """An empty or placeholder reason would re-open the hole under the guard's own cover."""
    reason = NOT_POLARS_GATED[rel]
    assert len(reason) >= 40, f"exemption for {rel} needs a real reason, got {reason!r}"
