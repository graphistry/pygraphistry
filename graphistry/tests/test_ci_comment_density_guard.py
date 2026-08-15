"""Pins for bin/ci_comment_density_guard.py: what it must catch, and what it must not."""

import importlib.util
import io
import os
import sys
from typing import List, Sequence

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUARD_PATH = os.path.join(REPO_ROOT, "bin", "ci_comment_density_guard.py")


pytestmark = pytest.mark.skipif(
    not os.path.exists(GUARD_PATH), reason="guard script is not shipped in the installed package"
)


def _load_guard() -> object:
    spec = importlib.util.spec_from_file_location("ci_comment_density_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard() if os.path.exists(GUARD_PATH) else None


def checks_for(tmp_path, source: str, checks: Sequence[str] = ()) -> List[str]:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    selected = list(checks) or list(GUARD.CHECKS)  # type: ignore[attr-defined]
    findings = GUARD.analyze_file(str(path), "sample.py", selected)  # type: ignore[attr-defined]
    return sorted(f.check for f in findings)


def test_two_line_comment_run_in_a_function_is_a_comment_block(tmp_path):
    source = "x = 1\n\n\ndef f():\n    # narrate the next block\n    # and keep narrating\n    return 1\n"
    assert checks_for(tmp_path, source) == ["comment-block"]


def test_single_line_comment_is_allowed(tmp_path):
    source = "x = 1\n\n\ndef f():\n    # the one constraint nothing else states\n    return 1\n"
    assert checks_for(tmp_path, source) == []


def test_module_level_comment_run_after_the_header_is_a_comment_block(tmp_path):
    source = "import os\n\n# first narration line\n# second narration line\n\nY = os.sep\n"
    assert checks_for(tmp_path, source) == ["comment-block"]


def test_leading_file_header_comment_run_is_exempt(tmp_path):
    source = "#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# Copyright line one\n# Copyright line two\nX = 1\n"
    assert checks_for(tmp_path, source) == []


def test_tool_directives_neither_start_nor_continue_a_run(tmp_path):
    source = "def f(a):\n    # type: (int) -> int\n    # noqa: E501\n    return a\n"
    assert checks_for(tmp_path, source) == []


def test_sphinx_run_is_allowed_at_two_lines_and_flagged_at_three(tmp_path):
    two = "X = 1\n\n#: first documented line\n#: second documented line\nY = 2\n"
    three = "X = 1\n\n#: first\n#: second\n#: third\nY = 2\n"
    assert checks_for(tmp_path, two) == []
    assert checks_for(tmp_path, three) == ["comment-block"]


@pytest.mark.parametrize("claim", [
    "one O(E) pass over the edges",
    "this shape is faster than the join",
    "the isin scan is expensive here",
    "the hot path pays nothing",
    "an A/B run showed the speedup",
])
def test_perf_vocabulary_in_a_comment_is_a_perf_claim(tmp_path, claim):
    assert "perf-claim" in checks_for(tmp_path, "X = 1  # %s\n" % claim, ["perf-claim"])


def test_perf_vocabulary_in_a_docstring_is_a_perf_claim(tmp_path):
    source = 'def f():\n    """Gather rows. O(len(rows))."""\n    return 1\n'
    assert checks_for(tmp_path, source, ["perf-claim"]) == ["perf-claim"]


@pytest.mark.parametrize("call", ["foo(x)", "into(rows)", "o(n)", "do(work)"])
def test_a_lowercase_call_is_not_asymptotic_notation(tmp_path, call):
    source = "X = 1  # delegates to %s for the rebind\n" % call
    assert checks_for(tmp_path, source, ["perf-claim"]) == []


def test_uppercase_complexity_notation_is_asymptotic_notation(tmp_path):
    source = "X = 1  # delegates to O(n) for the rebind\n"
    assert checks_for(tmp_path, source, ["perf-claim"]) == ["perf-claim"]


def test_correctness_regression_without_performance_context_is_not_a_perf_claim(tmp_path):
    source = "X = 1  # regression guard: the fallback must stay reachable\n"
    assert checks_for(tmp_path, source, ["perf-claim"]) == []


def test_regression_next_to_performance_vocabulary_is_a_perf_claim(tmp_path):
    source = "X = 1  # a perf regression on large edge frames\n"
    assert checks_for(tmp_path, source, ["perf-claim"]) == ["perf-claim"]


def test_pointing_at_pyg_bench_is_not_a_perf_claim(tmp_path):
    source = "X = 1  # shape coverage lives in pyg-bench, not here\n"
    assert checks_for(tmp_path, source, ["perf-claim"]) == []


def test_standalone_issue_citation_is_issue_rationale(tmp_path):
    source = "def f():\n    # #1891: null-extend the arm schema\n    return 1\n"
    assert "issue-rationale" in checks_for(tmp_path, source)


def test_trailing_issue_tag_on_a_line_of_code_is_allowed(tmp_path):
    assert checks_for(tmp_path, "X = compute()  # #1891\n") == []


def test_docstring_citing_an_issue_is_issue_rationale(tmp_path):
    source = 'def f():\n    """Flatten the carry stage (#1896)."""\n    return 1\n'
    assert checks_for(tmp_path, source, ["issue-rationale"]) == ["issue-rationale"]


def test_guard_ok_above_a_run_suppresses_comment_block(tmp_path):
    source = (
        "def f():\n"
        "    # guard-ok: comment-block -- spec wording quoted verbatim\n"
        "    # narration one\n"
        "    # narration two\n"
        "    return 1\n"
    )
    assert checks_for(tmp_path, source, ["comment-block"]) == []


def test_guard_ok_on_the_reported_line_suppresses_a_perf_claim(tmp_path):
    source = "X = 1  # one O(E) pass  # guard-ok: perf-claim -- external API contract\n"
    assert checks_for(tmp_path, source, ["perf-claim"]) == []


def test_tests_are_exempt_from_comment_block_but_not_from_perf_claim():
    assert GUARD.is_test_path("graphistry/tests/compute/test_hop.py")  # type: ignore[attr-defined]
    assert not GUARD.is_test_path("graphistry/compute/hop.py")  # type: ignore[attr-defined]
    skipped = GUARD.CHECKS_SKIPPING_TESTS  # type: ignore[attr-defined]
    assert skipped == frozenset(["comment-block", "issue-rationale"])


def test_repo_is_at_or_below_its_committed_baseline():
    assert GUARD.main([]) == 0  # type: ignore[attr-defined]


def test_every_guard_baseline_triggers_the_python_ci_lane():
    """A new guard must not need a ci.yml edit to have its baseline watched."""
    import glob
    import re

    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    workflow = io.open(
        os.path.join(root, ".github", "workflows", "ci.yml"), encoding="utf-8"
    ).read()
    block = workflow.split("emit python")[1].split("emit ")[0]
    patterns = [re.compile(m) for m in re.findall(r"'([^']+)'", block)]

    baselines = sorted(
        os.path.relpath(p, root).replace(os.sep, "/")
        for p in glob.glob(os.path.join(root, "bin", "ci_*baseline*.json"))
    )
    assert baselines, "no guard baselines found -- the glob or the layout changed"
    unwatched = [b for b in baselines if not any(p.search(b) for p in patterns)]
    assert unwatched == [], f"guard baselines not watched by the python lane: {unwatched}"
