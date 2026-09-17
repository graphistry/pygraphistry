"""Pins for bin/ci_pyright_guard.py: what the ratchet must fail on, and what it must not."""

import importlib.util
import json
import os
import sys
from typing import Any, Dict, List

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUARD_PATH = os.path.join(REPO_ROOT, "bin", "ci_pyright_guard.py")
BASELINE_PATH = os.path.join(REPO_ROOT, "bin", "ci_pyright_baseline.json")


pytestmark = pytest.mark.skipif(
    not os.path.exists(GUARD_PATH), reason="guard script is not shipped in the installed package"
)


def _load_guard() -> Any:
    spec = importlib.util.spec_from_file_location("ci_pyright_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard() if os.path.exists(GUARD_PATH) else None


def diagnostic(rule: str, path: str, line: int = 1) -> Dict[str, Any]:
    return {
        "file": os.path.join(REPO_ROOT, path),
        "severity": "error",
        "rule": rule,
        "message": "%s at %s:%d" % (rule, path, line),
        "range": {"start": {"line": line - 1, "character": 0}},
    }


def report(diagnostics: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"version": "1.1.414", "generalDiagnostics": diagnostics}


def run(tmp_path, diagnostics, baseline, argv=()) -> int:
    report_path = tmp_path / "pyright.json"
    report_path.write_text(json.dumps(report(diagnostics)), encoding="utf-8")
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    return GUARD.main(
        ["--from-json", str(report_path), "--baseline", str(baseline_path)] + list(argv)
    )


EMPTY_BASELINE = {"rules": {}}


def test_a_new_finding_in_a_ratcheted_rule_fails(tmp_path):
    found = [diagnostic("reportPossiblyUnboundVariable", "graphistry/a.py")]
    assert run(tmp_path, found, EMPTY_BASELINE) == 1


def test_a_new_finding_in_a_rule_that_is_not_ratcheted_passes(tmp_path):
    found = [diagnostic("reportAttributeAccessIssue", "graphistry/a.py")]
    assert run(tmp_path, found, EMPTY_BASELINE) == 0


def test_a_finding_within_its_file_budget_passes(tmp_path):
    found = [diagnostic("reportUndefinedVariable", "graphistry/a.py")]
    baseline = {"rules": {"reportUndefinedVariable": {"graphistry/a.py": 1}}}
    assert run(tmp_path, found, baseline) == 0


def test_one_more_finding_than_the_budget_fails(tmp_path):
    found = [
        diagnostic("reportUndefinedVariable", "graphistry/a.py", 1),
        diagnostic("reportUndefinedVariable", "graphistry/a.py", 2),
    ]
    baseline = {"rules": {"reportUndefinedVariable": {"graphistry/a.py": 1}}}
    assert run(tmp_path, found, baseline) == 1


def test_a_budget_does_not_carry_across_files(tmp_path):
    """Moved code is new code: b.py may not spend a.py's allowance."""
    found = [diagnostic("reportUndefinedVariable", "graphistry/b.py")]
    baseline = {"rules": {"reportUndefinedVariable": {"graphistry/a.py": 1}}}
    assert run(tmp_path, found, baseline) == 1


def test_a_budget_does_not_carry_across_rules(tmp_path):
    found = [diagnostic("reportUnusedExpression", "graphistry/a.py")]
    baseline = {"rules": {"reportUndefinedVariable": {"graphistry/a.py": 1}}}
    assert run(tmp_path, found, baseline) == 1


def test_an_improvement_passes_but_strict_demands_the_baseline_be_relocked(tmp_path):
    baseline = {"rules": {"reportUndefinedVariable": {"graphistry/a.py": 2}}}
    found = [diagnostic("reportUndefinedVariable", "graphistry/a.py")]
    assert run(tmp_path, found, baseline) == 0
    assert run(tmp_path, found, baseline, ["--strict"]) == 1


def test_report_always_exits_zero_even_over_budget(tmp_path):
    found = [diagnostic("reportPossiblyUnboundVariable", "graphistry/a.py")]
    assert run(tmp_path, found, EMPTY_BASELINE, ["--report"]) == 0


def test_every_ratcheted_rule_is_decided_without_consulting_a_type(tmp_path):
    """The gate's whole claim: these verdicts cannot move with the installed packages.

    Rules that read third-party stubs vary by an order of magnitude between
    environments, so admitting one here would make CI disagree with the developer
    who ran the same command.
    """
    assert set(GUARD.RATCHETED_RULES) == {
        "reportPossiblyUnboundVariable",
        "reportSelfClsParameterName",
        "reportUndefinedVariable",
        "reportUnsupportedDunderAll",
        "reportUnusedExpression",
    }


@pytest.mark.skipif(not os.path.exists(BASELINE_PATH), reason="baseline is not shipped")
def test_the_committed_baseline_covers_exactly_the_ratcheted_rules():
    with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
        baseline = json.load(handle)
    assert set(baseline["rules"]) == set(GUARD.RATCHETED_RULES)
    assert baseline["pyright_version"], "the baseline must record the pyright it was built with"


@pytest.mark.skipif(not os.path.exists(BASELINE_PATH), reason="baseline is not shipped")
def test_the_committed_baseline_is_what_the_pinned_pyright_version_produced():
    """A baseline built by a different pyright than bin/pyright.sh runs is not a baseline."""
    with open(BASELINE_PATH, "r", encoding="utf-8") as handle:
        recorded = json.load(handle)["pyright_version"]
    with open(os.path.join(REPO_ROOT, "bin", "pyright.sh"), "r", encoding="utf-8") as handle:
        script = handle.read()
    assert 'PYRIGHT_VERSION="${PYRIGHT_VERSION:-%s}"' % recorded in script


def test_writing_a_baseline_records_the_pyright_that_produced_it(tmp_path):
    """Without the version, a baseline cannot be told apart from one built by another tool."""
    out = tmp_path / "baseline.json"
    GUARD.write_baseline(str(out), {"reportUndefinedVariable": {"graphistry/a.py": 2}}, "1.2.3")
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["pyright_version"] == "1.2.3"
    assert "1.2.3" in written["_comment"]


def test_a_written_baseline_reads_back_as_the_same_budgets(tmp_path):
    out = tmp_path / "baseline.json"
    counts = {"reportUndefinedVariable": {"graphistry/a.py": 2}}
    GUARD.write_baseline(str(out), counts, "1.1.414")
    loaded = GUARD.load_baseline(str(out))
    assert loaded["reportUndefinedVariable"] == {"graphistry/a.py": 2}
    assert set(loaded) == set(GUARD.RATCHETED_RULES)
