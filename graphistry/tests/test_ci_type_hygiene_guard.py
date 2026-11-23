"""Pins for the dynamic-attr check of bin/ci/ci_type_hygiene_guard.py: what it must catch, and what it must not."""

import importlib.util
import json
import os
import sys
from typing import List

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GUARD_PATH = os.path.join(REPO_ROOT, "bin", "ci", "ci_type_hygiene_guard.py")


pytestmark = pytest.mark.skipif(
    not os.path.exists(GUARD_PATH), reason="guard script is not shipped in the installed package"
)


def _load_guard() -> object:
    spec = importlib.util.spec_from_file_location("ci_type_hygiene_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


GUARD = _load_guard() if os.path.exists(GUARD_PATH) else None


def findings_for(tmp_path, source: str, check: str = "dynamic-attr") -> List[int]:
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    return sorted(f.line for f in GUARD.analyze_file(str(path)) if f.check == check)  # type: ignore[attr-defined]


@pytest.mark.parametrize("call", [
    "hasattr(value, 'astype')",
    "getattr(value, 'dtype')",
    "getattr(value, 'dtype', None)",
    "setattr(value, 'name', 'x')",
])
def test_each_probed_builtin_is_flagged(tmp_path, call):
    assert findings_for(tmp_path, "def f(value: object) -> object:\n    return %s\n" % call) == [2]


def test_a_method_named_like_the_builtin_is_not_flagged(tmp_path):
    source = "def f(obj: object) -> object:\n    return obj.getattr('x') or obj.hasattr('y')\n"
    assert findings_for(tmp_path, source) == []


@pytest.mark.parametrize("shadow", [
    "def getattr(obj: object, name: str) -> object:\n    return obj\n\n\n",
    "getattr = lambda obj, name: obj\n\n\n",
    "from helpers import probe as getattr\n\n\n",
])
def test_a_local_definition_of_the_name_is_not_flagged(tmp_path, shadow):
    source = shadow + "def f(value: object) -> object:\n    return getattr(value, 'x')\n"
    assert findings_for(tmp_path, source) == []


def test_suppression_marker_is_honored(tmp_path):
    source = (
        "def f(value: object) -> bool:\n"
        "    return hasattr(value, 'x')  # hygiene-ok: dynamic-attr -- duck-typed third-party object\n"
    )
    assert findings_for(tmp_path, source) == []


def test_setattr_on_a_plottable_is_reported_once_under_plottable_setattr(tmp_path):
    source = "def f(g: Plottable) -> None:\n    setattr(g, 'x', 1)\n"
    assert findings_for(tmp_path, source, "plottable-setattr") == [2]
    assert findings_for(tmp_path, source) == []


def test_setattr_on_a_non_plottable_is_dynamic_attr(tmp_path):
    source = "def f(g: Plottable, other: object) -> None:\n    setattr(other, 'x', 1)\n"
    assert findings_for(tmp_path, source, "plottable-setattr") == []
    assert findings_for(tmp_path, source) == [2]


def _ratchet(tmp_path, monkeypatch, files, baseline):
    root = tmp_path / "repo"
    pkg = root / "graphistry"
    pkg.mkdir(parents=True)
    for name, source in files.items():
        (pkg / name).write_text(source, encoding="utf-8")
    baseline_path = root / "baseline.json"
    counts = dict((check, {}) for check in GUARD.CHECKS)  # type: ignore[attr-defined]
    counts["dynamic-attr"] = baseline
    baseline_path.write_text(json.dumps({"checks": counts}), encoding="utf-8")
    monkeypatch.setattr(GUARD, "REPO_ROOT", str(root))
    monkeypatch.setattr(GUARD, "SCAN_ROOT", str(pkg))
    return GUARD.main(["--baseline", str(baseline_path)])  # type: ignore[attr-defined]


CLEAN = "def f(value: object) -> bool:\n    return isinstance(value, str)\n"
ONE_CALL = "def f(value: object) -> bool:\n    return hasattr(value, 'x')\n"
TWO_CALLS = "def f(value: object) -> bool:\n    return hasattr(value, 'x') and hasattr(value, 'y')\n"


def test_a_file_within_its_baseline_passes(tmp_path, monkeypatch):
    assert _ratchet(tmp_path, monkeypatch, {"mod.py": ONE_CALL}, {"graphistry/mod.py": 1}) == 0


def test_a_file_gaining_one_call_fails(tmp_path, monkeypatch):
    assert _ratchet(tmp_path, monkeypatch, {"mod.py": TWO_CALLS}, {"graphistry/mod.py": 1}) == 1


def test_a_new_file_with_one_call_fails(tmp_path, monkeypatch):
    assert _ratchet(tmp_path, monkeypatch, {"old.py": CLEAN, "new.py": ONE_CALL}, {}) == 1


def test_list_and_report_cover_the_check(tmp_path, monkeypatch, capsys):
    _ratchet(tmp_path, monkeypatch, {"mod.py": TWO_CALLS}, {"graphistry/mod.py": 2})
    assert GUARD.main(["--list", "dynamic-attr"]) == 0  # type: ignore[attr-defined]
    assert "-- 2 finding(s) for dynamic-attr" in capsys.readouterr().out
    assert GUARD.main(["--report"]) == 0  # type: ignore[attr-defined]
    assert "dynamic-attr" in capsys.readouterr().out
