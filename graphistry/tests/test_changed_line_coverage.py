from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "bin" / "changed_line_coverage.py"
    spec = importlib.util.spec_from_file_location("changed_line_coverage", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parse_changed_lines_tracks_added_new_file_lines() -> None:
    module = _load_module()
    diff = """diff --git a/graphistry/example.py b/graphistry/example.py
--- a/graphistry/example.py
+++ b/graphistry/example.py
@@ -10,0 +11,2 @@
+first = 1
+second = 2
"""

    assert module._parse_changed_lines(diff) == {"graphistry/example.py": {11, 12}}


def test_parse_changed_lines_tracks_modified_replacement_lines() -> None:
    module = _load_module()
    diff = """diff --git a/graphistry/example.py b/graphistry/example.py
--- a/graphistry/example.py
+++ b/graphistry/example.py
@@ -7 +7 @@
-old = 1
+new = 2
@@ -20,2 +20,3 @@
 context = True
-old_call()
+new_call()
+extra_call()
"""

    assert module._parse_changed_lines(diff) == {"graphistry/example.py": {7, 21, 22}}


def test_eligible_path_defaults_include_package_and_exclude_tests() -> None:
    module = _load_module()

    assert module._is_eligible_path("graphistry/compute/foo.py", module.DEFAULT_INCLUDE, module.DEFAULT_EXCLUDE)
    assert not module._is_eligible_path("graphistry/tests/test_foo.py", module.DEFAULT_INCLUDE, module.DEFAULT_EXCLUDE)
    assert not module._is_eligible_path("bin/changed_line_coverage.py", module.DEFAULT_INCLUDE, module.DEFAULT_EXCLUDE)


def test_build_report_passes_when_changed_statements_meet_threshold() -> None:
    module = _load_module()
    source = Path(module.REPO_ROOT) / "graphistry" / "__init__.py"

    class FakeCov:
        def analysis2(self, path: str) -> Any:
            assert path == str(source)
            return (path, [1, 2, 3], [], [3], "")

    report = module._build_report(
        cov=FakeCov(),
        changed_lines={"graphistry/__init__.py": {1, 2, 3, 4}},
        base_ref="base",
        head_ref="head",
        min_percent=60.0,
        includes=module.DEFAULT_INCLUDE,
        excludes=module.DEFAULT_EXCLUDE,
    )

    assert report.status == "pass"
    assert report.changed_statement_lines == 3
    assert report.covered_lines == 2
    assert report.missing_lines == 1
    assert report.coverage_percent == 66.67
    assert report.files[0].missing_line_numbers == [3]


def test_build_report_fails_below_threshold() -> None:
    module = _load_module()
    source = Path(module.REPO_ROOT) / "graphistry" / "__init__.py"

    class FakeCov:
        def analysis2(self, path: str) -> Any:
            assert path == str(source)
            return (path, [1, 2, 3, 4], [], [2, 3, 4], "")

    report = module._build_report(
        cov=FakeCov(),
        changed_lines={"graphistry/__init__.py": {1, 2, 3, 4}},
        base_ref="base",
        head_ref="head",
        min_percent=80.0,
        includes=module.DEFAULT_INCLUDE,
        excludes=module.DEFAULT_EXCLUDE,
    )

    assert report.status == "fail"
    assert report.coverage_percent == 25.0
