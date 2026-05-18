from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_module() -> Any:
    path = Path(__file__).resolve().parents[2] / "bin" / "coverage_audit.py"
    spec = importlib.util.spec_from_file_location("coverage_audit", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_normalize_pytest_args_uses_profile_default_after_separator_only() -> None:
    module = _load_module()
    profile = module.PROFILES["gfql"]

    assert module._normalize_pytest_args(["--"], profile) == list(profile.default_pytest_args)


def test_gfql_profile_resolves_current_target_files_and_broad_tests() -> None:
    module = _load_module()
    profile = module.PROFILES["gfql"]
    target_files = [module._repo_rel(path) for path in module._resolve_target_paths(profile)]

    assert profile.name == "gfql"
    assert "graphistry/compute/gfql/cypher/parser.py" in target_files
    assert "graphistry/compute/gfql/cypher/reentry/scope.py" in target_files
    assert "graphistry/compute/gfql/row/entity_text.py" in target_files
    assert "graphistry/tests/compute/gfql" in profile.default_pytest_args


def test_symbol_coverage_reports_zero_hit_symbol() -> None:
    module = _load_module()
    symbol = module.SymbolSpan(kind="function", name="dead_helper", start_line=10, end_line=14)

    hit = module._symbol_coverage(
        symbol=symbol,
        statements={10, 11, 12, 13, 14},
        executed=set(),
        low_symbol_percent=20.0,
    )

    assert hit is not None
    assert hit.name == "dead_helper"
    assert hit.covered_lines == 0
    assert hit.coverage_percent == 0.0


def test_symbol_coverage_ignores_well_covered_symbol() -> None:
    module = _load_module()
    symbol = module.SymbolSpan(kind="function", name="hot_helper", start_line=1, end_line=5)

    hit = module._symbol_coverage(
        symbol=symbol,
        statements={1, 2, 3, 4, 5},
        executed={1, 2, 3, 4, 5},
        low_symbol_percent=20.0,
    )

    assert hit is None


def test_baseline_compare_fails_when_file_drops_below_floor() -> None:
    module = _load_module()
    files = [
        module.FileCoverage(
            path="graphistry/compute/gfql/cypher/parser.py",
            statements=100,
            covered_lines=89,
            missing_lines=11,
            coverage_percent=89.0,
            zero_hit_symbols=[],
            low_hit_symbols=[],
        )
    ]

    checks = module._compare_baseline(
        files=files,
        baseline={"graphistry/compute/gfql/cypher/parser.py": 90.0},
        tolerance_percent=0.5,
    )

    assert checks[0].status == "fail"
    assert checks[0].delta_percent == -1.0
    assert checks[0].reason == "below floor"


def test_baseline_compare_allows_small_tolerance() -> None:
    module = _load_module()
    files = [
        module.FileCoverage(
            path="graphistry/compute/gfql/cypher/parser.py",
            statements=100,
            covered_lines=89,
            missing_lines=11,
            coverage_percent=89.75,
            zero_hit_symbols=[],
            low_hit_symbols=[],
        )
    ]

    checks = module._compare_baseline(
        files=files,
        baseline={"graphistry/compute/gfql/cypher/parser.py": 90.0},
        tolerance_percent=0.5,
    )

    assert checks[0].status == "pass"
    assert checks[0].delta_percent == -0.25
    assert checks[0].reason == "within floor"


def test_baseline_compare_fails_when_target_missing_from_baseline() -> None:
    module = _load_module()
    files = [
        module.FileCoverage(
            path="graphistry/compute/gfql/cypher/new_helper.py",
            statements=10,
            covered_lines=0,
            missing_lines=10,
            coverage_percent=0.0,
            zero_hit_symbols=[],
            low_hit_symbols=[],
        )
    ]

    checks = module._compare_baseline(
        files=files,
        baseline={},
        tolerance_percent=0.5,
    )

    assert checks == []

    checks = module._compare_baseline(
        files=files,
        baseline={"graphistry/compute/gfql/cypher/parser.py": 90.0},
        tolerance_percent=0.5,
    )

    assert [check.status for check in checks] == ["fail", "fail"]
    assert checks[0].reason == "baseline file is no longer in the resolved target set"
    assert checks[1].reason == "resolved target file is missing from the baseline"


def test_load_baseline_accepts_object_schema(tmp_path: Path) -> None:
    module = _load_module()
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "tolerance_percent": 0.25,
                "files": {
                    "graphistry/compute/gfql/cypher/parser.py": {
                        "min_coverage_percent": 85.25,
                        "statements": 949,
                        "covered_lines": 809,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    files, tolerance = module._load_baseline(str(baseline))

    assert files == {"graphistry/compute/gfql/cypher/parser.py": 85.25}
    assert tolerance == 0.25
