#!/usr/bin/env python3
"""Profile-based coverage audit for dead-code triage and coverage hygiene.

This is an evidence tool, not a deletion oracle. It runs a focused pytest
slice under coverage.py and reports low/zero-hit files and symbols for a
named source profile.

Usage:
    python bin/coverage_audit.py --profile gfql

    python bin/coverage_audit.py --profile gfql --engine-label pandas-cpu \
        --output-dir build/gfql-coverage-audit -- -q path/to/test_file.py

For RAPIDS/cuDF validation, run through docker/test-rapids-official-local.sh on
dgx-spark with WITH_COVERAGE_AUDIT=1, COVERAGE_PROFILE=gfql,
COVERAGE_ENGINE_LABEL=rapids-25.02-cudf or rapids-26.02-cudf, and the matching
COVERAGE_BASELINE_FILE. The DGX box is shared, so prefer focused TEST_FILES
slices when GPU memory is tight.
"""

from __future__ import annotations

import argparse
import ast
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GFQL_TARGET_PATTERNS = (
    "graphistry/compute/ast.py",
    "graphistry/compute/ast_temporal.py",
    "graphistry/compute/chain.py",
    "graphistry/compute/chain_let.py",
    "graphistry/compute/gfql_unified.py",
    "graphistry/compute/gfql/cypher/**/*.py",
    "graphistry/compute/gfql/row/*.py",
    "graphistry/compute/gfql/temporal/*.py",
    "graphistry/compute/gfql/temporal_text.py",
)

GFQL_SOURCE_PATHS = (
    "graphistry/compute",
)

GFQL_DEFAULT_PYTEST_ARGS = (
    "-q",
    "graphistry/tests/compute/test_ast.py",
    "graphistry/tests/compute/test_chain.py",
    "graphistry/tests/compute/test_chain_let.py",
    "graphistry/tests/compute/test_chain_concat.py",
    "graphistry/tests/compute/test_dataframe_primitives.py",
    "graphistry/tests/compute/test_hop.py",
    "graphistry/tests/compute/gfql",
    "tests/gfql/ref",
)


@dataclass(frozen=True)
class AuditProfile:
    name: str
    title: str
    coverage_basis: str
    target_patterns: Tuple[str, ...]
    source_paths: Tuple[str, ...]
    default_pytest_args: Tuple[str, ...]
    next_triage: Tuple[str, ...]


POLARS_TARGET_PATTERNS = (
    "graphistry/compute/gfql/lazy/**/*.py",
)

PROFILES: Dict[str, AuditProfile] = {
    "gfql-polars": AuditProfile(
        name="gfql-polars",
        title="GFQL Polars Engine Coverage Audit",
        coverage_basis="line coverage from `coverage.py` over the polars lane (bin/test-polars.sh)",
        target_patterns=POLARS_TARGET_PATTERNS,
        source_paths=GFQL_SOURCE_PATHS,
        default_pytest_args=GFQL_DEFAULT_PYTEST_ARGS,
        next_triage=(
            "The native polars engine files are exercised only by the polars lane (engine='polars').",
            "Raise floors as coverage grows; keep parity/NIE behavior gated by the conformance suite.",
        ),
    ),
    "gfql": AuditProfile(
        name="gfql",
        title="GFQL Coverage Audit",
        coverage_basis="line coverage from `coverage.py` over focused GFQL tests",
        target_patterns=GFQL_TARGET_PATTERNS,
        source_paths=GFQL_SOURCE_PATHS,
        default_pytest_args=GFQL_DEFAULT_PYTEST_ARGS,
        next_triage=(
            "Confirm any candidate with static references, public-surface checks, and focused positive/negative tests.",
            "Compare pandas and cuDF reports before deleting dataframe-sensitive paths.",
            "File deletion follow-ups as narrow issues with the report path, commit SHA, test command, and engine label.",
        ),
    )
}


@dataclass(frozen=True)
class SymbolSpan:
    kind: str
    name: str
    start_line: int
    end_line: int


@dataclass(frozen=True)
class SymbolCoverage:
    kind: str
    name: str
    start_line: int
    end_line: int
    statement_lines: int
    covered_lines: int
    coverage_percent: float


@dataclass(frozen=True)
class FileCoverage:
    path: str
    statements: int
    covered_lines: int
    missing_lines: int
    coverage_percent: float
    zero_hit_symbols: List[SymbolCoverage]
    low_hit_symbols: List[SymbolCoverage]


@dataclass(frozen=True)
class FileBaselineCheck:
    path: str
    actual_percent: Optional[float]
    min_percent: float
    tolerance_percent: float
    status: str
    delta_percent: Optional[float]
    reason: str


class AuditError(RuntimeError):
    """Coverage audit setup or runtime failure."""


class _SymbolVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.stack: List[str] = []
        self.symbols: List[SymbolSpan] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        self._add("class", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        self._add("function", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        self._add("async_function", node)
        self.stack.append(node.name)
        self.generic_visit(node)
        self.stack.pop()

    def _add(self, kind: str, node: ast.AST) -> None:
        name = getattr(node, "name", "<unknown>")
        qualified = ".".join(self.stack + [name])
        end_line = getattr(node, "end_lineno", getattr(node, "lineno", 0))
        self.symbols.append(
            SymbolSpan(
                kind=kind,
                name=qualified,
                start_line=getattr(node, "lineno", 0),
                end_line=end_line,
            )
        )


def _load_coverage_module() -> Any:
    try:
        import coverage  # type: ignore
    except Exception as exc:
        raise AuditError("coverage.py is required; install the test extra or run `python -m pip install coverage`.") from exc
    return coverage


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _has_glob_magic(pattern: str) -> bool:
    return any(char in pattern for char in "*?[")


def _resolve_target_paths(profile: AuditProfile) -> List[Path]:
    paths: Set[Path] = set()
    for pattern in profile.target_patterns:
        if _has_glob_magic(pattern):
            matches = sorted((REPO_ROOT).glob(pattern))
        else:
            matches = [REPO_ROOT / pattern]
        file_matches = [path.resolve() for path in matches if path.is_file()]
        if not file_matches:
            raise AuditError(f"profile `{profile.name}` target pattern matched no files: {pattern}")
        paths.update(file_matches)
    return sorted(paths, key=_repo_rel)


def _parse_symbols(path: Path) -> List[SymbolSpan]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    visitor = _SymbolVisitor()
    visitor.visit(module)
    return visitor.symbols


def _symbol_coverage(
    symbol: SymbolSpan,
    statements: Set[int],
    executed: Set[int],
    low_symbol_percent: float,
) -> Optional[SymbolCoverage]:
    symbol_statements = {line for line in statements if symbol.start_line <= line <= symbol.end_line}
    if not symbol_statements:
        return None
    covered = symbol_statements & executed
    coverage_percent = 100.0 * len(covered) / len(symbol_statements)
    if covered and coverage_percent > low_symbol_percent:
        return None
    return SymbolCoverage(
        kind=symbol.kind,
        name=symbol.name,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        statement_lines=len(symbol_statements),
        covered_lines=len(covered),
        coverage_percent=round(coverage_percent, 2),
    )


def _collect_file_coverage(cov: Any, target_paths: Sequence[Path], low_symbol_percent: float) -> List[FileCoverage]:
    data = cov.get_data()
    files: List[FileCoverage] = []

    for path in target_paths:
        if not path.exists():
            continue

        _, statements_raw, _, missing_raw, _ = cov.analysis2(str(path))
        statements = set(statements_raw)
        missing = set(missing_raw)
        executed = set(data.lines(str(path)) or [])
        covered = statements - missing
        symbol_hits: List[SymbolCoverage] = []

        for symbol in _parse_symbols(path):
            hit = _symbol_coverage(symbol, statements, executed, low_symbol_percent)
            if hit is not None:
                symbol_hits.append(hit)

        zero_hit_symbols = [hit for hit in symbol_hits if hit.covered_lines == 0]
        low_hit_symbols = [hit for hit in symbol_hits if hit.covered_lines != 0]
        percent = 100.0 * len(covered) / len(statements) if statements else 100.0
        files.append(
            FileCoverage(
                path=_repo_rel(path),
                statements=len(statements),
                covered_lines=len(covered),
                missing_lines=len(missing),
                coverage_percent=round(percent, 2),
                zero_hit_symbols=sorted(zero_hit_symbols, key=lambda x: (x.start_line, x.name)),
                low_hit_symbols=sorted(low_hit_symbols, key=lambda x: (x.coverage_percent, x.start_line, x.name)),
            )
        )

    return files


def _write_markdown(
    path: Path,
    profile: AuditProfile,
    files: Sequence[FileCoverage],
    engine_label: str,
    pytest_args: Sequence[str],
    low_symbol_percent: float,
    max_symbols: int,
    baseline_checks: Sequence[FileBaselineCheck],
) -> None:
    lowest = sorted(files, key=lambda f: (f.coverage_percent, f.path))
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines = [
        f"# {profile.title}",
        "",
        f"- Generated: `{now}`",
        f"- Profile: `{profile.name}`",
        f"- Engine label: `{engine_label}`",
        f"- Python: `{platform.python_version()}`",
        f"- Coverage basis: {profile.coverage_basis}",
        f"- Low-hit symbol threshold: `<= {low_symbol_percent:.2f}%` statement-line coverage",
        "",
        "Coverage is triage evidence only. A zero-hit line or symbol still needs static analysis, API checks, and focused tests before deletion.",
        "",
        "## Test Command",
        "",
        "```bash",
        f"python bin/coverage_audit.py --profile {profile.name} -- " + " ".join(pytest_args),
        "```",
        "",
        "## Target File Summary",
        "",
        "| File | Statements | Covered | Missing | Line coverage | Zero-hit symbols | Low-hit symbols |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    for item in lowest:
        lines.append(
            "| {path} | {statements} | {covered} | {missing} | {percent:.2f}% | {zero} | {low} |".format(
                path=item.path,
                statements=item.statements,
                covered=item.covered_lines,
                missing=item.missing_lines,
                percent=item.coverage_percent,
                zero=len(item.zero_hit_symbols),
                low=len(item.low_hit_symbols),
            )
        )

    if baseline_checks:
        lines.extend(
            [
                "",
                "## Per-File Baseline Lock-In",
                "",
                "| File | Actual | Floor | Tolerance | Delta | Status |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for check in sorted(baseline_checks, key=lambda x: (x.status != "fail", x.path)):
            actual = "missing" if check.actual_percent is None else f"{check.actual_percent:.2f}%"
            delta = "n/a" if check.delta_percent is None else f"{check.delta_percent:+.2f}%"
            lines.append(
                "| {path} | {actual} | {floor:.2f}% | {tolerance:.2f}% | {delta} | {status}: {reason} |".format(
                    path=check.path,
                    actual=actual,
                    floor=check.min_percent,
                    tolerance=check.tolerance_percent,
                    delta=delta,
                    status=check.status,
                    reason=check.reason,
                )
            )

    lines.extend(["", "## Zero-Hit Symbols", ""])
    zero_rows = [
        (item.path, symbol)
        for item in lowest
        for symbol in item.zero_hit_symbols
    ][:max_symbols]
    if zero_rows:
        lines.extend(["| File | Symbol | Lines | Statements |", "|---|---|---:|---:|"])
        for file_path, symbol in zero_rows:
            lines.append(
                f"| {file_path} | `{symbol.kind} {symbol.name}` | {symbol.start_line}-{symbol.end_line} | {symbol.statement_lines} |"
            )
    else:
        lines.append("No zero-hit symbols in the target set.")

    lines.extend(["", "## Low-Hit Symbols", ""])
    low_rows = [
        (item.path, symbol)
        for item in lowest
        for symbol in item.low_hit_symbols
    ][:max_symbols]
    if low_rows:
        lines.extend(["| File | Symbol | Lines | Covered / Statements | Coverage |", "|---|---|---:|---:|---:|"])
        for file_path, symbol in low_rows:
            lines.append(
                "| {file} | `{kind} {name}` | {start}-{end} | {covered} / {statements} | {percent:.2f}% |".format(
                    file=file_path,
                    kind=symbol.kind,
                    name=symbol.name,
                    start=symbol.start_line,
                    end=symbol.end_line,
                    covered=symbol.covered_lines,
                    statements=symbol.statement_lines,
                    percent=symbol.coverage_percent,
                )
            )
    else:
        lines.append("No low-hit symbols above zero in the target set.")

    lines.extend(
        [
            "",
            "## Next Triage",
            "",
            *[f"- {item}" for item in profile.next_triage],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(
    path: Path,
    profile: AuditProfile,
    files: Sequence[FileCoverage],
    engine_label: str,
    pytest_args: Sequence[str],
    exit_code: int,
    baseline_checks: Sequence[FileBaselineCheck],
) -> None:
    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "profile": profile.name,
        "engine_label": engine_label,
        "python": platform.python_version(),
        "pytest_args": list(pytest_args),
        "pytest_exit_code": exit_code,
        "baseline_exit_code": 3 if any(check.status == "fail" for check in baseline_checks) else 0,
        "baseline_checks": [asdict(item) for item in baseline_checks],
        "files": [asdict(item) for item in files],
        "note": "Coverage is triage evidence only, not automatic deletion proof.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _baseline_items(raw: Dict[str, Any]) -> Dict[str, float]:
    files = raw.get("files")
    if not isinstance(files, dict):
        raise AuditError("baseline file must contain a `files` object keyed by repo-relative path")

    out: Dict[str, float] = {}
    for path, value in files.items():
        if isinstance(value, (int, float)):
            out[path] = float(value)
            continue
        if isinstance(value, dict):
            raw_percent = value.get("min_coverage_percent")
            if isinstance(raw_percent, (int, float)):
                out[path] = float(raw_percent)
                continue
        raise AuditError(f"baseline entry for `{path}` must be a number or object with `min_coverage_percent`")
    return out


def _load_baseline(path: Optional[str]) -> Tuple[Dict[str, float], Optional[float]]:
    if path is None:
        return {}, None
    baseline_path = (REPO_ROOT / path).resolve()
    try:
        raw = json.loads(baseline_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise AuditError(f"could not read baseline file `{baseline_path}`") from exc
    except json.JSONDecodeError as exc:
        raise AuditError(f"baseline file `{baseline_path}` is not valid JSON") from exc
    tolerance = raw.get("tolerance_percent")
    if tolerance is not None and not isinstance(tolerance, (int, float)):
        raise AuditError("baseline `tolerance_percent` must be numeric when set")
    return _baseline_items(raw), None if tolerance is None else float(tolerance)


def _compare_baseline(
    files: Sequence[FileCoverage],
    baseline: Dict[str, float],
    tolerance_percent: float,
) -> List[FileBaselineCheck]:
    if not baseline:
        return []
    actual = {item.path: item.coverage_percent for item in files}
    checks: List[FileBaselineCheck] = []
    for path, min_percent in sorted(baseline.items()):
        actual_percent = actual.get(path)
        if actual_percent is None:
            checks.append(
                FileBaselineCheck(
                    path=path,
                    actual_percent=None,
                    min_percent=round(min_percent, 2),
                    tolerance_percent=round(tolerance_percent, 2),
                    status="fail",
                    delta_percent=None,
                    reason="baseline file is no longer in the resolved target set",
                )
            )
            continue
        delta = actual_percent - min_percent
        status = "fail" if actual_percent + tolerance_percent < min_percent else "pass"
        checks.append(
            FileBaselineCheck(
                path=path,
                actual_percent=round(actual_percent, 2),
                min_percent=round(min_percent, 2),
                tolerance_percent=round(tolerance_percent, 2),
                status=status,
                delta_percent=round(delta, 2),
                reason="below floor" if status == "fail" else "within floor",
            )
        )
    for path in sorted(set(actual) - set(baseline)):
        checks.append(
            FileBaselineCheck(
                path=path,
                actual_percent=round(actual[path], 2),
                min_percent=0.0,
                tolerance_percent=round(tolerance_percent, 2),
                status="fail",
                delta_percent=None,
                reason="resolved target file is missing from the baseline",
            )
        )
    return checks


def _normalize_pytest_args(args: Sequence[str], profile: AuditProfile) -> List[str]:
    if not args:
        return list(profile.default_pytest_args)
    out = list(args)
    if out and out[0] == "--":
        out = out[1:]
    return out or list(profile.default_pytest_args)


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="gfql", help="Coverage audit profile to run")
    parser.add_argument("--output-dir", default=None, help="Directory for markdown/json/coverage data")
    parser.add_argument("--engine-label", default="pandas-cpu", help="Report label, e.g. pandas-cpu, rapids-25.02-cudf")
    parser.add_argument("--data-file", default=None, help="coverage.py data file path; defaults under --output-dir")
    parser.add_argument("--low-symbol-percent", type=float, default=20.0, help="Report symbols at or below this coverage percent")
    parser.add_argument("--max-symbols", type=int, default=200, help="Maximum zero/low-hit symbols per markdown section")
    parser.add_argument("--baseline-file", default=None, help="Optional JSON file with per-file coverage floors")
    parser.add_argument(
        "--baseline-tolerance",
        type=float,
        default=None,
        help="Allowed percent-point drift below per-file baseline floors; overrides baseline file tolerance",
    )
    parser.add_argument("--skip-tests", action="store_true", help="Analyze existing coverage data instead of running pytest")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="pytest args after `--`; defaults to the profile's audit set")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    profile = PROFILES[args.profile]
    output_dir = (REPO_ROOT / (args.output_dir or f"build/{profile.name}-coverage-audit")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_file = Path(args.data_file).resolve() if args.data_file else output_dir / ".coverage"
    pytest_args = _normalize_pytest_args(args.pytest_args, profile)

    coverage = _load_coverage_module()
    cov = coverage.Coverage(
        data_file=str(data_file),
        source=[str(REPO_ROOT / path) for path in profile.source_paths],
    )

    exit_code = 0
    if args.skip_tests:
        cov.load()
    else:
        cov.erase()
        run_cmd = [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--timid",
            "--data-file",
            str(data_file),
            "--source",
            ",".join(profile.source_paths),
            "-m",
            "pytest",
        ] + list(pytest_args)
        exit_code = subprocess.run(run_cmd, cwd=str(REPO_ROOT), check=False).returncode
        cov.load()

    target_paths = _resolve_target_paths(profile)
    files = _collect_file_coverage(cov, target_paths, args.low_symbol_percent)
    baseline, file_tolerance = _load_baseline(args.baseline_file)
    tolerance = args.baseline_tolerance if args.baseline_tolerance is not None else (file_tolerance or 0.0)
    baseline_checks = _compare_baseline(files, baseline, tolerance)

    markdown_path = output_dir / f"{profile.name}-coverage-audit.md"
    json_path = output_dir / f"{profile.name}-coverage-audit.json"
    _write_markdown(
        markdown_path,
        profile,
        files,
        args.engine_label,
        pytest_args,
        args.low_symbol_percent,
        args.max_symbols,
        baseline_checks,
    )
    _write_json(json_path, profile, files, args.engine_label, pytest_args, exit_code, baseline_checks)

    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    if exit_code:
        print(f"pytest failed with exit code {exit_code}", file=sys.stderr)
    failing_baseline_checks = [check for check in baseline_checks if check.status == "fail"]
    if failing_baseline_checks:
        print("per-file coverage baseline failed", file=sys.stderr)
        for check in failing_baseline_checks:
            actual = "missing" if check.actual_percent is None else f"{check.actual_percent:.2f}%"
            delta = "n/a" if check.delta_percent is None else f"{check.delta_percent:+.2f}%"
            print(
                "  {path}: actual={actual} floor={floor:.2f}% tolerance={tolerance:.2f}% delta={delta} reason={reason}".format(
                    path=check.path,
                    actual=actual,
                    floor=check.min_percent,
                    tolerance=check.tolerance_percent,
                    delta=delta,
                    reason=check.reason,
                ),
                file=sys.stderr,
            )
        return exit_code or 3
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
