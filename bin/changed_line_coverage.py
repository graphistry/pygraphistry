#!/usr/bin/env python3
"""Changed-line coverage gate for Python PR hygiene.

This tool consumes an existing coverage.py data file and a git diff, then
reports coverage for changed executable lines only. It is intentionally about
PR hygiene: it does not judge historical uncovered code and it does not provide
dead-code deletion evidence.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set


REPO_ROOT = Path(__file__).resolve().parent.parent
ZERO_SHA = "0000000000000000000000000000000000000000"
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

DEFAULT_INCLUDE = ("graphistry/*.py", "graphistry/**/*.py")
DEFAULT_EXCLUDE = (
    "graphistry/tests/*.py",
    "graphistry/tests/**/*.py",
    "graphistry/_version.py",
)


@dataclass(frozen=True)
class ChangedFileCoverage:
    path: str
    changed_statement_lines: int
    covered_lines: int
    missing_lines: int
    coverage_percent: float
    missing_line_numbers: List[int]


@dataclass(frozen=True)
class ChangedLineReport:
    base_ref: str
    head_ref: str
    generated_at: str
    min_percent: float
    status: str
    changed_statement_lines: int
    covered_lines: int
    missing_lines: int
    coverage_percent: float
    files: List[ChangedFileCoverage]
    skipped_files: List[str]


class ChangedLineCoverageError(RuntimeError):
    """Changed-line coverage setup or runtime failure."""


def _load_coverage_module() -> Any:
    try:
        import coverage  # type: ignore
    except Exception as exc:
        raise ChangedLineCoverageError("coverage.py is required; install the test extra.") from exc
    return coverage


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _matches_any(path: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _is_eligible_path(path: str, includes: Sequence[str], excludes: Sequence[str]) -> bool:
    return _matches_any(path, includes) and not _matches_any(path, excludes)


def _parse_changed_lines(diff_text: str) -> Dict[str, Set[int]]:
    changed: Dict[str, Set[int]] = {}
    current_path: Optional[str] = None
    new_line: Optional[int] = None

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ "):
            value = raw_line[4:].strip()
            current_path = None if value == "/dev/null" else (value[2:] if value.startswith("b/") else value)
            if current_path is not None:
                changed.setdefault(current_path, set())
            new_line = None
            continue

        match = HUNK_RE.match(raw_line)
        if match:
            new_line = int(match.group(1))
            continue

        if current_path is None or new_line is None:
            continue

        if raw_line.startswith("+") and not raw_line.startswith("+++"):
            changed[current_path].add(new_line)
            new_line += 1
        elif raw_line.startswith("-") and not raw_line.startswith("---"):
            continue
        else:
            new_line += 1

    return {path: lines for path, lines in changed.items() if lines}


def _git_changed_lines(base_ref: str, head_ref: str) -> Dict[str, Set[int]]:
    if not base_ref or base_ref == ZERO_SHA:
        return {}
    cmd = [
        "git",
        "diff",
        "--unified=0",
        "--no-color",
        "--no-ext-diff",
        f"{base_ref}...{head_ref}",
        "--",
        "*.py",
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ChangedLineCoverageError(
            "git diff failed for changed-line coverage: " + (result.stderr.strip() or result.stdout.strip())
        )
    return _parse_changed_lines(result.stdout)


def _file_line_coverage(cov: Any, path: Path, changed_lines: Set[int]) -> Optional[ChangedFileCoverage]:
    if not path.exists():
        return None

    _, statements_raw, _, missing_raw, _ = cov.analysis2(str(path))
    statements = set(statements_raw)
    missing = set(missing_raw)
    changed_statement_lines = sorted(changed_lines & statements)
    if not changed_statement_lines:
        return None

    missing_changed = sorted(line for line in changed_statement_lines if line in missing)
    covered = len(changed_statement_lines) - len(missing_changed)
    percent = 100.0 * covered / len(changed_statement_lines)
    return ChangedFileCoverage(
        path=_repo_rel(path),
        changed_statement_lines=len(changed_statement_lines),
        covered_lines=covered,
        missing_lines=len(missing_changed),
        coverage_percent=round(percent, 2),
        missing_line_numbers=missing_changed,
    )


def _build_report(
    cov: Any,
    changed_lines: Mapping[str, Set[int]],
    base_ref: str,
    head_ref: str,
    min_percent: float,
    includes: Sequence[str],
    excludes: Sequence[str],
) -> ChangedLineReport:
    files: List[ChangedFileCoverage] = []
    skipped: List[str] = []

    for rel_path in sorted(changed_lines):
        if not _is_eligible_path(rel_path, includes, excludes):
            skipped.append(rel_path)
            continue
        item = _file_line_coverage(cov, REPO_ROOT / rel_path, changed_lines[rel_path])
        if item is None:
            skipped.append(rel_path)
        else:
            files.append(item)

    changed_statement_lines = sum(item.changed_statement_lines for item in files)
    covered_lines = sum(item.covered_lines for item in files)
    missing_lines = sum(item.missing_lines for item in files)
    coverage_percent = 100.0 * covered_lines / changed_statement_lines if changed_statement_lines else 100.0
    status = "pass" if coverage_percent >= min_percent else "fail"

    return ChangedLineReport(
        base_ref=base_ref,
        head_ref=head_ref,
        generated_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        min_percent=round(min_percent, 2),
        status=status,
        changed_statement_lines=changed_statement_lines,
        covered_lines=covered_lines,
        missing_lines=missing_lines,
        coverage_percent=round(coverage_percent, 2),
        files=sorted(files, key=lambda item: (item.coverage_percent, item.path)),
        skipped_files=skipped,
    )


def _format_lines(lines: Sequence[int], limit: int = 20) -> str:
    if not lines:
        return ""
    rendered = ", ".join(str(line) for line in lines[:limit])
    if len(lines) > limit:
        rendered += f", ... (+{len(lines) - limit})"
    return rendered


def _write_markdown(path: Path, report: ChangedLineReport) -> None:
    lines = [
        "# Changed-Line Coverage",
        "",
        f"- Generated: `{report.generated_at}`",
        f"- Base: `{report.base_ref}`",
        f"- Head: `{report.head_ref}`",
        f"- Threshold: `{report.min_percent:.2f}%`",
        f"- Status: `{report.status}`",
        f"- Changed executable lines: `{report.changed_statement_lines}`",
        f"- Coverage: `{report.covered_lines}/{report.changed_statement_lines} = {report.coverage_percent:.2f}%`",
        "",
        "Changed-line coverage is a PR hygiene gate. It ignores historical uncovered lines outside the diff.",
        "",
        "## Files",
        "",
    ]
    if report.files:
        lines.extend(
            [
                "| File | Changed stmt lines | Covered | Missing | Coverage | Missing changed lines |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for item in report.files:
            lines.append(
                "| {path} | {changed} | {covered} | {missing} | {percent:.2f}% | {lines} |".format(
                    path=item.path,
                    changed=item.changed_statement_lines,
                    covered=item.covered_lines,
                    missing=item.missing_lines,
                    percent=item.coverage_percent,
                    lines=_format_lines(item.missing_line_numbers) or "-",
                )
            )
    else:
        lines.append("No eligible changed executable package lines were found.")

    if report.skipped_files:
        lines.extend(["", "## Skipped Changed Python Files", ""])
        lines.extend(f"- `{path}`" for path in report.skipped_files[:100])
        if len(report.skipped_files) > 100:
            lines.append(f"- ... (+{len(report.skipped_files) - 100})")

    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_json(path: Path, report: ChangedLineReport) -> None:
    path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-file", required=True, help="coverage.py data file to analyze")
    parser.add_argument("--base-ref", required=True, help="git base ref/SHA for the PR diff")
    parser.add_argument("--head-ref", default="HEAD", help="git head ref/SHA for the PR diff")
    parser.add_argument("--min-percent", type=float, default=80.0, help="Minimum changed-line coverage percentage")
    parser.add_argument("--output-dir", default="build/changed-line-coverage", help="Directory for markdown/json reports")
    parser.add_argument("--include", action="append", default=None, help="Eligible repo-relative fnmatch pattern")
    parser.add_argument("--exclude", action="append", default=None, help="Excluded repo-relative fnmatch pattern")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    coverage = _load_coverage_module()
    data_file = (REPO_ROOT / args.data_file).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cov = coverage.Coverage(data_file=str(data_file))
    cov.load()
    changed_lines = _git_changed_lines(args.base_ref, args.head_ref)
    report = _build_report(
        cov=cov,
        changed_lines=changed_lines,
        base_ref=args.base_ref,
        head_ref=args.head_ref,
        min_percent=args.min_percent,
        includes=tuple(args.include or DEFAULT_INCLUDE),
        excludes=tuple(args.exclude or DEFAULT_EXCLUDE),
    )

    markdown_path = output_dir / "changed-line-coverage.md"
    json_path = output_dir / "changed-line-coverage.json"
    _write_markdown(markdown_path, report)
    _write_json(json_path, report)
    print(f"Wrote {markdown_path}")
    print(f"Wrote {json_path}")
    if report.status != "pass":
        print(
            f"changed-line coverage {report.coverage_percent:.2f}% is below {report.min_percent:.2f}%",
            file=sys.stderr,
        )
        return 3
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChangedLineCoverageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
