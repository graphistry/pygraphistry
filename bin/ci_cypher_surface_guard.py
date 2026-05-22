#!/usr/bin/env python3
"""CI guardrail for Cypher lowering size and CompiledCypher surface growth.

Tracks:
- `graphistry/compute/gfql/cypher/lowering.py` total line count
- Dataclass field + `@property` counts for:
  - `CompiledCypherQuery`
  - `CompiledGraphBinding`
  - `CompiledCypherGraphQuery`

Default mode compares current metrics against committed maxima baseline and
fails on growth. Use `--write-baseline` for intentional baseline updates.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


REPO_ROOT = Path(__file__).resolve().parent.parent
LOWERING_PATH = REPO_ROOT / "graphistry/compute/gfql/cypher/lowering.py"
DEFAULT_BASELINE_PATH = REPO_ROOT / "bin/ci_cypher_surface_guard_baseline.json"
TARGET_CLASSES = (
    "CompiledCypherQuery",
    "CompiledGraphBinding",
    "CompiledCypherGraphQuery",
)


@dataclass(frozen=True)
class ClassSurface:
    fields: int
    properties: int


@dataclass(frozen=True)
class Metrics:
    lowering_py_lines: int
    compiled_surfaces: Dict[str, ClassSurface]


class GuardError(RuntimeError):
    """Configuration or validation failure in metric collection/checking."""


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def _collect_class_surfaces(path: Path) -> Dict[str, ClassSurface]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    class_nodes = {
        node.name: node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name in TARGET_CLASSES
    }

    missing = [name for name in TARGET_CLASSES if name not in class_nodes]
    if missing:
        raise GuardError(f"Missing expected classes in lowering.py: {', '.join(missing)}")

    out: Dict[str, ClassSurface] = {}
    for class_name in TARGET_CLASSES:
        node = class_nodes[class_name]
        fields = 0
        properties = 0
        for member in node.body:
            if isinstance(member, ast.AnnAssign) and isinstance(member.target, ast.Name):
                fields += 1
            elif isinstance(member, ast.FunctionDef):
                if any(isinstance(dec, ast.Name) and dec.id == "property" for dec in member.decorator_list):
                    properties += 1
        out[class_name] = ClassSurface(fields=fields, properties=properties)

    return out


def collect_metrics() -> Metrics:
    if not LOWERING_PATH.exists():
        raise GuardError(f"Expected lowering file not found: {LOWERING_PATH}")

    return Metrics(
        lowering_py_lines=_count_lines(LOWERING_PATH),
        compiled_surfaces=_collect_class_surfaces(LOWERING_PATH),
    )


def _metrics_to_baseline(metrics: Metrics) -> dict:
    return {
        "lowering_py_max_lines": metrics.lowering_py_lines,
        "compiled_surfaces": {
            class_name: {
                "max_fields": surface.fields,
                "max_properties": surface.properties,
            }
            for class_name, surface in metrics.compiled_surfaces.items()
        },
    }


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise GuardError(
            f"Baseline file missing: {path}. Run with --write-baseline to create it intentionally."
        ) from exc
    except json.JSONDecodeError as exc:
        raise GuardError(f"Invalid JSON in baseline file {path}: {exc}") from exc


def _write_baseline(path: Path, metrics: Metrics) -> None:
    path.write_text(
        json.dumps(_metrics_to_baseline(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _markdown_report(metrics: Metrics, baseline: dict | None, violations: list[str]) -> str:
    lines = [
        "### Cypher Surface Guard",
        "",
        f"- `lowering.py` lines: `{metrics.lowering_py_lines}`",
    ]
    if baseline is not None:
        lines.append(f"- baseline max `lowering.py` lines: `{baseline.get('lowering_py_max_lines', 'n/a')}`")
    lines.extend(
        [
            "",
            "| Surface | Fields | Max Fields | Properties | Max Properties |",
            "|---|---:|---:|---:|---:|",
        ]
    )

    baseline_surfaces = baseline.get("compiled_surfaces", {}) if baseline is not None else {}
    for class_name in TARGET_CLASSES:
        metric = metrics.compiled_surfaces[class_name]
        base = baseline_surfaces.get(class_name, {}) if baseline is not None else {}
        lines.append(
            "| {name} | {fields} | {max_fields} | {props} | {max_props} |".format(
                name=class_name,
                fields=metric.fields,
                max_fields=base.get("max_fields", "n/a"),
                props=metric.properties,
                max_props=base.get("max_properties", "n/a"),
            )
        )

    lines.append("")
    if violations:
        lines.append("Guard result: ❌ fail")
        lines.append("")
        lines.extend(f"- {msg}" for msg in violations)
    else:
        lines.append("Guard result: ✅ pass")
    lines.append("")
    return "\n".join(lines)


def _check_against_baseline(metrics: Metrics, baseline: dict) -> list[str]:
    violations: list[str] = []

    max_lines = baseline.get("lowering_py_max_lines")
    if not isinstance(max_lines, int):
        raise GuardError("Baseline missing integer key: lowering_py_max_lines")
    if metrics.lowering_py_lines > max_lines:
        violations.append(
            f"`lowering.py` line count grew ({metrics.lowering_py_lines} > {max_lines})."
        )

    baseline_surfaces = baseline.get("compiled_surfaces")
    if not isinstance(baseline_surfaces, dict):
        raise GuardError("Baseline missing object key: compiled_surfaces")

    for class_name in TARGET_CLASSES:
        cls_base = baseline_surfaces.get(class_name)
        if not isinstance(cls_base, dict):
            raise GuardError(f"Baseline missing class entry: {class_name}")

        max_fields = cls_base.get("max_fields")
        max_properties = cls_base.get("max_properties")
        if not isinstance(max_fields, int) or not isinstance(max_properties, int):
            raise GuardError(
                f"Baseline class entry {class_name} must contain integer max_fields/max_properties"
            )

        current = metrics.compiled_surfaces[class_name]
        if current.fields > max_fields:
            violations.append(
                f"{class_name} field count grew ({current.fields} > {max_fields})."
            )
        if current.properties > max_properties:
            violations.append(
                f"{class_name} property count grew ({current.properties} > {max_properties})."
            )

    return violations


def _append_summary(markdown: str) -> None:
    summary_target = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_target:
        return
    summary_file = Path(summary_target)
    try:
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("a", encoding="utf-8") as f:
            f.write(markdown)
            if not markdown.endswith("\n"):
                f.write("\n")
    except OSError:
        # Never fail the guard solely due to summary write issues.
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE_PATH,
        help=f"Baseline JSON path (default: {DEFAULT_BASELINE_PATH})",
    )
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="Write current metrics as baseline and exit 0.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        metrics = collect_metrics()

        if args.write_baseline:
            _write_baseline(args.baseline, metrics)
            print(f"Wrote baseline: {args.baseline}")
            print(json.dumps(_metrics_to_baseline(metrics), indent=2, sort_keys=True))
            return 0

        baseline = _load_json(args.baseline)
        violations = _check_against_baseline(metrics, baseline)
        report = _markdown_report(metrics, baseline, violations)

        print(report)
        _append_summary(report)

        if violations:
            print(
                "\nIf growth is intentional, regenerate baseline and include rationale in PR:\n"
                f"  python {Path(__file__).as_posix()} --write-baseline\n"
            )
            return 1

        return 0
    except GuardError as exc:
        print(f"cypher-surface-guard error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
