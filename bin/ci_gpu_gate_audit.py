#!/usr/bin/env python3
"""Audit the cuDF test gates and keep the size of the unprotected surface visible.

No CI lane installs cudf or sets ``TEST_CUDF``, so every cuDF-gated test is
developer-local evidence only. This guard makes that gap loud rather than silent:

1. every cuDF gate must be attributable -- a ``reason=`` naming ``TEST_CUDF``, so
   ``pytest -rs`` names what was not run rather than reporting a bare ``s``;
2. every cuDF gate must read the flag from the environment, so a gate cannot
   quietly become a constant;
3. ``DEVELOP.md`` must carry the unprotected-receipts note exactly while no
   workflow sets ``TEST_CUDF`` -- wiring a real GPU lane retires the note, and
   deleting the note without wiring a lane fails.

The audit is static: it proves the gates are well formed and counts them. It does
not and cannot prove the gated assertions are true; only a GPU lane does that.
"""
import ast
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "graphistry" / "tests"
WORKFLOWS = REPO / ".github" / "workflows"
DEVELOP = REPO / "DEVELOP.md"

FLAG = "TEST_CUDF"
UNPROTECTED_NOTE = "no CI lane executes cuDF"


class Gate:
    def __init__(self, path: Path, lineno: int, source: str, reason: Optional[str]) -> None:
        self.path = path
        self.lineno = lineno
        self.source = source
        self.reason = reason

    def where(self) -> str:
        return f"{self.path.relative_to(REPO)}:{self.lineno}"


def _reason_of(call: ast.Call) -> Optional[str]:
    for kw in call.keywords:
        if kw.arg == "reason" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
            return kw.value.value
    return None


def _skip_message_of(call: ast.Call) -> Optional[str]:
    for arg in call.args:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value
    return _reason_of(call)


def _callee_name(call: ast.Call) -> str:
    node = call.func
    parts: List[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _gate_context(text: str, call: ast.Call, parents) -> str:
    """Source that decides the gate: the call, widened to the ``if`` that guards a bare skip."""
    segment = ast.get_source_segment(text, call) or ""
    node = call
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            return (ast.get_source_segment(text, node.test) or "") + "\n" + segment
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
            break
    return segment


def collect_gates(root: Path) -> Tuple[List[Gate], List[str]]:
    gates: List[Gate] = []
    parse_errors: List[str] = []
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if FLAG not in text:
            continue
        try:
            tree = ast.parse(text, filename=str(path))
        except SyntaxError as e:
            parse_errors.append(f"{path.relative_to(REPO)}: {e}")
            continue
        parents = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _callee_name(node)
            if not (name.endswith("skipif") or name.endswith("skip")):
                continue
            segment = ast.get_source_segment(text, node) or ""
            if FLAG not in segment:
                continue
            reason = _reason_of(node) if name.endswith("skipif") else _skip_message_of(node)
            gates.append(Gate(path, node.lineno, _gate_context(text, node, parents), reason))
    return gates, parse_errors


def workflows_setting_flag(root: Path) -> List[str]:
    if not root.is_dir():
        return []
    assignments = (f"{FLAG}:", f"{FLAG}=")
    return sorted(
        p.name for p in root.glob("*.yml")
        if any(a in p.read_text(encoding="utf-8") for a in assignments)
    )


def emit(line: str) -> None:
    print(line)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def main() -> int:
    gates, parse_errors = collect_gates(TESTS)
    wired = workflows_setting_flag(WORKFLOWS)
    develop = DEVELOP.read_text(encoding="utf-8")
    failures: List[str] = []

    failures.extend(f"test file does not parse: {e}" for e in parse_errors)

    if not gates:
        failures.append(
            f"found zero {FLAG} gates under {TESTS.relative_to(REPO)}; the gating convention moved "
            "and this audit is now blind -- update it"
        )

    for gate in gates:
        if not gate.reason or FLAG not in gate.reason:
            failures.append(
                f"{gate.where()}: cuDF gate has no reason naming {FLAG}, so `pytest -rs` cannot "
                f"attribute the skip: {gate.source.splitlines()[0]}"
            )
        if "environ" not in gate.source and "getenv" not in gate.source:
            failures.append(
                f"{gate.where()}: cuDF gate does not read {FLAG} from the environment, so it "
                f"cannot be turned on: {gate.source.splitlines()[0]}"
            )

    note_present = UNPROTECTED_NOTE in develop
    if wired and note_present:
        failures.append(
            f"workflow(s) {wired} now set {FLAG}; remove the '{UNPROTECTED_NOTE}' note from DEVELOP.md"
        )
    if not wired and not note_present:
        failures.append(
            f"no workflow sets {FLAG}, so cuDF receipts are unprotected; DEVELOP.md must say "
            f"'{UNPROTECTED_NOTE}'"
        )

    files = sorted({str(g.path.relative_to(REPO)) for g in gates})
    emit(f"cuDF gates: {len(gates)} across {len(files)} test files")
    emit(f"workflows setting {FLAG}: {wired or 'NONE -- these gates are never executed by CI'}")

    if failures:
        for f in failures:
            print(f"ERROR: {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
