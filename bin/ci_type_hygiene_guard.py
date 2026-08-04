#!/usr/bin/env python3
"""CI guardrail for the type-hygiene defect classes that keep coming back in review.

Runs over `graphistry/` (tests excluded, see EXCLUDE_DIRS) using only the stdlib,
so it works unchanged on every interpreter in the `python-lint-types` matrix
(3.8 - 3.14). Invoked from `bin/lint.sh`; see DEVELOP.md "Type hygiene guard".

Checks:

  missing-annotations   parameter / return annotation absent
                        (same ground as ruff ANN001/002/003/201/202/204/205/206)
  explicit-any          `Any` anywhere inside an annotation (superset of ruff ANN401)
  explicit-cast         `cast(...)` / `typing.cast(...)` call
  bare-generic          unsubscripted `list`/`dict`/`List`/`Dict`/... in an annotation
  plottable-setattr     `setattr()` onto a parameter annotated as a Plottable
  plottable-attr-write  `param.attr = ...` onto a parameter annotated as a Plottable
  vocab-str-param       closed-vocabulary parameter (table/kind/direction/how/mode/
                        engine) annotated as plain `str` rather than a `Literal[...]`

Enforcement is a **per-file count ratchet**: a file may not gain findings relative
to the committed baseline, and a file absent from the baseline must have zero.
Existing debt is grandfathered; new and moved code is held to the rule.

  ./bin/ci_type_hygiene_guard.py                  # check (this is what CI runs)
  ./bin/ci_type_hygiene_guard.py --report         # totals per check, always exit 0
  ./bin/ci_type_hygiene_guard.py --list CHECK     # every current finding for CHECK
  ./bin/ci_type_hygiene_guard.py --update-baseline
  ./bin/ci_type_hygiene_guard.py --strict         # also fail when the baseline has
                                                  # gone stale-loose (time to tighten)

Escape hatch: put `# hygiene-ok` on the reported line, ideally as
`# hygiene-ok: <check-id> -- <reason>`. Suppressed findings are not counted.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOT = os.path.join(REPO_ROOT, "graphistry")
DEFAULT_BASELINE = os.path.join(REPO_ROOT, "bin", "ci_type_hygiene_baseline.json")

# Tests are excluded to match `mypy.ini` (`exclude = ...|graphistry/tests`).
# Including them would add ~5000 missing-annotation findings from throwaway
# fixtures and drown the signal this guard exists to protect.
EXCLUDE_DIRS = ("__pycache__", "tests")
EXCLUDE_FILES = ("graph_vector_pb2.py", "_version.py", "versioneer.py")

SUPPRESS_MARKER = "hygiene-ok"

CHECKS = (
    "missing-annotations",
    "explicit-any",
    "explicit-cast",
    "bare-generic",
    "plottable-setattr",
    "plottable-attr-write",
    "vocab-str-param",
)

# Unsubscripted forms of these lose the element type. Abstract shapes
# (Callable/Iterable/Sequence/...) are deliberately excluded: bare `Callable` is
# idiomatic here and flagging it would be noise.
BARE_GENERIC_NAMES = frozenset([
    "list", "dict", "set", "frozenset", "tuple", "type",
    "List", "Dict", "Set", "FrozenSet", "Tuple", "Type",
    "DefaultDict", "Deque", "OrderedDict", "Counter",
])

# Annotations that mean "this argument is a caller-owned graph object".
PLOTTABLE_NAMES = frozenset(["Plottable", "PlotterBase", "Plotter"])

# Parameter names whose vocabulary this repo has already committed to as a
# `Literal` alias somewhere (e.g. `GraphEntityKind = Literal['nodes', 'edges']`
# in graphistry/models/compute/features.py). Deliberately tiny: this is the only
# slice of "should be a Literal" that is mechanically decidable at a precision
# worth a reviewer's time.
VOCAB_PARAM_NAMES = frozenset(["table", "kind", "direction", "how", "mode", "engine"])

# `self`/`cls` are never annotated (ruff dropped ANN101/ANN102 for this reason).
IMPLICIT_PARAMS = frozenset(["self", "cls"])

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
FuncDef = Union[ast.FunctionDef, ast.AsyncFunctionDef]

CountsByFile = Dict[str, int]
Counts = Dict[str, CountsByFile]


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    line: int
    message: str


def iter_source_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for name in sorted(filenames):
            if name.endswith(".py") and name not in EXCLUDE_FILES:
                out.append(os.path.join(dirpath, name))
    return sorted(out)


def rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def unwrap_annotation(node: Optional[ast.expr]) -> Optional[ast.expr]:
    """Resolve a quoted annotation / forward reference into real AST nodes."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        try:
            return ast.parse(node.value, mode="eval").body
        except SyntaxError:
            return None
    return node


def annotation_idents(node: Optional[ast.expr]) -> Set[str]:
    resolved = unwrap_annotation(node)
    if resolved is None:
        return set()
    idents: Set[str] = set()
    for sub in ast.walk(resolved):
        if isinstance(sub, ast.Name):
            idents.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            idents.add(sub.attr)
    return idents


def is_plain_str(node: Optional[ast.expr]) -> bool:
    resolved = unwrap_annotation(node)
    return isinstance(resolved, ast.Name) and resolved.id == "str"


def root_name(node: ast.expr) -> Optional[str]:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


def decorator_idents(func: FuncDef) -> Set[str]:
    idents: Set[str] = set()
    for dec in func.decorator_list:
        for sub in ast.walk(dec):
            if isinstance(sub, ast.Name):
                idents.add(sub.id)
            elif isinstance(sub, ast.Attribute):
                idents.add(sub.attr)
    return idents


def parameters(func: FuncDef) -> List[Tuple[str, Optional[ast.expr], int]]:
    """(name, annotation, lineno) for every parameter, star-args included."""
    args = func.args
    out: List[Tuple[str, Optional[ast.expr], int]] = []
    for group in (getattr(args, "posonlyargs", []), args.args, args.kwonlyargs):
        for arg in group:
            out.append((arg.arg, arg.annotation, arg.lineno))
    for extra in (args.vararg, args.kwarg):
        if extra is not None:
            out.append((extra.arg, extra.annotation, extra.lineno))
    return out


def subscript_bases(annotation: ast.expr) -> Set[int]:
    """id()s of nodes that are the base of a subscript, i.e. `List` in `List[int]`."""
    return set(
        id(sub.value) for sub in ast.walk(annotation) if isinstance(sub, ast.Subscript)
    )


def suppressed_lines(source: str) -> Set[int]:
    return set(
        idx + 1
        for idx, line in enumerate(source.splitlines())
        if "#" in line and SUPPRESS_MARKER in line.split("#", 1)[1]
    )


def nested_node_ids(func: FuncDef) -> Set[int]:
    """Nodes belonging to an inner def, so each def is judged on its own body."""
    nested: Set[int] = set()
    for sub in ast.walk(func):
        if isinstance(sub, FUNC_TYPES) and sub is not func:
            for inner in ast.walk(sub):
                nested.add(id(inner))
    return nested


def analyze_file(path: str) -> List[Finding]:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise SystemExit("ci_type_hygiene_guard: cannot parse %s: %s" % (rel(path), exc))

    relpath = rel(path)
    skip = suppressed_lines(source)
    findings: List[Finding] = []

    def emit(check: str, line: int, message: str) -> None:
        if line not in skip:
            findings.append(Finding(check, relpath, line, message))

    for node in ast.walk(tree):
        if not isinstance(node, FUNC_TYPES):
            continue
        func: FuncDef = node
        decorators = decorator_idents(func)
        params = parameters(func)

        # --- class 1: function contracts missing types --------------------
        if "overload" not in decorators:
            for name, ann, line in params:
                if ann is None and name not in IMPLICIT_PARAMS:
                    emit("missing-annotations", line,
                         "parameter `%s` of `%s()` has no type annotation" % (name, func.name))
            if func.returns is None:
                emit("missing-annotations", func.lineno,
                     "`%s()` has no return type annotation" % func.name)

        # --- classes 3a / 4 / 5: annotation shape -------------------------
        slots: List[Tuple[str, Optional[ast.expr], int]] = list(params)
        slots.append(("return", func.returns, func.lineno))
        for name, ann, line in slots:
            resolved = unwrap_annotation(ann)
            if resolved is not None:
                bases = subscript_bases(resolved)
                for sub in ast.walk(resolved):
                    if not isinstance(sub, (ast.Name, ast.Attribute)) or id(sub) in bases:
                        continue
                    ident = sub.id if isinstance(sub, ast.Name) else sub.attr
                    sub_line = getattr(sub, "lineno", line)
                    if ident == "Any":
                        emit("explicit-any", sub_line,
                             "`Any` in the annotation of `%s` in `%s()`; prefer engine-agnostic "
                             "SeriesT/DataFrameT plus a localized `# type: ignore`"
                             % (name, func.name))
                    elif ident in BARE_GENERIC_NAMES:
                        emit("bare-generic", sub_line,
                             "bare `%s` in the annotation of `%s` in `%s()`; parameterize it "
                             "as `%s[...]`" % (ident, name, func.name, ident))
            if name in VOCAB_PARAM_NAMES and is_plain_str(ann):
                emit("vocab-str-param", line,
                     "`%s: str` in `%s()` names a closed vocabulary; use a `Literal[...]` alias"
                     % (name, func.name))

        # --- class 2: writes onto a caller-owned Plottable ----------------
        plottable_params = set(
            name for name, ann, _line in params
            if name not in IMPLICIT_PARAMS and (annotation_idents(ann) & PLOTTABLE_NAMES)
        )
        if not plottable_params:
            continue
        nested = nested_node_ids(func)
        for sub in ast.walk(func):
            if id(sub) in nested:
                continue
            target: Optional[ast.expr] = None
            check: Optional[str] = None
            label = ""
            if (
                isinstance(sub, ast.Call)
                and isinstance(sub.func, ast.Name)
                and sub.func.id == "setattr"
                and sub.args
            ):
                target, check, label = sub.args[0], "plottable-setattr", "setattr()"
            elif isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    if isinstance(tgt, ast.Attribute):
                        target, check, label = tgt, "plottable-attr-write", "`.%s =`" % tgt.attr
                        break
            elif isinstance(sub, ast.AugAssign) and isinstance(sub.target, ast.Attribute):
                target = sub.target
                check = "plottable-attr-write"
                label = "`.%s =`" % sub.target.attr
            if target is None or check is None:
                continue
            if root_name(target) in plottable_params:
                emit(check, sub.lineno,
                     "%s writes onto `%s`, a caller-owned Plottable parameter of `%s()`; "
                     "return a new object or thread a per-execution cache (see issue #1825)"
                     % (label, root_name(target), func.name))

    # --- class 3b: cast() ------------------------------------------------
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        ident: Optional[str] = None
        if isinstance(node.func, ast.Name):
            ident = node.func.id
        elif isinstance(node.func, ast.Attribute):
            ident = node.func.attr
        if ident == "cast":
            emit("explicit-cast", node.lineno,
                 "`cast(...)`; prefer engine-agnostic SeriesT/DataFrameT plus a localized "
                 "`# type: ignore` over `Any` + call-site casts")

    return findings


def collect(root: str) -> Tuple[Counts, Dict[str, List[Finding]]]:
    counts: Counts = dict((check, {}) for check in CHECKS)
    by_check: Dict[str, List[Finding]] = dict((check, []) for check in CHECKS)
    for path in iter_source_files(root):
        for finding in analyze_file(path):
            bucket = counts[finding.check]
            bucket[finding.path] = bucket.get(finding.path, 0) + 1
            by_check[finding.check].append(finding)
    return counts, by_check


def load_baseline(path: str) -> Counts:
    if not os.path.exists(path):
        return dict((check, {}) for check in CHECKS)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    checks = raw.get("checks", {})
    return dict((check, dict(checks.get(check, {}))) for check in CHECKS)


def write_baseline(path: str, counts: Counts) -> None:
    payload = {
        "_comment": (
            "Per-file ratchet for bin/ci_type_hygiene_guard.py. Counts may shrink, never "
            "grow; a file absent here must have zero findings. Regenerate with "
            "`./bin/ci_type_hygiene_guard.py --update-baseline` and explain the delta in "
            "the PR description."
        ),
        "checks": dict((check, dict(sorted(counts[check].items()))) for check in CHECKS),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", action="store_true", help="print totals, always exit 0")
    parser.add_argument("--list", dest="list_check", metavar="CHECK", default=None,
                        help="print every current finding for CHECK")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when the baseline is looser than reality")
    args = parser.parse_args(argv)

    counts, by_check = collect(SCAN_ROOT)

    if args.list_check is not None:
        if args.list_check not in CHECKS:
            parser.error("unknown check %r; pick one of: %s"
                         % (args.list_check, ", ".join(CHECKS)))
        rows = sorted(by_check[args.list_check], key=lambda f: (f.path, f.line))
        for finding in rows:
            print("%s:%d: %s" % (finding.path, finding.line, finding.message))
        print("-- %d finding(s) for %s" % (len(rows), args.list_check))
        return 0

    if args.report:
        print("type-hygiene report (graphistry/, tests excluded)")
        for check in CHECKS:
            print("  %-22s %5d finding(s) across %3d file(s)"
                  % (check, sum(counts[check].values()), len(counts[check])))
        return 0

    if args.update_baseline:
        write_baseline(args.baseline, counts)
        print("wrote %s" % rel(args.baseline))
        for check in CHECKS:
            print("  %-22s %5d" % (check, sum(counts[check].values())))
        return 0

    baseline = load_baseline(args.baseline)
    regressions: List[str] = []
    slack: List[str] = []
    for check in CHECKS:
        current = counts[check]
        allowed = baseline[check]
        for path in sorted(set(current) | set(allowed)):
            now = current.get(path, 0)
            cap = allowed.get(path, 0)
            if now > cap:
                regressions.append("%s: %s has %d finding(s); baseline allows %d"
                                   % (check, path, now, cap))
                for finding in sorted(by_check[check], key=lambda f: f.line):
                    if finding.path == path:
                        regressions.append("    %s:%d: %s"
                                           % (path, finding.line, finding.message))
            elif now < cap:
                slack.append("%s: %s now %d (baseline %d)" % (check, path, now, cap))

    if regressions:
        print("Type-hygiene guard FAILED - findings above the committed baseline:\n")
        for line in regressions:
            print("  " + line)
        print("\nFix the finding, or - if it is genuinely correct - annotate that line with")
        print("`# hygiene-ok: <check> -- <reason>`. Raising a cap via --update-baseline is")
        print("not the intended remedy; see DEVELOP.md \"Type hygiene guard\".")
        return 1

    total = sum(sum(counts[check].values()) for check in CHECKS)
    print("Type-hygiene guard OK (%d grandfathered finding(s), no growth)." % total)
    if slack:
        print("%d file(s) now below baseline; run --update-baseline to lock the improvement."
              % len(slack))
        if args.strict:
            for line in slack:
                print("  " + line)
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
