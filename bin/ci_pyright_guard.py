#!/usr/bin/env python3
"""CI guardrail that holds pyright's reproducible findings to a per-file ratchet.

Runs `bin/pyright.sh --outputjson` over `graphistry/` and compares the result to
`bin/ci_pyright_baseline.json`. A file may not gain findings relative to the
baseline, and a file absent from the baseline must have none. Existing debt is
grandfathered; new and moved code is held to the rule.

Only the rules in RATCHETED_RULES gate, and the criterion is deliberately narrow:
a rule qualifies when the source file's own control flow, names and syntax decide
it, with no type consulted. Everything else reads third-party stubs, so its verdict
moves with the interpreter and with whichever optional dependencies happen to be
installed -- `reportAttributeAccessIssue` ranges from 146 to 810 findings on this
same tree depending only on the environment. Gating on those would fail for a
developer who has cudf installed, pass in CI, and drift on every pandas release.
They are still collected and printed by --report; they just never gate.

The criterion is a principle rather than an experiment on purpose. Measuring
agreement across environments is necessary but not sufficient: three type-dependent
rules (reportOptionalSubscript, reportTypedDictNotRequiredAccess, reportIndexIssue)
produced identical counts across four environments and then diverged on the fifth,
simply because they are rare enough to agree by luck. Corroboration: the rules below
have identical per-file counts on python 3.8 / 3.11 / 3.12 / 3.14 and on a
workstation carrying polars, cudf, scipy and scikit-learn.

  ./bin/ci_pyright_guard.py                   # check (this is what CI runs)
  ./bin/ci_pyright_guard.py --report          # totals per rule, always exit 0
  ./bin/ci_pyright_guard.py --list RULE       # every current finding for RULE
  ./bin/ci_pyright_guard.py --update-baseline
  ./bin/ci_pyright_guard.py --strict          # also fail when the baseline is
                                              # looser than reality
  ./bin/ci_pyright_guard.py --from-json FILE  # reuse a saved pyright run

Escape hatch: pyright's own `# pyright: ignore[<rule>]` on the reported line.
Suppressed findings never reach this guard.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_BASELINE = os.path.join(REPO_ROOT, "bin", "ci_pyright_baseline.json")
PYRIGHT_SH = os.path.join(REPO_ROOT, "bin", "pyright.sh")

# Rules decided by the source file alone, with no type consulted. See the module
# docstring for why the bar is a principle and not merely measured agreement.
RATCHETED_RULES = (
    "reportPossiblyUnboundVariable",   # control flow within the function
    "reportSelfClsParameterName",      # syntax
    "reportUndefinedVariable",         # name resolution within the module
    "reportUnsupportedDunderAll",      # name resolution within the module
    "reportUnusedExpression",          # syntax
)

CountsByFile = Dict[str, int]
Counts = Dict[str, CountsByFile]


class Finding(object):
    __slots__ = ("rule", "path", "line", "message")

    def __init__(self, rule: str, path: str, line: int, message: str) -> None:
        self.rule = rule
        self.path = path
        self.line = line
        self.message = message


def rel(path: str) -> str:
    return os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")


def run_pyright() -> dict:
    """Invoke the pinned pyright and return its parsed JSON report."""
    proc = subprocess.run(
        [PYRIGHT_SH, "--outputjson"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=None,
        universal_newlines=True,
    )
    # pyright exits non-zero whenever it reports anything, which is the normal
    # state of a ratcheted tree, so the exit code says nothing; empty output does.
    if not proc.stdout.strip():
        raise SystemExit(
            "bin/ci_pyright_guard.py: bin/pyright.sh produced no JSON (exit %d)." % proc.returncode
        )
    return json.loads(proc.stdout)


def parse_report(report: dict) -> Tuple[Counts, Dict[str, List[Finding]], Dict[str, int]]:
    """Split a pyright report into ratcheted counts, their findings, and the rest."""
    counts = dict((rule, {}) for rule in RATCHETED_RULES)  # type: Counts
    by_rule = dict((rule, []) for rule in RATCHETED_RULES)  # type: Dict[str, List[Finding]]
    ungated = {}  # type: Dict[str, int]
    for diagnostic in report.get("generalDiagnostics", []):
        rule = diagnostic.get("rule", "<no-rule>")
        path = rel(diagnostic["file"])
        if rule not in counts:
            ungated[rule] = ungated.get(rule, 0) + 1
            continue
        line = diagnostic.get("range", {}).get("start", {}).get("line", 0) + 1
        message = diagnostic.get("message", "").splitlines()[0]
        counts[rule][path] = counts[rule].get(path, 0) + 1
        by_rule[rule].append(Finding(rule, path, line, message))
    return counts, by_rule, ungated


def load_baseline(path: str) -> Counts:
    if not os.path.exists(path):
        return dict((rule, {}) for rule in RATCHETED_RULES)
    with open(path, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    rules = raw.get("rules", {})
    return dict((rule, dict(rules.get(rule, {}))) for rule in RATCHETED_RULES)


def write_baseline(path: str, counts: Counts, version: str) -> None:
    payload = {
        "_comment": (
            "Per-file ratchet for bin/ci_pyright_guard.py, built with pyright %s. Counts "
            "may shrink, never grow; a file absent here must have zero findings. Only the "
            "rules listed are gated -- the rest move with the installed dependencies. "
            "Regenerate with `./bin/ci_pyright_guard.py --update-baseline` and explain the "
            "delta in the PR description." % version
        ),
        "pyright_version": version,
        "rules": dict((rule, dict(sorted(counts.get(rule, {}).items()))) for rule in RATCHETED_RULES),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--update-baseline", action="store_true")
    parser.add_argument("--report", action="store_true", help="print totals, always exit 0")
    parser.add_argument("--list", dest="list_rule", metavar="RULE", default=None,
                        help="print every current finding for RULE")
    parser.add_argument("--strict", action="store_true",
                        help="also fail when the baseline is looser than reality")
    parser.add_argument("--from-json", dest="from_json", default=None,
                        help="read a saved `pyright --outputjson` report instead of running it")
    args = parser.parse_args(argv)

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as handle:
            report = json.load(handle)
    else:
        report = run_pyright()
    counts, by_rule, ungated = parse_report(report)
    version = str(report.get("version", "unknown"))

    if args.list_rule is not None:
        if args.list_rule not in RATCHETED_RULES:
            parser.error("unknown rule %r; pick one of: %s"
                         % (args.list_rule, ", ".join(RATCHETED_RULES)))
        rows = sorted(by_rule[args.list_rule], key=lambda f: (f.path, f.line))
        for finding in rows:
            print("%s:%d: %s" % (finding.path, finding.line, finding.message))
        print("-- %d finding(s) for %s" % (len(rows), args.list_rule))
        return 0

    if args.report:
        print("pyright report (graphistry/, tests excluded), pyright %s" % version)
        print("  gated:")
        for rule in RATCHETED_RULES:
            print("    %-36s %5d finding(s) across %3d file(s)"
                  % (rule, sum(counts[rule].values()), len(counts[rule])))
        print("  not gated (environment-dependent, informational only):")
        for rule in sorted(ungated, key=lambda r: (-ungated[r], r)):
            print("    %-36s %5d finding(s)" % (rule, ungated[rule]))
        return 0

    if args.update_baseline:
        write_baseline(args.baseline, counts, version)
        print("wrote %s (pyright %s)" % (rel(args.baseline), version))
        for rule in RATCHETED_RULES:
            print("  %-36s %5d" % (rule, sum(counts[rule].values())))
        return 0

    baseline = load_baseline(args.baseline)
    regressions = []  # type: List[str]
    slack = []  # type: List[str]
    for rule in RATCHETED_RULES:
        current = counts[rule]
        allowed = baseline[rule]
        for path in sorted(set(current) | set(allowed)):
            now = current.get(path, 0)
            cap = allowed.get(path, 0)
            if now > cap:
                regressions.append("%s: %s has %d finding(s); baseline allows %d"
                                   % (rule, path, now, cap))
                for finding in sorted(by_rule[rule], key=lambda f: f.line):
                    if finding.path == path:
                        regressions.append("    %s:%d: %s"
                                           % (finding.path, finding.line, finding.message))
            elif now < cap:
                slack.append("%s: %s now %d (baseline %d)" % (rule, path, now, cap))

    if regressions:
        print("Pyright guard FAILED - findings above the committed baseline:\n")
        for line in regressions:
            print("  " + line)
        print("\nFix the finding, or - if pyright is wrong here - annotate that line with")
        print("`# pyright: ignore[<rule>]`. Raising a cap via --update-baseline is not the")
        print("intended remedy; see DEVELOP.md \"Pyright ratchet\".")
        return 1

    total = sum(sum(counts[rule].values()) for rule in RATCHETED_RULES)
    print("Pyright guard OK (%d grandfathered finding(s), no growth)." % total)
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
