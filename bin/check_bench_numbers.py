#!/usr/bin/env python3
"""Gate published benchmark numbers on the single source-of-truth.

Runs the same data validation as the ``gfql_bench`` Sphinx extension, without
Sphinx and without re-running any benchmark, plus two checks a docs build cannot
make:

1. **Commit drift** — how many commits touching the query engine have landed
   since a published run was measured. This is the check that catches "the board
   moved and nobody re-measured": a docs build is perfectly happy to render a
   number taken on a six-PRs-ago tree.
2. **Hand-typed literals** — the managed pages may not contain a bare latency or
   speedup literal. A number typed by hand is by construction not a reference to
   the source-of-truth, which is how a figure becomes authoritative and then gets
   re-copied.

Exit status is non-zero on any finding, so it can gate CI and pre-commit.

Usage::

    bin/check_bench_numbers.py                 # full gate
    bin/check_bench_numbers.py --no-git        # skip the commit-drift check
    bin/check_bench_numbers.py --today 2027-01-01   # what will break, and when
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_SOURCE = os.path.join(REPO_ROOT, 'docs', 'source')
sys.path.insert(0, os.path.join(DOCS_SOURCE, '_ext'))

from gfql_bench_data import (  # noqa: E402
    BenchData,
    BenchDataError,
    load_bench_data,
)

#: Paths whose churn invalidates a published query-latency number.
ENGINE_PATHS = (
    'graphistry/compute',
    'graphistry/Engine.py',
    'graphistry/models/gfql',
)

_ROLE_RE = re.compile(r':(bench|bench-diag):`([^`]+)`')
_PROVENANCE_RE = re.compile(r'^\s*\.\.\s+bench-provenance::\s*(\S+)\s*$', re.MULTILINE)
_DISCLOSURES_RE = re.compile(r'^\s*\.\.\s+bench-disclosures::\s*$', re.MULTILINE)

#: A latency literal: "68 ms", "13.83s", "1.20s".
_LATENCY_RE = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?\s*(?:ms|µs|us|s)(?![\w])')
#: A speedup literal: "38x", "9.4×", "~56×".
_SPEEDUP_RE = re.compile(r'(?<![\w.])\d[\d,]*(?:\.\d+)?\s*[x×](?![\w])')


class Finding:
    def __init__(self, where: str, message: str) -> None:
        self.where = where
        self.message = message

    def render(self) -> str:
        return '{}: {}'.format(self.where, self.message)


def _read(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as handle:
        return handle.read()


def _git(args: Sequence[str]) -> Optional[str]:
    try:
        completed = subprocess.run(
            ['git'] + list(args),
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.decode('utf-8', 'replace')


def check_reference_integrity(data: BenchData) -> List[Finding]:
    """Every ``:bench:`` reference resolves, and its page carries the caveats."""
    findings: List[Finding] = []
    for rel in data.policy.managed_docs:
        path = os.path.join(DOCS_SOURCE, rel)
        if not os.path.isfile(path):
            findings.append(Finding(rel, 'managed doc is listed in policy.managed_docs but does not exist'))
            continue
        text = _read(path)
        keys = [match.group(2) for match in _ROLE_RE.finditer(text)]
        declared_runs = set(_PROVENANCE_RE.findall(text))
        has_disclosures = bool(_DISCLOSURES_RE.search(text))

        needed_runs: List[str] = []
        needs_disclosure = False
        for key in keys:
            if key not in data.cells:
                findings.append(Finding(rel, 'references unknown benchmark key {!r}'.format(key)))
                continue
            cell = data.cells[key]
            if cell.run_id not in needed_runs:
                needed_runs.append(cell.run_id)
            if cell.disclosures:
                needs_disclosure = True
        for run_id in needed_runs:
            if run_id not in declared_runs:
                findings.append(Finding(
                    rel, 'publishes run {!r} without a `.. bench-provenance:: {}` block'.format(run_id, run_id)))
        if needs_disclosure and not has_disclosures:
            findings.append(Finding(
                rel, 'publishes a disclosure-bearing number but has no `.. bench-disclosures::` block'))
    return findings


def check_freshness(data: BenchData, today: datetime.date) -> List[Finding]:
    findings: List[Finding] = []
    for run, age in data.stale_runs(today):
        findings.append(Finding(
            'runs.{}'.format(run.run_id),
            'measured {} ({} days ago); policy.max_age_days is {}. Re-measure or drop the claim.'.format(
                run.measured_at.isoformat(), age, data.policy.max_age_days)))
    return findings


def check_commit_drift(data: BenchData) -> List[Finding]:
    """Fail when the query engine has moved materially since a run was measured."""
    findings: List[Finding] = []
    limit = data.policy.max_compute_commit_drift
    for run_id in sorted(data.runs):
        run = data.runs[run_id]
        exists = _git(['cat-file', '-e', run.pygraphistry_commit + '^{commit}'])
        if exists is None:
            findings.append(Finding(
                'runs.{}'.format(run_id),
                'pygraphistry_commit {} is not in this repository, so the run cannot be placed in '
                'history and its numbers cannot be validated'.format(run.pygraphistry_commit)))
            continue
        log = _git(['log', '--oneline', '{}..HEAD'.format(run.pygraphistry_commit), '--'] + list(ENGINE_PATHS))
        if log is None:
            findings.append(Finding(
                'runs.{}'.format(run_id),
                'could not compute commit drift from {}'.format(run.pygraphistry_commit)))
            continue
        drift = len([line for line in log.splitlines() if line.strip()])
        if drift > limit:
            findings.append(Finding(
                'runs.{}'.format(run_id),
                '{} commits touching {} have landed since {} was measured (policy max {}). '
                'The published numbers describe a tree that no longer exists.'.format(
                    drift, '/'.join(ENGINE_PATHS), run.pygraphistry_commit, limit)))
    return findings


def _strip_literal_directives(text: str) -> str:
    """Blank out regions where a literal is not a published claim.

    Code blocks, shell recipes and inline literals hold reproduction commands and
    API names, not board numbers.
    """
    literal_start = re.compile(r'^\s*\.\.\s+(?:code-block|parsed-literal|literalinclude|math)::')
    literal_paragraph = re.compile(r'(?<!\.)::\s*$')
    out: List[str] = []
    in_block = False
    block_indent = 0
    for line in text.split('\n'):
        stripped = line.strip()
        if in_block:
            indent = len(line) - len(line.lstrip())
            if stripped and indent <= block_indent:
                in_block = False
            else:
                out.append('')
                continue
        if literal_start.match(line) or (literal_paragraph.search(line) and not stripped.startswith('..')):
            in_block = True
            block_indent = len(line) - len(line.lstrip())
            out.append('')
            continue
        out.append(re.sub(r'``[^`]*``', '', line))
    return '\n'.join(out)


def check_hand_typed_literals(data: BenchData) -> List[Finding]:
    findings: List[Finding] = []
    for rel in data.policy.managed_docs:
        path = os.path.join(DOCS_SOURCE, rel)
        if not os.path.isfile(path):
            continue
        allowed = data.policy.allowed_literals(rel)
        text = _strip_literal_directives(_read(path))
        for lineno, line in enumerate(text.split('\n'), start=1):
            without_roles = _ROLE_RE.sub('', line)
            for pattern in (_LATENCY_RE, _SPEEDUP_RE):
                for match in pattern.finditer(without_roles):
                    literal = match.group(0).strip()
                    if literal in allowed:
                        continue
                    findings.append(Finding(
                        '{}:{}'.format(rel, lineno),
                        'hand-typed benchmark literal {!r}. Publish it as :bench:`<key>` from the '
                        'source-of-truth, or add it to policy.literal_allowlist if it is not a '
                        'measured claim.'.format(literal)))
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=None, help='path to gfql_benchmarks.json')
    parser.add_argument('--today', default=None, help='YYYY-MM-DD; override the clock for freshness checks')
    parser.add_argument('--no-git', action='store_true', help='skip the commit-drift check')
    args = parser.parse_args(argv)

    try:
        data = load_bench_data(args.data)
    except BenchDataError as exc:
        sys.stderr.write('FAIL benchmark source-of-truth is invalid: {}\n'.format(exc))
        return 2

    if args.today:
        year, month, day = args.today.split('-')
        today = datetime.date(int(year), int(month), int(day))
    else:
        today = datetime.date.today()

    checks: List[Tuple[str, List[Finding]]] = [
        ('reference integrity', check_reference_integrity(data)),
        ('freshness', check_freshness(data, today)),
        ('hand-typed literals', check_hand_typed_literals(data)),
    ]
    if not args.no_git:
        checks.append(('commit drift', check_commit_drift(data)))

    total = 0
    for name, findings in checks:
        if findings:
            total += len(findings)
            sys.stderr.write('\nFAIL {} ({} finding(s)):\n'.format(name, len(findings)))
            for finding in findings:
                sys.stderr.write('  - {}\n'.format(finding.render()))
        else:
            sys.stdout.write('ok   {}\n'.format(name))

    if total:
        sys.stderr.write(
            '\n{} published benchmark number(s) cannot be defended from {}.\n'.format(
                total, os.path.relpath(data.source_path, REPO_ROOT)))
        return 1
    sys.stdout.write('\nAll published benchmark numbers trace to {} run(s) with full provenance.\n'.format(
        len(data.runs)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
