#!/usr/bin/env python3
"""CI guardrail for the comment-encoding rules in `agents/skills/review/SKILL.md`.

Meaning belongs in a **name**, a **test/pin**, or the **structure** of the code.
Prose is the last resort. Every other rule on that stack already has a machine
gate (`bin/ci_type_hygiene_guard.py`, `bin/ci_cypher_surface_guard.py`, the
per-file coverage floors); comment discipline was the only one left to human
review, which is why it is the only one that kept reaching the owner.

Stdlib only (`tokenize` + `ast`), so it runs unchanged on every interpreter in
the `python-lint-types` matrix (3.8 - 3.14). Invoked from `bin/lint.sh`; see
DEVELOP.md "Comment density guard".

Checks:

  comment-block     a run of 2+ consecutive full-line `#` comments (3+ for a
                    Sphinx `#:` run, which is otherwise a prose loophole).
                    Multi-line narration of what the next block does: extract a
                    helper whose NAME states the rule instead.
  perf-claim        performance / complexity / benchmark vocabulary in a comment
                    or a docstring. Measurement lives in pyg-bench as a test
                    that fails loudly; prose is a claim nobody re-measures.
  issue-rationale   a standalone comment or a docstring citing an issue / PR
                    number as the explanation. The pin's test NAME carries the
                    rationale; an issue ref may stay only as a trailing tag on a
                    line of code.

`comment-block` is a form rule and reads `#` comments only. `perf-claim` and
`issue-rationale` are content rules -- the claim does not become admissible by
moving into a docstring -- so they read docstrings too.

Scope: `graphistry/` for every check, with `graphistry/tests/` excluded from
`comment-block` and `issue-rationale` (a test may explain its oracle) but
INCLUDED for `perf-claim` (perf assertions belong in pyg-bench regardless).

Enforcement is a **per-file count ratchet** against the committed baseline: a
file may not gain findings, and a file absent from the baseline must have zero.
Existing debt is grandfathered; new and moved code is held to the rule.

  ./bin/ci_comment_density_guard.py                  # check (this is what CI runs)
  ./bin/ci_comment_density_guard.py --report         # totals per check, always exit 0
  ./bin/ci_comment_density_guard.py --list CHECK     # every current finding for CHECK
  ./bin/ci_comment_density_guard.py --update-baseline
  ./bin/ci_comment_density_guard.py --strict         # also fail when the baseline has
                                                     # gone stale-loose (time to tighten)

Escape hatch: put `# guard-ok: <check> -- <reason>` on the reported line, or on
the line directly above a reported comment run. Suppressed findings are not
counted.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import re
import sys
import tokenize
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOT = os.path.join(REPO_ROOT, "graphistry")
DEFAULT_BASELINE = os.path.join(REPO_ROOT, "bin", "ci_comment_density_baseline.json")

EXCLUDE_DIRS = ("__pycache__",)
EXCLUDE_FILES = ("graph_vector_pb2.py", "_version.py", "versioneer.py")
TEST_DIR_PARTS = ("tests",)

SUPPRESS_MARKER = "guard-ok"

CHECKS = ("comment-block", "perf-claim", "issue-rationale")

# A measured claim belongs in pyg-bench whether or not the file stating it is a test.
CHECKS_SKIPPING_TESTS = frozenset(["comment-block", "issue-rationale"])

# Tool directives are machine-readable, so they never start, continue, or become prose.
TOOL_DIRECTIVE_PREFIXES = (
    "type:", "noqa", "pragma", SUPPRESS_MARKER, "hygiene-ok", "fmt:", "isort:",
    "mypy:", "flake8:", "ruff:", "pylint:", "nosec", "yapf:", "coding:", "-*-",
)

# Sphinx attribute docs legitimately wrap to two lines; three is prose either way.
SPHINX_PREFIX = "#:"
PLAIN_RUN_LIMIT = 1
SPHINX_RUN_LIMIT = 2

ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")

# Complexity notation is matched case-sensitively so `foo(` and `into(` cannot hit.
PERF_PATTERNS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"\bO\([^)]*\)"), "asymptotic-complexity notation"),
    (re.compile(r"\b(?:faster|slower|vectoriz\w*|measurably|cheap\w*|expensive\w*)\b",
                re.IGNORECASE), "performance vocabulary"),
    (re.compile(r"\bhot[\s-]path\b", re.IGNORECASE), "hot-path claim"),
)

# `regress`/`A/B` also name correctness concepts, so they count only next to perf vocabulary.
PERF_CONTEXTUAL_PATTERNS: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile(r"\bregress\w*\b", re.IGNORECASE), "performance-regression claim"),
    (re.compile(r"\bA/B\b"), "A/B-comparison claim"),
)
PERF_CONTEXT_RE = re.compile(
    r"\b(?:perf|performance|speed|speedup|latency|throughput|runtime|slow\w*|fast\w*|"
    r"cost|costly|overhead|\d+x|\d+\s*(?:ms|us|s)|measurably|benchmark\w*)\b",
    re.IGNORECASE,
)

# Naming pyg-bench points at where the measurement lives, so it is a pointer, not a claim.
BENCH_PATTERN = re.compile(r"\bbenchmark\w*\b", re.IGNORECASE)
BENCH_POINTER_RE = re.compile(r"pyg[-_]bench", re.IGNORECASE)


def perf_claim(text: str, pointer_scope: str) -> Optional[Tuple[str, str]]:
    """(label, matched-text) for the first performance claim in `text`, if any.

    The contextual gate reads `text` alone; the pyg-bench pointer exemption reads
    the whole surrounding prose unit.
    """
    for pattern, label in PERF_PATTERNS:
        match = pattern.search(text)
        if match is not None:
            return label, match.group(0)
    if PERF_CONTEXT_RE.search(text):
        for pattern, label in PERF_CONTEXTUAL_PATTERNS:
            match = pattern.search(text)
            if match is not None:
                return label, match.group(0)
    if not BENCH_POINTER_RE.search(pointer_scope):
        match = BENCH_PATTERN.search(text)
        if match is not None:
            return "benchmark claim", match.group(0)
    return None

ISSUE_RE = re.compile(r"#\d{3,}")

CountsByFile = Dict[str, int]
Counts = Dict[str, CountsByFile]


@dataclass(frozen=True)
class Finding:
    check: str
    path: str
    line: int
    message: str


@dataclass(frozen=True)
class Comment:
    line: int
    col: int
    text: str
    standalone: bool


def is_test_path(relpath: str) -> bool:
    return any(part in TEST_DIR_PARTS for part in relpath.split("/")[:-1])


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


def comment_body(text: str) -> str:
    """The prose of a comment token, with its `#`/`#:` marker stripped."""
    stripped = text.lstrip("#")
    return stripped[1:].strip() if stripped[:1] == ":" else stripped.strip()


def is_directive(comment: Comment) -> bool:
    body = comment.text.lstrip("#").strip().lower()
    return any(body.startswith(prefix) for prefix in TOOL_DIRECTIVE_PREFIXES)


def is_file_preamble(comment: Comment) -> bool:
    """Shebang and PEP 263 encoding lines, which are not prose."""
    if comment.line == 1 and comment.text.startswith("#!"):
        return True
    return comment.line <= 2 and bool(ENCODING_RE.search(comment.text))


def read_comments(source: str, path: str) -> List[Comment]:
    lines = source.splitlines()
    out: List[Comment] = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise SystemExit("ci_comment_density_guard: cannot tokenize %s: %s" % (rel(path), exc))
    for tok in tokens:
        if tok.type != tokenize.COMMENT:
            continue
        row, col = tok.start
        prefix = lines[row - 1][:col] if row - 1 < len(lines) else ""
        out.append(Comment(row, col, tok.string, prefix.strip() == ""))
    return out


def parse(source: str, path: str) -> ast.Module:
    try:
        return ast.parse(source, filename=path)
    except SyntaxError as exc:
        raise SystemExit("ci_comment_density_guard: cannot parse %s: %s" % (rel(path), exc))


def first_statement_line(tree: ast.Module) -> int:
    """Line of the module's first statement; comments above it are the header."""
    return min((node.lineno for node in tree.body), default=sys.maxsize)


DOCSTRING_OWNERS = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def docstring_spans(tree: ast.Module) -> List[Tuple[int, int]]:
    """(first, last) source lines of every module/class/function docstring."""
    spans: List[Tuple[int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, DOCSTRING_OWNERS):
            continue
        body = getattr(node, "body", None)
        if not body or not isinstance(body[0], ast.Expr):
            continue
        value = body[0].value
        if not (isinstance(value, ast.Constant) and isinstance(value.value, str)):
            continue
        spans.append((value.lineno, getattr(value, "end_lineno", value.lineno) or value.lineno))
    return sorted(spans)


def comment_runs(comments: Sequence[Comment], header_line: int) -> List[List[Comment]]:
    """Maximal groups of adjacent standalone comment lines, header excluded.

    Directives and shebang/encoding lines separate runs rather than joining them.
    """
    runs: List[List[Comment]] = []
    current: List[Comment] = []
    for comment in comments:
        eligible = (
            comment.standalone
            and comment.line > header_line
            and not is_directive(comment)
            and not is_file_preamble(comment)
        )
        if not eligible:
            if current:
                runs.append(current)
                current = []
            continue
        if current and comment.line != current[-1].line + 1:
            runs.append(current)
            current = []
        current.append(comment)
    if current:
        runs.append(current)
    return runs


def suppressed_lines(source: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for idx, line in enumerate(source.splitlines()):
        if "#" in line and SUPPRESS_MARKER in line.split("#", 1)[1]:
            out[idx + 1] = line.strip()
    return out


def analyze_file(path: str, relpath: str, checks: Sequence[str]) -> List[Finding]:
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    comments = read_comments(source, path)
    tree = parse(source, path)
    lines = source.splitlines()
    skip = suppressed_lines(source)
    findings: List[Finding] = []

    def emit(check: str, line: int, message: str, allow: Sequence[int] = ()) -> None:
        if line not in skip and not any(other in skip for other in allow):
            findings.append(Finding(check, relpath, line, message))

    if "comment-block" in checks:
        header_line = first_statement_line(tree)
        for run in comment_runs(comments, header_line):
            sphinx = all(c.text.startswith(SPHINX_PREFIX) for c in run)
            limit = SPHINX_RUN_LIMIT if sphinx else PLAIN_RUN_LIMIT
            if len(run) <= limit:
                continue
            start, end = run[0].line, run[-1].line
            if any(line in skip for line in range(start - 1, end + 1)):
                continue
            findings.append(Finding(
                "comment-block", relpath, start,
                "%d-line %scomment run (through line %d); a comment earns its place only as "
                "ONE line stating a constraint the code and tests cannot express -- extract a "
                "helper whose NAME states the rule, or write the test"
                % (len(run), "`#:` " if sphinx else "", end),
            ))

    for comment in comments:
        body = comment_body(comment.text)
        if not body:
            continue
        if "perf-claim" in checks and not is_directive(comment):
            claim = perf_claim(body, body)
            if claim is not None:
                emit("perf-claim", comment.line,
                     "%s %r in a comment; performance and complexity claims belong in "
                     "pyg-bench as a measured test that fails loudly, not in prose nobody "
                     "re-measures" % claim)
        if (
            "issue-rationale" in checks
            and comment.standalone
            and not is_directive(comment)
            and not is_file_preamble(comment)
        ):
            match = ISSUE_RE.search(comment.text)
            if match is not None:
                emit("issue-rationale", comment.line,
                     "standalone comment cites %s as the rationale; the pin's test NAME carries "
                     "it (`test_<contract>_<condition>`) -- an issue ref may stay only as a "
                     "trailing tag on a line of code" % match.group(0))

    for start, end in docstring_spans(tree):
        allow = (start, start - 1, end)
        context = "\n".join(lines[start - 1:end])
        claimed = False
        cited = False
        for lineno in range(start, min(end, len(lines)) + 1):
            text = lines[lineno - 1]
            if "perf-claim" in checks and not claimed:
                claim = perf_claim(text, context)
                if claim is not None:
                    emit("perf-claim", lineno,
                         "%s %r in a docstring; performance and complexity claims belong in "
                         "pyg-bench as a measured test that fails loudly, not in prose nobody "
                         "re-measures" % claim, allow)
                    claimed = True
            if "issue-rationale" in checks and not cited:
                match = ISSUE_RE.search(text)
                if match is not None:
                    emit("issue-rationale", lineno,
                         "docstring cites %s as the rationale; state the contract by NAME and let "
                         "the pin's test name (`test_<contract>_<condition>`) carry the issue"
                         % match.group(0), allow)
                    cited = True

    return findings


def collect(root: str) -> Tuple[Counts, Dict[str, List[Finding]]]:
    counts: Counts = dict((check, {}) for check in CHECKS)
    by_check: Dict[str, List[Finding]] = dict((check, []) for check in CHECKS)
    for path in iter_source_files(root):
        relpath = rel(path)
        checks = [
            check for check in CHECKS
            if not (is_test_path(relpath) and check in CHECKS_SKIPPING_TESTS)
        ]
        if not checks:
            continue
        for finding in analyze_file(path, relpath, checks):
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
            "Per-file ratchet for bin/ci_comment_density_guard.py. Counts may shrink, never "
            "grow; a file absent here must have zero findings. Regenerate with "
            "`./bin/ci_comment_density_guard.py --update-baseline` and explain the delta in "
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
        print("comment-encoding report (graphistry/; tests scanned for perf-claim only)")
        for check in CHECKS:
            print("  %-18s %5d finding(s) across %3d file(s)"
                  % (check, sum(counts[check].values()), len(counts[check])))
        return 0

    if args.update_baseline:
        write_baseline(args.baseline, counts)
        print("wrote %s" % rel(args.baseline))
        for check in CHECKS:
            print("  %-18s %5d" % (check, sum(counts[check].values())))
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
        print("Comment-encoding guard FAILED - findings above the committed baseline:\n")
        for line in regressions:
            print("  " + line)
        print("\nEncode the meaning in a name, a test, or the structure and delete the prose.")
        print("If the comment genuinely states a constraint nothing else can, cut it to ONE")
        print("line. If it is genuinely correct as written, annotate it with")
        print("`# guard-ok: <check> -- <reason>`. Raising a cap via --update-baseline is not")
        print("the intended remedy; see DEVELOP.md \"Comment density guard\".")
        return 1

    total = sum(sum(counts[check].values()) for check in CHECKS)
    print("Comment-encoding guard OK (%d grandfathered finding(s), no growth)." % total)
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
