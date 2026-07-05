"""Full-repo LALR==Earley grammar-parity differential (CI gate).

Scrapes every cypher-looking string literal from the graphistry package
(python ``ast`` over all ``.py`` files — tests, source, corpora) and runs each
through the PRODUCTION pipeline twice: once with the production LALR parser
and once with an Earley parser built from the SAME grammar. Asserts:

- every query both parsers accept produces a byte-identical AST;
- both parsers reject the same queries (rejection parity), with the same
  error class — EXCEPT the pinned deliberate language fixes (see
  ``DELIBERATE_LANGUAGE_FIXES`` in test_grammar_invariants.py), where
  Earley's dynamic lexer accepts a lexing accident that LALR correctly
  rejects.

This is the broad-corpus complement to the constructed corpus in
``test_grammar_invariants.py``: it sweeps everything the repo actually says,
so a grammar edit that shifts parser behavior on ANY in-repo query fails
here. The file name matches the ``test_differential*.py`` glob of the
``cypher-frontend-differential-parity`` CI job.
"""
from __future__ import annotations

import ast as pyast
import os
from typing import Any, List, Set, Tuple

import pytest

from graphistry.compute.gfql.cypher import parse_cypher
from graphistry.compute.gfql.cypher import parser as parser_mod

lark = pytest.importorskip("lark")

# Queries where LALR (correctly) rejects what Earley's dynamic lexer
# accepted by accident — the pinned language fixes. Loaded from the sibling
# invariants module by path (the tests tree is not a package).
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_tgi", os.path.join(os.path.dirname(__file__), "test_grammar_invariants.py")
)
assert _spec is not None and _spec.loader is not None
_tgi = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tgi)
DELIBERATE_LANGUAGE_FIXES = _tgi.DELIBERATE_LANGUAGE_FIXES

_PKG_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(parser_mod.__file__), "..", "..", "..")
)


def _looks_cypher(s: str) -> bool:
    if not (10 <= len(s) <= 4000) or "\x00" in s:
        return False
    return "RETURN" in s or "MATCH" in s


def _scrape_corpus() -> List[str]:
    corpus: Set[str] = set()
    for dirpath, dirnames, filenames in os.walk(_PKG_ROOT):
        dirnames[:] = [d for d in dirnames if d != "__pycache__"]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            try:
                with open(os.path.join(dirpath, fn), encoding="utf-8") as f:
                    tree = pyast.parse(f.read())
            except (SyntaxError, OSError):
                continue
            for node in pyast.walk(tree):
                if isinstance(node, pyast.Constant) and isinstance(node.value, str):
                    s = node.value.strip()
                    if _looks_cypher(s):
                        corpus.add(s)
    return sorted(corpus)


def _run_with(parser: Any, query: str, monkeypatch: pytest.MonkeyPatch) -> Tuple[str, str]:
    monkeypatch.setattr(parser_mod, "_parser_lalr", lambda: parser)
    parser_mod._parse_cypher_cached.cache_clear()
    try:
        return ("OK", repr(parse_cypher(query)))
    except Exception as exc:  # both syntax + validation rejections count
        return ("REJECT", type(exc).__name__)
    finally:
        monkeypatch.undo()
        parser_mod._parse_cypher_cached.cache_clear()


def test_full_repo_lalr_earley_differential(monkeypatch: pytest.MonkeyPatch) -> None:
    earley = lark.Lark(
        parser_mod._GRAMMAR,
        start="start",
        parser="earley",
        maybe_placeholders=False,
        propagate_positions=True,
    )
    lalr = parser_mod._parser_lalr()
    corpus = _scrape_corpus()
    assert len(corpus) > 500, "corpus scrape looks broken (too few queries)"

    known_fixes = set(DELIBERATE_LANGUAGE_FIXES)
    ast_diverge: List[str] = []
    lang_diverge: List[str] = []
    err_diverge: List[str] = []
    n_ok = n_rej = 0
    for query in corpus:
        lalr_status, lalr_val = _run_with(lalr, query, monkeypatch)
        earley_status, earley_val = _run_with(earley, query, monkeypatch)
        if lalr_status == earley_status == "OK":
            n_ok += 1
            if lalr_val != earley_val:
                ast_diverge.append(query)
        elif lalr_status == earley_status == "REJECT":
            n_rej += 1
            if lalr_val != earley_val:
                err_diverge.append(f"{query!r}: LALR={lalr_val} Earley={earley_val}")
        elif query in known_fixes:
            assert lalr_status == "REJECT", (
                f"pinned language fix should be LALR-rejected: {query!r}"
            )
        else:
            lang_diverge.append(f"{query!r}: LALR={lalr_status} Earley={earley_status}")

    assert not ast_diverge, (
        f"AST divergence on {len(ast_diverge)} queries, e.g.:\n"
        + "\n".join(repr(q) for q in ast_diverge[:10])
    )
    assert not lang_diverge, (
        "unpinned accept/reject divergence (add to DELIBERATE_LANGUAGE_FIXES "
        "only if deliberate):\n" + "\n".join(lang_diverge[:10])
    )
    assert not err_diverge, "error-class divergence:\n" + "\n".join(err_diverge[:10])
    assert n_ok > 500 and n_rej > 50, f"corpus balance unexpected: ok={n_ok} rej={n_rej}"
