"""``gfql_clear_caches()`` must account for EVERY process-lifetime GFQL cache.

Not "clear every cache" -- some must survive, and that is the point. The rule is that no
memo may be *unaccounted for*: each one is either emptied by ``gfql_clear_caches`` or listed
in ``EXEMPT`` below with a written reason. Adding an ``@lru_cache`` to the GFQL tree and
saying nothing fails this file.

Why the enumeration is the test rather than a checklist in a docstring: a stale cache that
nobody remembers is a *correctness* bug (results become order-dependent, so a test that
passes alone fails in a suite) and a *measurement* bug. It produced a real published error --
a "cold-process" benchmark arm was reported as costing 2.3-10.2 ms of query compilation when
the Cypher AST memo was in fact never being emptied, because the clear targeted
``parse_cypher`` while the ``lru_cache`` sits on the ``_parse_cypher_cached`` body it
delegates to. The old ``getattr(obj, "cache_clear", None)`` lookup skipped the miss in
silence. Nothing failed; the number was simply wrong for days.

STATIC + FUNCTIONAL: the AST scan catches a *new* cache, and the runtime assertions catch a
clear that stops working.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set

import pytest

GFQL_TREE = Path(__file__).resolve().parents[3] / "compute" / "gfql"
GFQL_UNIFIED = Path(__file__).resolve().parents[3] / "compute" / "gfql_unified.py"


# Emptied by gfql_clear_caches(). These are keyed by CALLER INPUT, so they grow with traffic
# and their contents change what a later call costs.
CLEARED: Set[str] = {
    "_parse_cypher_cached",   # cypher query text -> frozen AST      (maxsize=512)
    "_parse_expr_cached",     # row-expression text -> ExprNode      (maxsize=1024)
}

# Deliberately NOT emptied. Every entry is a maxsize=1 singleton that is a pure function of
# the CODE, not of any input: it cannot grow, and rebuilding it costs strictly more than the
# work it saves. Emptying these would make a "cold" measurement include one-time process
# setup that no caller can ever pay twice.
EXEMPT: Dict[str, str] = {
    "_parser_lalr":
        "Lark LALR(1) tables for the whole-query grammar; function of the grammar, built "
        "once per process, and far dearer than any single parse it serves",
    "_pattern_parser":
        "Lark LALR(1) tables for the pattern-fragment start rule; same reasoning",
    "_where_predicate_chain_parser":
        "Lark LALR(1) tables for the flat WHERE-chain start rule; same reasoning",
    "_parser":
        "Lark LALR(1) tables for the row-expression grammar; same reasoning",
    "_ast_builder_class":
        "returns the row-expression Transformer CLASS, a function of the code; it is cached "
        "because @dataclass re-execs generated __init__/__eq__ source on every rebuild, which "
        "was 40% of GFQL compile time. Instances are still created per parse, so nothing "
        "stateful is shared",
    "_where_rows_expr_parser_fn":
        "binds imported callables and returns None when lark is absent; a resolved import "
        "is not stale state, and re-resolving it cannot change the answer",
    "_gfql_expr_runtime_parser_bundle":
        "same: an import-resolution bundle, None on a minimal install",
    "_gfql_cudf_list_sort_requires_host_bridge":
        "probes the installed cuDF version for a segfaulting list-sort; the installed "
        "version cannot change inside a process",
    "_ddl_prefix_re":
        "a compiled regex over a module-level pattern constant",
    "_ddl_res":
        "compiled regexes over module-level pattern constants",
}


def _lru_cached_functions() -> Dict[str, Path]:
    """Every ``@lru_cache``/``@cache``-decorated def in the GFQL tree, name -> file."""
    found: Dict[str, Path] = {}
    files = sorted(GFQL_TREE.rglob("*.py")) + [GFQL_UNIFIED]
    for path in files:
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                name = (
                    target.attr if isinstance(target, ast.Attribute)
                    else target.id if isinstance(target, ast.Name)
                    else ""
                )
                if name in ("lru_cache", "cache"):
                    found[node.name] = path
    return found


def test_every_gfql_cache_is_either_cleared_or_exempted_with_a_reason() -> None:
    """THE LOCK. A new memo in the GFQL tree must be classified, in writing, right here."""
    discovered = _lru_cached_functions()
    assert discovered, "the AST scan found no caches at all -- the scan itself is broken"

    unaccounted = sorted(set(discovered) - CLEARED - set(EXEMPT))
    assert not unaccounted, (
        "these @lru_cache functions are neither cleared by gfql_clear_caches() nor exempted:\n"
        + "\n".join(f"  {n}  ({discovered[n]})" for n in unaccounted)
        + "\n\nDecide which it is. If it is keyed by caller input, clear it and add it to "
        "CLEARED. If it is a process-lifetime singleton that is a function of the code, add "
        "it to EXEMPT with the reason. Do not leave it silent: an unaccounted cache makes "
        "results order-dependent and makes 'cold' measurements wrong."
    )

    stale = sorted((CLEARED | set(EXEMPT)) - set(discovered))
    assert not stale, (
        f"these names are classified here but no longer exist as caches: {stale}. Delete the "
        "entries so the lists keep describing the code."
    )


def test_every_exemption_carries_a_real_reason() -> None:
    """An exemption with an empty or placeholder reason is just a silent cache again."""
    for name, reason in EXEMPT.items():
        assert len(reason.split()) >= 6, f"{name}: reason is too thin to be a reason"
        assert "TODO" not in reason.upper(), f"{name}: TODO is not a justification"


def test_clear_caches_actually_empties_the_cypher_ast_memo() -> None:
    """THE REGRESSION PIN for the published-number bug.

    ``parse_cypher`` has no ``cache_clear`` of its own -- the memo is on the
    ``_parse_cypher_cached`` body. Clearing the wrong name used to be a silent no-op.
    """
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    assert not hasattr(cypher_parser.parse_cypher, "cache_clear"), (
        "parse_cypher grew a cache_clear; the indirection this test guards has changed, so "
        "re-read gfql_clear_caches() before relaxing anything here"
    )

    cypher_parser.parse_cypher("MATCH (n) RETURN n")
    assert cypher_parser._parse_cypher_cached.cache_info().currsize > 0

    gfql_clear_caches()
    assert cypher_parser._parse_cypher_cached.cache_info().currsize == 0, (
        "gfql_clear_caches() left the Cypher AST memo populated -- every 'cold-process' "
        "measurement taken through it is invalid"
    )


def test_clear_caches_actually_empties_the_row_expression_memo() -> None:
    pytest.importorskip("lark")
    from graphistry.compute.gfql import expr_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    expr_parser.parse_expr("age > 30")
    assert expr_parser._parse_expr_cached.cache_info().currsize > 0

    gfql_clear_caches()
    assert expr_parser._parse_expr_cached.cache_info().currsize == 0


def test_clear_caches_leaves_the_lalr_tables_built() -> None:
    """The exemption is load-bearing, so pin it: rebuilding the LALR tables on every clear
    would put grammar construction inside any 'cold' number measured through this call."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    before = cypher_parser._parser_lalr()
    gfql_clear_caches()
    assert cypher_parser._parser_lalr() is before, (
        "the whole-query LALR parser was rebuilt by gfql_clear_caches()"
    )


def test_clear_caches_raises_rather_than_skipping_a_missing_target() -> None:
    """FAIL LOUD. If a clear target loses its ``cache_clear``, the call must break, not
    quietly do less than it says."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.cypher import parser as cypher_parser
    from graphistry.compute.gfql_unified import gfql_clear_caches

    original = cypher_parser._parse_cypher_cached

    class _NoClear:
        def __call__(self, query: str) -> object:
            raise AssertionError("not called")

    try:
        cypher_parser._parse_cypher_cached = _NoClear()  # type: ignore[assignment]
        with pytest.raises(AttributeError):
            gfql_clear_caches()
    finally:
        cypher_parser._parse_cypher_cached = original  # type: ignore[assignment]

    gfql_clear_caches()  # and it still works once the real target is back
