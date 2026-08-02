"""The row-expression transformer CLASS is built once per process; instances are not shared.

Why this is worth a test rather than just a commit message: the expensive thing was never
parsing. Profiling q7 of the matched graph benchmark (31 cold compiles, 0.649 s total) put
**0.261 s — 40% of all compile time — in ``_build_transformer``**, which was re-creating two
``@dataclass(frozen=True)`` helpers and a Lark ``Transformer`` subclass on every
``_parse_expr_cached`` miss: 434 calls producing 868 dataclass creations, 0.120 s of it in
``builtins.exec`` because ``@dataclass`` generates ``__init__``/``__eq__`` as source and execs
it. The whole LALR(1) parse, by comparison, was 0.151 s.

So the invariant has two halves and both matter. Build the class **once** (or the cost returns),
and keep instantiating **per parse** (or a future stateful transformer silently starts sharing
state between unrelated queries). A test that only checked the first half would bless a
regression into the second.
"""

from __future__ import annotations

import pytest


def test_the_transformer_class_is_created_once_per_process() -> None:
    """The whole point: identical class object, so no dataclass rebuild."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.expr_parser import _ast_builder_class

    first = _ast_builder_class()
    assert _ast_builder_class() is first, (
        "the transformer class is being rebuilt; @dataclass will re-exec generated source on "
        "every row-expression parse, which was 40% of GFQL compile time"
    )


def test_each_parse_still_gets_its_own_transformer_instance() -> None:
    """The other half. Sharing one instance would be a bigger win and a worse idea: Lark
    transformers are only incidentally stateless here, and a shared instance would make that
    an unwritten requirement of every future method added to the class."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.expr_parser import _ast_builder_class, _build_transformer

    a = _build_transformer()
    b = _build_transformer()
    assert a is not b, "transformer instances are being shared between parses"
    assert type(a) is type(b) is _ast_builder_class()


def test_the_transformer_class_survives_a_cache_clear() -> None:
    """``gfql_clear_caches()`` must NOT drop it — it is a function of the code, not of any
    query, exactly like the Lark parser objects. Clearing it would put class construction back
    inside every 'cold' measurement, which is a cost no caller can pay twice."""
    pytest.importorskip("lark")
    from graphistry.compute.gfql.expr_parser import _ast_builder_class
    from graphistry.compute.gfql_unified import gfql_clear_caches

    before = _ast_builder_class()
    gfql_clear_caches()
    assert _ast_builder_class() is before


@pytest.mark.parametrize(
    "expr",
    [
        "age > 30",
        "toLower(name) = 'bob'",
        "a.x IS NULL",
        "a.x IS NOT NULL",
        "CASE WHEN age > 30 THEN 1 ELSE 0 END",
        "count(DISTINCT a.id)",
        "(age > 30) AND (score < 5.5)",
        "a.x IN [1, 2, 3]",
    ],
)
def test_parsing_is_unchanged_by_the_hoist(expr: str) -> None:
    """Parity. Hoisting a class definition must not move a single AST node.

    Also exercises the two hoisted dataclasses specifically: ``_CaseArm`` via CASE/WHEN and
    ``_FunctionArgs`` via a DISTINCT aggregate — they are closed over by the transformer, so a
    botched hoist would surface here rather than in a generic expression.
    """
    pytest.importorskip("lark")
    from graphistry.compute.gfql.expr_parser import parse_expr
    from graphistry.compute.gfql_unified import gfql_clear_caches

    first = repr(parse_expr(expr))
    gfql_clear_caches()
    assert repr(parse_expr(expr)) == first, (
        f"{expr!r} parses differently across a cache clear"
    )
