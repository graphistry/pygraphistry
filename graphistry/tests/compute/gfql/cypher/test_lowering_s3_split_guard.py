from __future__ import annotations

import inspect

from graphistry.compute.gfql.cypher import lowering, projection_planning
from graphistry.compute.gfql.cypher.reentry import compiletime


def test_issue_1301_projection_split_delegator_round_trip() -> None:
    expr = "person.name"
    lowered = lowering._split_qualified_name(expr, line=1, column=1)
    extracted = projection_planning._split_qualified_name(expr, line=1, column=1)
    assert lowered == extracted == ("person", "name")

def test_issue_1301_projection_delegator_forwards_args(monkeypatch) -> None:
    captured = {}
    alias_obj = object()
    alias_targets = {"a": alias_obj}

    def _stub(expr, *, alias_targets, params, field, line, column):
        captured["args"] = (expr, alias_targets, params, field, line, column)
        return ("x", "y")

    monkeypatch.setattr(projection_planning, "_projection_ref_from_expr", _stub)
    out = lowering._projection_ref_from_expr(
        "a.b",
        alias_targets=alias_targets,
        params={"p": 1},
        field="return.item",
        line=7,
        column=11,
    )

    assert out == ("x", "y")
    assert captured["args"] == ("a.b", alias_targets, {"p": 1}, "return.item", 7, 11)


def test_issue_1471_reentry_compiletime_has_no_lowering_symbol_table_shim() -> None:
    source = inspect.getsource(compiletime)

    assert "globals().update(vars(" not in source
    assert not hasattr(compiletime, "_lowering")
    assert compiletime.__name__ == "graphistry.compute.gfql.cypher.reentry.compiletime"
    assert compiletime.__package__ == "graphistry.compute.gfql.cypher.reentry"


def test_issue_1471_reentry_compiletime_keeps_lowering_entrypoints() -> None:
    for name in (
        "_compile_bounded_reentry_query",
        "_drop_bare_alias_items_from_stage",
    ):
        helper = getattr(compiletime, name, None)
        assert callable(helper), name
