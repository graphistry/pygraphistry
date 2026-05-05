from __future__ import annotations

from graphistry.compute.chain import Chain
from graphistry.compute.gfql.cypher import lowering, projection_planning
from graphistry.compute.gfql.cypher.reentry import runtime as reentry_runtime


def test_issue_1301_projection_split_delegator_round_trip() -> None:
    expr = "person.name"
    lowered = lowering._split_qualified_name(expr, line=1, column=1)
    extracted = projection_planning._split_qualified_name(expr, line=1, column=1)
    assert lowered == extracted == ("person", "name")


def test_issue_1301_reentry_runtime_delegator_identity_path() -> None:
    compiled = lowering.CompiledCypherQuery(chain=Chain([], validate=False))

    lowered = lowering._map_terminal_reentry_query(compiled, transform=lambda q: q)
    extracted = reentry_runtime._map_terminal_reentry_query(compiled, transform=lambda q: q)

    assert lowered is compiled
    assert extracted is compiled


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


def test_issue_1301_reentry_compile_delegator_forwards_params(monkeypatch) -> None:
    captured = {}

    def _stub(query, *, params):
        captured["query"] = query
        captured["params"] = params
        return "compiled"

    monkeypatch.setattr(reentry_runtime, "_compile_bounded_reentry_query", _stub)
    query = object()
    params = {"limit": 5}
    out = lowering._compile_bounded_reentry_query(query, params=params)

    assert out == "compiled"
    assert captured["query"] is query
    assert captured["params"] == params
