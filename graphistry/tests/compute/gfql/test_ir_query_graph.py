"""Tests for QueryGraph extraction from BoundIR.

Covers:
  - Empty BoundIR → empty QueryGraph
  - Single MATCH scope → one ConnectedComponent with correct aliases
  - Two parts sharing alias → merged into one component (connected)
  - Two parts with no shared aliases → two distinct components
  - WITH/RETURN boundary → aliases crossing it land in boundary_aliases
  - OPTIONAL arm variables → OptionalArm with correct nullable_aliases
  - Optional join_aliases (aliases shared between optional and required parts)
  - Edge variable classification into edge_aliases
"""
from __future__ import annotations

from graphistry.compute.gfql.ir.bound_ir import (
    BoundIR,
    BoundQueryPart,
    BoundVariable,
    SemanticTable,
)
from graphistry.compute.gfql.ir.query_graph import (
    QueryGraph,
    extract_query_graph,
)
from graphistry.compute.gfql.ir.types import EdgeRef, NodeRef


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _var(name: str, nullable: bool = False, null_extended_from: frozenset[str] = frozenset()) -> BoundVariable:
    return BoundVariable(
        name=name,
        logical_type=NodeRef(),
        nullable=nullable,
        null_extended_from=null_extended_from,
        entity_kind="node",
    )


def _edge_var(name: str) -> BoundVariable:
    return BoundVariable(
        name=name,
        logical_type=EdgeRef(),
        nullable=False,
        null_extended_from=frozenset(),
        entity_kind="edge",
    )


def _part(clause: str, inputs: frozenset[str] = frozenset(), outputs: frozenset[str] = frozenset()) -> BoundQueryPart:
    return BoundQueryPart(clause=clause, inputs=inputs, outputs=outputs)


def _ir(parts: list[BoundQueryPart], variables: dict[str, BoundVariable] | None = None) -> BoundIR:
    table = SemanticTable(variables=variables or {})
    return BoundIR(query_parts=parts, semantic_table=table)


# ---------------------------------------------------------------------------
# Empty / trivial cases
# ---------------------------------------------------------------------------

class TestExtractEmpty:
    def test_empty_bound_ir_gives_empty_querygraph(self) -> None:
        qg = extract_query_graph(_ir([]))
        assert qg.components == []
        assert qg.boundary_aliases == {}
        assert qg.optional_arms == []

    def test_single_match_no_aliases(self) -> None:
        qg = extract_query_graph(_ir([_part("match")]))
        assert len(qg.components) == 1
        assert qg.components[0].node_aliases == []
        assert qg.components[0].edge_aliases == []

    def test_returns_querygraph_instance(self) -> None:
        qg = extract_query_graph(_ir([]))
        assert isinstance(qg, QueryGraph)


# ---------------------------------------------------------------------------
# Connected components
# ---------------------------------------------------------------------------

class TestConnectedComponents:
    def test_single_match_part_one_component(self) -> None:
        part = _part("match", outputs=frozenset({"a", "b"}))
        qg = extract_query_graph(_ir([part], {"a": _var("a"), "b": _var("b")}))
        assert len(qg.components) == 1
        assert set(qg.components[0].node_aliases) == {"a", "b"}

    def test_two_parts_sharing_alias_one_component(self) -> None:
        # (a)-->(b) and (b)-->(c) share "b" → one connected component
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        p2 = _part("match", outputs=frozenset({"b", "c"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c")}
        qg = extract_query_graph(_ir([p1, p2], vars_))
        assert len(qg.components) == 1
        assert set(qg.components[0].node_aliases) == {"a", "b", "c"}

    def test_two_parts_no_shared_alias_two_components(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        p2 = _part("match", outputs=frozenset({"x", "y"}))
        vars_ = {v: _var(v) for v in ("a", "b", "x", "y")}
        qg = extract_query_graph(_ir([p1, p2], vars_))
        assert len(qg.components) == 2
        alias_sets = [set(c.node_aliases) for c in qg.components]
        assert {"a", "b"} in alias_sets
        assert {"x", "y"} in alias_sets

    def test_three_parts_chain_connected(self) -> None:
        # a-b, b-c, c-d → all connected via shared aliases
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        p2 = _part("match", outputs=frozenset({"b", "c"}))
        p3 = _part("match", outputs=frozenset({"c", "d"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c", "d")}
        qg = extract_query_graph(_ir([p1, p2, p3], vars_))
        assert len(qg.components) == 1
        assert set(qg.components[0].node_aliases) == {"a", "b", "c", "d"}

    def test_with_clause_separates_scopes(self) -> None:
        # MATCH (a)-->(b) WITH b MATCH (c)-->(d) → two separate scopes
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pw = _part("with", inputs=frozenset({"b"}), outputs=frozenset({"b"}))
        p2 = _part("match", outputs=frozenset({"c", "d"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c", "d")}
        qg = extract_query_graph(_ir([p1, pw, p2], vars_))
        # a,b are in one scope; c,d in another — should be separate components
        assert len(qg.components) == 2


# ---------------------------------------------------------------------------
# Boundary aliases
# ---------------------------------------------------------------------------

class TestBoundaryAliases:
    def test_with_input_becomes_boundary_alias(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pw = _part("with", inputs=frozenset({"b"}), outputs=frozenset({"b"}))
        p2 = _part("match", outputs=frozenset({"c"}))
        vars_ = {"a": _var("a"), "b": _var("b"), "c": _var("c")}
        qg = extract_query_graph(_ir([p1, pw, p2], vars_))
        assert "b" in qg.boundary_aliases

    def test_non_projected_alias_not_boundary(self) -> None:
        # "a" is not in WITH inputs → not a boundary alias
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pw = _part("with", inputs=frozenset({"b"}), outputs=frozenset({"b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, pw], vars_))
        assert "a" not in qg.boundary_aliases

    def test_no_with_clause_no_boundary_aliases(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1], vars_))
        assert qg.boundary_aliases == {}

    def test_boundary_alias_has_logical_type(self) -> None:
        p1 = _part("match", outputs=frozenset({"n"}))
        pw = _part("with", inputs=frozenset({"n"}), outputs=frozenset({"n"}))
        vars_ = {"n": _var("n")}
        qg = extract_query_graph(_ir([p1, pw], vars_))
        assert "n" in qg.boundary_aliases
        assert isinstance(qg.boundary_aliases["n"], NodeRef)


# ---------------------------------------------------------------------------
# Optional arms
# ---------------------------------------------------------------------------

class TestOptionalArms:
    def test_no_optional_vars_no_arms(self) -> None:
        p1 = _part("match", outputs=frozenset({"a"}))
        qg = extract_query_graph(_ir([p1], {"a": _var("a")}))
        assert qg.optional_arms == []

    def test_optional_var_produces_arm(self) -> None:
        nullable_var = _var("b", nullable=True, null_extended_from=frozenset({"opt_0"}))
        required_var = _var("a")
        p1 = _part("match", outputs=frozenset({"a"}))
        p2 = _part("optional_match", outputs=frozenset({"b"}))
        qg = extract_query_graph(_ir([p1, p2], {"a": required_var, "b": nullable_var}))
        assert len(qg.optional_arms) == 1
        assert "b" in qg.optional_arms[0].nullable_aliases

    def test_optional_arm_id_matches_null_extended_from(self) -> None:
        nullable_var = _var("b", nullable=True, null_extended_from=frozenset({"arm_42"}))
        qg = extract_query_graph(_ir(
            [_part("optional_match", outputs=frozenset({"b"}))],
            {"b": nullable_var},
        ))
        assert qg.optional_arms[0].arm_id == "arm_42"

    def test_two_optional_arms_distinct(self) -> None:
        v1 = _var("b", nullable=True, null_extended_from=frozenset({"arm_0"}))
        v2 = _var("c", nullable=True, null_extended_from=frozenset({"arm_1"}))
        qg = extract_query_graph(_ir([], {"b": v1, "c": v2}))
        assert len(qg.optional_arms) == 2
        arm_ids = {a.arm_id for a in qg.optional_arms}
        assert arm_ids == {"arm_0", "arm_1"}

    def test_join_aliases_shared_between_optional_and_required(self) -> None:
        # "a" is required and also appears as input to optional arm → join alias
        required = _var("a")
        nullable = _var("b", nullable=True, null_extended_from=frozenset({"arm_0"}))
        p1 = _part("match", outputs=frozenset({"a"}))
        p2 = _part("optional_match", inputs=frozenset({"a"}), outputs=frozenset({"a", "b"}))
        qg = extract_query_graph(_ir([p1, p2], {"a": required, "b": nullable}))
        arm = qg.optional_arms[0]
        assert "a" in arm.join_aliases

    def test_multi_var_same_arm(self) -> None:
        v1 = _var("b", nullable=True, null_extended_from=frozenset({"arm_0"}))
        v2 = _var("c", nullable=True, null_extended_from=frozenset({"arm_0"}))
        qg = extract_query_graph(_ir([], {"b": v1, "c": v2}))
        assert len(qg.optional_arms) == 1
        assert {"b", "c"} == qg.optional_arms[0].nullable_aliases


# ---------------------------------------------------------------------------
# Edge aliases
# ---------------------------------------------------------------------------

class TestEdgeAliases:
    def test_edge_var_goes_to_edge_aliases(self) -> None:
        part = _part("match", outputs=frozenset({"n", "r", "m"}))
        vars_ = {"n": _var("n"), "r": _edge_var("r"), "m": _var("m")}
        qg = extract_query_graph(_ir([part], vars_))
        assert len(qg.components) == 1
        comp = qg.components[0]
        assert set(comp.edge_aliases) == {"r"}
        assert set(comp.node_aliases) == {"n", "m"}

    def test_alias_without_semantic_table_entry_goes_to_node_aliases(self) -> None:
        # Alias present in part.outputs but absent from semantic_table → defaults to node_aliases
        part = _part("match", outputs=frozenset({"x"}))
        qg = extract_query_graph(_ir([part], {}))
        comp = qg.components[0]
        assert "x" in comp.node_aliases
        assert comp.edge_aliases == []


# ---------------------------------------------------------------------------
# Scope boundary edge cases
# ---------------------------------------------------------------------------

class TestScopeBoundaries:
    def test_return_clause_splits_scope(self) -> None:
        # RETURN acts as scope boundary the same way WITH does
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pr = _part("return", inputs=frozenset({"a"}))
        p2 = _part("match", outputs=frozenset({"c"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c")}
        qg = extract_query_graph(_ir([p1, pr, p2], vars_))
        assert len(qg.components) == 2

    def test_with_multiple_inputs_all_become_boundary_aliases(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b", "c"}))
        pw = _part("with", inputs=frozenset({"a", "b"}), outputs=frozenset({"a", "b"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c")}
        qg = extract_query_graph(_ir([p1, pw], vars_))
        assert "a" in qg.boundary_aliases
        assert "b" in qg.boundary_aliases
        assert "c" not in qg.boundary_aliases

    def test_with_input_missing_from_semantic_table_skipped_gracefully(self) -> None:
        # The WITH references an alias that was never bound → no crash, no boundary entry
        pw = _part("with", inputs=frozenset({"ghost"}), outputs=frozenset())
        qg = extract_query_graph(_ir([pw], {}))
        assert "ghost" not in qg.boundary_aliases
