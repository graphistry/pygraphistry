"""Tests for QueryGraph extraction from BoundIR.

Covers:
  - Empty BoundIR → empty QueryGraph
  - Single MATCH scope → one ConnectedComponent with correct aliases
  - Two parts sharing alias → merged into one component (connected)
  - Two parts with no shared aliases → two distinct components
  - WITH boundary → aliases in WITH outputs become boundary_aliases
    (RETURN only splits scope, does not project aliases)
  - OPTIONAL arm variables → OptionalArm with correct nullable_aliases
  - Optional join_aliases (aliases shared between optional and required parts)
  - Edge variable classification into edge_aliases
"""
from __future__ import annotations

from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import (
    BoundIR,
    BoundQueryPart,
    BoundVariable,
    SemanticTable,
)
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.query_graph import (
    QueryGraph,
    extract_query_graph,
)
from graphistry.compute.gfql.ir.types import EdgeRef, NodeRef, ScalarType
from graphistry.compute.gfql.cypher.parser import parse_cypher


def _bind(cypher: str) -> BoundIR:
    return FrontendBinder().bind(parse_cypher(cypher), PlanContext())


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
    def test_with_output_becomes_boundary_alias(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pw = _part("with", inputs=frozenset({"b"}), outputs=frozenset({"b"}))
        p2 = _part("match", outputs=frozenset({"c"}))
        vars_ = {"a": _var("a"), "b": _var("b"), "c": _var("c")}
        qg = extract_query_graph(_ir([p1, pw, p2], vars_))
        assert "b" in qg.boundary_aliases

    def test_non_projected_alias_not_boundary(self) -> None:
        # "a" is not in WITH outputs → not a boundary alias
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

    def test_scalar_var_falls_to_node_aliases(self) -> None:
        # entity_kind="scalar" is not "edge" → falls to node_aliases (not rejected)
        scalar_var = BoundVariable(
            name="users", logical_type=ScalarType(),
            nullable=False, null_extended_from=frozenset(),
            entity_kind="scalar",
        )
        part = _part("match", outputs=frozenset({"users"}))
        qg = extract_query_graph(_ir([part], {"users": scalar_var}))
        assert "users" in qg.components[0].node_aliases
        assert qg.components[0].edge_aliases == []


# ---------------------------------------------------------------------------
# Scope boundary edge cases
# ---------------------------------------------------------------------------

class TestScopeBoundaries:
    def test_return_clause_splits_scope(self) -> None:
        # RETURN splits scope groups but does NOT project boundary aliases (it's terminal)
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pr = _part("return", inputs=frozenset({"a"}))
        p2 = _part("match", outputs=frozenset({"c"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c")}
        qg = extract_query_graph(_ir([p1, pr, p2], vars_))
        assert len(qg.components) == 2
        assert "a" not in qg.boundary_aliases  # RETURN is terminal, not a scope projection
        alias_sets = [set(c.node_aliases) for c in qg.components]
        assert {"a", "b"} in alias_sets
        assert {"c"} in alias_sets

    def test_return_as_final_clause_no_boundary_aliases(self) -> None:
        # RETURN at end of query: preceding scope still yields its components,
        # RETURN inputs never appear in boundary_aliases
        p1 = _part("match", outputs=frozenset({"a", "b"}))
        pr = _part("return", inputs=frozenset({"a", "b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, pr], vars_))
        assert len(qg.components) == 1
        assert set(qg.components[0].node_aliases) == {"a", "b"}
        assert qg.boundary_aliases == {}

    def test_with_multiple_inputs_all_become_boundary_aliases(self) -> None:
        p1 = _part("match", outputs=frozenset({"a", "b", "c"}))
        pw = _part("with", inputs=frozenset({"a", "b"}), outputs=frozenset({"a", "b"}))
        vars_ = {v: _var(v) for v in ("a", "b", "c")}
        qg = extract_query_graph(_ir([p1, pw], vars_))
        assert "a" in qg.boundary_aliases
        assert "b" in qg.boundary_aliases
        assert "c" not in qg.boundary_aliases

    def test_with_output_missing_from_semantic_table_skipped_gracefully(self) -> None:
        # WITH output alias absent from semantic_table → no crash, no boundary entry
        pw = _part("with", inputs=frozenset(), outputs=frozenset({"ghost"}))
        qg = extract_query_graph(_ir([pw], {}))
        assert "ghost" not in qg.boundary_aliases

    def test_consecutive_with_clauses_no_empty_scope_groups(self) -> None:
        # Two back-to-back WITH clauses: second sees empty current_scope (guard prevents
        # empty scope group). Result: 2 components, boundary_aliases set idempotently.
        p1 = _part("match", outputs=frozenset({"a"}))
        pw1 = _part("with", inputs=frozenset({"a"}), outputs=frozenset({"a"}))
        pw2 = _part("with", inputs=frozenset({"a"}), outputs=frozenset({"a"}))
        p2 = _part("match", outputs=frozenset({"b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, pw1, pw2, p2], vars_))
        assert len(qg.components) == 2
        alias_sets = [set(c.node_aliases) for c in qg.components]
        assert {"a"} in alias_sets
        assert {"b"} in alias_sets
        assert "a" in qg.boundary_aliases

    def test_unwind_does_not_split_scope(self) -> None:
        # UNWIND is not in _SCOPE_SPLIT_CLAUSES — stays in same scope as surrounding parts.
        # Observable contract: if UNWIND incorrectly split scope, its output aliases would
        # be silently dropped (the part would fall into neither scope_group).
        p1 = _part("match", outputs=frozenset({"a"}))
        pu = _part("unwind", outputs=frozenset({"x"}))
        p2 = _part("match", outputs=frozenset({"b"}))
        vars_ = {"a": _var("a"), "x": _var("x"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, pu, p2], vars_))
        assert len(qg.components) == 3  # all three parts in one scope, no alias overlap
        all_aliases = {a for c in qg.components for a in c.node_aliases + c.edge_aliases}
        assert all_aliases == {"a", "x", "b"}
        assert qg.boundary_aliases == {}

    def test_with_rename_output_alias_becomes_boundary_alias(self) -> None:
        # WITH a AS b: output alias "b" goes into boundary_aliases (not input "a").
        # Extractor keys off part.outputs, not part.inputs, so renames are handled correctly.
        p1 = _part("match", outputs=frozenset({"a"}))
        pw = _part("with", inputs=frozenset({"a"}), outputs=frozenset({"b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, pw], vars_))
        assert "b" in qg.boundary_aliases
        assert "a" not in qg.boundary_aliases


# ---------------------------------------------------------------------------
# Connectivity contract and mixed-scope edge cases
# ---------------------------------------------------------------------------

class TestConnectivityContract:
    def test_input_only_overlap_produces_separate_components(self) -> None:
        # BoundQueryPart connectivity contract: shared aliases must appear in outputs
        # of BOTH parts to be merged. Input-only overlap does not imply connectivity.
        p1 = _part("match", outputs=frozenset({"a"}))
        p2 = _part("match", inputs=frozenset({"a"}), outputs=frozenset({"b"}))
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p1, p2], vars_))
        assert len(qg.components) == 2
        alias_sets = [set(c.node_aliases) for c in qg.components]
        assert {"a"} in alias_sets
        assert {"b"} in alias_sets

    def test_empty_part_alongside_nonempty_creates_isolated_component(self) -> None:
        # Empty part (no outputs) produces its own bare component alongside populated ones
        p_nonempty = _part("match", outputs=frozenset({"a", "b"}))
        p_empty = _part("match", outputs=frozenset())
        vars_ = {"a": _var("a"), "b": _var("b")}
        qg = extract_query_graph(_ir([p_nonempty, p_empty], vars_))
        assert len(qg.components) == 2
        node_alias_sets = [set(c.node_aliases) for c in qg.components]
        assert {"a", "b"} in node_alias_sets
        assert set() in node_alias_sets

    def test_two_empty_parts_same_scope_produce_two_empty_components(self) -> None:
        # Each empty-output part produces its own bare component
        p1 = _part("match", outputs=frozenset())
        p2 = _part("match", outputs=frozenset())
        qg = extract_query_graph(_ir([p1, p2], {}))
        assert len(qg.components) == 2
        assert all(c.node_aliases == [] and c.edge_aliases == [] for c in qg.components)


# ---------------------------------------------------------------------------
# Optional arm edge cases
# ---------------------------------------------------------------------------

class TestOptionalArmEdgeCases:
    def test_optional_match_with_no_nullable_outputs_produces_no_arm(self) -> None:
        # optional_match whose outputs are all non-nullable → part_arm_ids empty → no arm
        non_nullable = _var("r", nullable=False, null_extended_from=frozenset())
        p1 = _part("match", outputs=frozenset({"a"}))
        p2 = _part("optional_match", outputs=frozenset({"r"}))
        qg = extract_query_graph(_ir([p1, p2], {"a": _var("a"), "r": non_nullable}))
        assert qg.optional_arms == []

    def test_variable_in_multiple_optional_arms(self) -> None:
        # A variable can be nullable from two independent optional arms
        b_var = _var("b", nullable=True, null_extended_from=frozenset({"arm_0", "arm_1"}))
        qg = extract_query_graph(_ir([], {"b": b_var}))
        assert len(qg.optional_arms) == 2
        arm_ids = {a.arm_id for a in qg.optional_arms}
        assert arm_ids == {"arm_0", "arm_1"}
        for arm in qg.optional_arms:
            assert "b" in arm.nullable_aliases

    def test_optional_match_input_missing_from_semantic_table_no_join_alias(self) -> None:
        # optional_match.inputs references an unbound alias → var lookup returns None,
        # alias is silently skipped (not added to join_aliases)
        p1 = _part("match", outputs=frozenset({"a"}))
        p2 = _part("optional_match", inputs=frozenset({"ghost"}), outputs=frozenset({"b"}))
        nullable_var = _var("b", nullable=True, null_extended_from=frozenset({"arm_0"}))
        qg = extract_query_graph(_ir([p1, p2], {"a": _var("a"), "b": nullable_var}))
        arm = qg.optional_arms[0]
        assert "ghost" not in arm.join_aliases


# ---------------------------------------------------------------------------
# Binder integration — real FrontendBinder output (clause strings are UPPERCASE)
# ---------------------------------------------------------------------------

class TestBinderIntegration:
    """End-to-end: FrontendBinder → extract_query_graph.

    These tests would have failed before the clause-normalization fix because
    the binder emits UPPERCASE clause tokens ("MATCH", "OPTIONAL MATCH", "WITH",
    "RETURN") while the extractor previously compared against lowercase literals.
    """

    def test_simple_match_return_one_component(self) -> None:
        # MATCH (a)-[r]->(b) RETURN b → 1 scope, 1 component containing a, r, b
        qg = extract_query_graph(_bind("MATCH (a)-[r]->(b) RETURN b"))
        assert len(qg.components) == 1
        all_aliases = set(qg.components[0].node_aliases) | set(qg.components[0].edge_aliases)
        assert "a" in all_aliases
        assert "b" in all_aliases
        assert "r" in qg.components[0].edge_aliases
        assert "r" not in qg.components[0].node_aliases
        assert qg.boundary_aliases == {}
        assert qg.optional_arms == []

    def test_with_boundary_splits_scope_and_sets_boundary_alias(self) -> None:
        # MATCH (a) WITH a MATCH (b) RETURN b → 2 scope groups; "a" in boundary_aliases
        qg = extract_query_graph(_bind("MATCH (a) WITH a MATCH (b) RETURN b"))
        assert len(qg.components) == 2
        assert "a" in qg.boundary_aliases

    def test_optional_match_produces_arm(self) -> None:
        # MATCH (a) OPTIONAL MATCH (a)-->(b) RETURN b → 1 optional arm
        qg = extract_query_graph(_bind("MATCH (a) OPTIONAL MATCH (a)-->(b) RETURN b"))
        assert len(qg.optional_arms) == 1

    def test_with_rename_output_alias_becomes_boundary_alias_binder(self) -> None:
        # WITH a AS b renames: outputs={"b"} → "b" in boundary_aliases, "a" not
        qg = extract_query_graph(_bind("MATCH (a) WITH a AS b MATCH (c) RETURN c"))
        assert "b" in qg.boundary_aliases
        assert "a" not in qg.boundary_aliases

    def test_combined_with_optional_rename(self) -> None:
        # WITH boundary + OPTIONAL MATCH arm + alias rename in one query
        qg = extract_query_graph(
            _bind("MATCH (a) WITH a AS x OPTIONAL MATCH (x)-->(b) RETURN x, b")
        )
        assert "x" in qg.boundary_aliases
        assert len(qg.optional_arms) >= 1
