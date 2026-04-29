from __future__ import annotations

from typing import cast

import pytest

from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError, GFQLValidationError
from graphistry.compute.gfql.cypher import (
    CallClause,
    CypherGraphQuery,
    CypherQuery,
    ExpressionText,
    LabelRef,
    NodePattern,
    ParameterRef,
    PropertyRef,
    RelationshipPattern,
    CypherUnionQuery,
    WherePredicate,
    WherePatternPredicate,
    parse_cypher,
)
from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.ast import GraphBinding, GraphConstructor, UseClause


def _parse_query(query: str) -> CypherQuery:
    return cast(CypherQuery, parse_cypher(query))


def test_parse_minimal_match_return() -> None:
    parsed = _parse_query("MATCH (n) RETURN n")

    assert parsed.match is not None
    assert parsed.match.pattern[0].variable == "n"
    assert parsed.unwinds == ()
    assert parsed.return_.distinct is False
    assert parsed.return_.kind == "return"
    assert parsed.return_.items[0].expression.text == "n"
    assert parsed.return_.items[0].alias is None
    assert parsed.order_by is None
    assert parsed.skip is None
    assert parsed.limit is None
    assert parsed.trailing_semicolon is False


def test_parse_graph_property_access_allowed() -> None:
    parsed = _parse_query("MATCH (n) RETURN n.graph")

    assert parsed.return_.items[0].expression.text == "n.graph"


def test_parse_rejects_reserved_graph_identifier_for_aliases_and_variables() -> None:
    with pytest.raises(GFQLSyntaxError):
        _parse_query("MATCH (graph) RETURN n")

    with pytest.raises(GFQLSyntaxError):
        _parse_query("MATCH (n) RETURN n.id AS graph")


def test_parse_standalone_graph_constructor() -> None:
    parsed = parse_cypher("GRAPH { MATCH (a)-[r]->(b) }")

    assert isinstance(parsed, CypherGraphQuery)
    assert len(parsed.constructor.matches) == 1
    assert parsed.constructor.where is None
    assert parsed.constructor.use is None
    assert parsed.graph_bindings == ()


def test_parse_standalone_graph_constructor_with_where() -> None:
    parsed = parse_cypher("GRAPH { MATCH (a)-[r]->(b) WHERE a.id = 'x' }")

    assert isinstance(parsed, CypherGraphQuery)
    assert parsed.constructor.where is not None


def test_parse_graph_binding() -> None:
    parsed = parse_cypher(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
        "USE g1 MATCH (x) RETURN x"
    )

    assert isinstance(parsed, CypherQuery)
    assert len(parsed.graph_bindings) == 1
    assert parsed.graph_bindings[0].name == "g1"
    assert parsed.use is not None
    assert parsed.use.ref == "g1"


def test_parse_multi_graph_binding() -> None:
    parsed = parse_cypher(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
        "GRAPH g2 = GRAPH { MATCH (c)-[s]->(d) } "
        "USE g2 MATCH (x) RETURN x"
    )

    assert isinstance(parsed, CypherQuery)
    assert len(parsed.graph_bindings) == 2
    assert parsed.graph_bindings[0].name == "g1"
    assert parsed.graph_bindings[1].name == "g2"


def test_parse_graph_constructor_with_use_inside() -> None:
    parsed = parse_cypher(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
        "GRAPH { USE g1 MATCH (c)-[s]->(d) }"
    )

    assert isinstance(parsed, CypherGraphQuery)
    assert len(parsed.graph_bindings) == 1
    assert parsed.constructor.use is not None
    assert parsed.constructor.use.ref == "g1"


def test_parse_graph_constructor_with_call() -> None:
    parsed = parse_cypher("GRAPH { CALL graphistry.degree.write() }")

    assert isinstance(parsed, CypherGraphQuery)
    assert parsed.constructor.call is not None
    assert parsed.constructor.call.procedure == "graphistry.degree.write"
    assert parsed.constructor.matches == ()


def test_parse_rejects_match_and_call_in_graph_constructor() -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("GRAPH { MATCH (a)-[r]->(b) CALL graphistry.degree.write() }")


def test_parse_rejects_empty_graph_constructor() -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("GRAPH { }")


def test_parse_rejects_return_inside_graph_constructor() -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("GRAPH { MATCH (a) RETURN a }")


def test_parse_rejects_unwind_inside_graph_constructor() -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("GRAPH { UNWIND [1,2] AS x }")


def test_parse_rejects_duplicate_graph_binding_name() -> None:
    with pytest.raises(GFQLValidationError):
        parse_cypher(
            "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
            "GRAPH g1 = GRAPH { MATCH (c)-[s]->(d) } "
            "MATCH (x) RETURN x"
        )


def test_parse_rejects_unresolved_use_reference() -> None:
    with pytest.raises(GFQLValidationError):
        parse_cypher("USE missing MATCH (x) RETURN x")


def test_parse_rejects_forward_use_reference() -> None:
    with pytest.raises(GFQLValidationError):
        parse_cypher(
            "GRAPH g1 = GRAPH { USE g2 MATCH (a)-[r]->(b) } "
            "GRAPH g2 = GRAPH { MATCH (c)-[s]->(d) } "
            "MATCH (x) RETURN x"
        )


def test_parse_rejects_self_referential_use() -> None:
    with pytest.raises(GFQLValidationError):
        parse_cypher(
            "GRAPH g1 = GRAPH { USE g1 MATCH (a)-[r]->(b) } "
            "MATCH (x) RETURN x"
        )


def test_parse_linear_pattern_with_labels_properties_and_aliases() -> None:
    parsed = _parse_query(
        'MATCH (p:Person {id: $person_id})-[r:FOLLOWS]->(q:Person {active: true}) RETURN p.name AS person_name, q'
    )

    assert parsed.match is not None
    assert len(parsed.match.pattern) == 3

    left = parsed.match.pattern[0]
    rel = parsed.match.pattern[1]
    right = parsed.match.pattern[2]

    assert isinstance(left, NodePattern)
    assert isinstance(rel, RelationshipPattern)
    assert isinstance(right, NodePattern)

    assert left.variable == "p"
    assert left.labels == ("Person",)
    assert left.properties[0].key == "id"
    assert isinstance(left.properties[0].value, ParameterRef)
    assert left.properties[0].value.name == "person_id"

    assert rel.variable == "r"
    assert rel.direction == "forward"
    assert rel.types == ("FOLLOWS",)

    assert right.variable == "q"
    assert right.labels == ("Person",)
    assert right.properties[0].key == "active"
    assert right.properties[0].value is True

    assert parsed.return_.items[0].expression.text == "p.name"
    assert parsed.return_.items[0].alias == "person_name"
    assert parsed.return_.items[1].expression.text == "q"


def test_parse_expression_valued_pattern_property_entry() -> None:
    parsed = _parse_query("MATCH (a)-[:R]->(b {id: a.id}) RETURN b")

    assert parsed.match is not None
    right = parsed.match.pattern[2]
    assert isinstance(right, NodePattern)
    assert isinstance(right.properties[0].value, ExpressionText)
    assert right.properties[0].value.text == "a.id"


def test_parse_identifier_valued_pattern_property_entry() -> None:
    parsed = _parse_query("MATCH (a)-[:R]->(b {id: carriedId}) RETURN b")

    assert parsed.match is not None
    right = parsed.match.pattern[2]
    assert isinstance(right, NodePattern)
    assert isinstance(right.properties[0].value, ExpressionText)
    assert right.properties[0].value.text == "carriedId"


def test_parse_relationship_expression_valued_pattern_property_entry() -> None:
    parsed = _parse_query("MATCH (a)-[r:R {weight: a.num}]->(b) RETURN r")

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert isinstance(rel.properties[0].value, ExpressionText)
    assert rel.properties[0].value.text == "a.num"


def test_parse_shortest_path_bound_pattern() -> None:
    parsed = _parse_query(
        "MATCH (a:Person {id: $person1Id}), (b:Person {id: $person2Id}), "
        "path = shortestPath((a)-[:KNOWS*]-(b)) "
        "RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength"
    )

    assert parsed.match is not None
    assert len(parsed.match.patterns) == 3
    assert parsed.match.pattern_aliases == (None, None, "path")
    assert parsed.match.pattern_alias_kinds == ("pattern", "pattern", "shortestPath")


@pytest.mark.parametrize(
    "query,direction",
    [
        ("MATCH (a)<-[r:KNOWS]-(b) RETURN r", "reverse"),
        ("MATCH (a)-[r:KNOWS]-(b) RETURN r;", "undirected"),
        ("MATCH (a)-->(b) RETURN b", "forward"),
        ("MATCH (a)<--(b) RETURN b", "reverse"),
        ("MATCH (a)--(b) RETURN b", "undirected"),
        ("MATCH (a)<-->(b) RETURN b", "undirected"),
    ],
)
def test_parse_relationship_directions(query: str, direction: str) -> None:
    parsed = _parse_query(query)

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.direction == direction


def test_parse_reports_trailing_semicolon() -> None:
    parsed = _parse_query("MATCH (n) RETURN n;")
    assert parsed.trailing_semicolon is True


def test_parse_union_distinct_query() -> None:
    parsed = parse_cypher("RETURN 1 AS x UNION RETURN 2 AS x")

    assert isinstance(parsed, CypherUnionQuery)
    assert parsed.union_kind == "distinct"
    assert len(parsed.branches) == 2
    assert [branch.return_.items[0].alias for branch in parsed.branches] == ["x", "x"]


def test_parse_union_all_query() -> None:
    parsed = parse_cypher("RETURN 1 AS x UNION ALL RETURN 2 AS x;")

    assert isinstance(parsed, CypherUnionQuery)
    assert parsed.union_kind == "all"
    assert parsed.trailing_semicolon is True


def test_parse_mixed_union_kinds_rejected() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        parse_cypher("RETURN 1 AS x UNION RETURN 2 AS x UNION ALL RETURN 3 AS x")

    assert exc_info.value.code == ErrorCode.E107


def test_parse_call_with_optional_yield_and_return() -> None:
    parsed = _parse_query("CALL graphistry.igraph.pagerank() YIELD nodeId, pagerank AS pr RETURN nodeId, pr")

    assert isinstance(parsed.call, CallClause)
    assert parsed.call.procedure == "graphistry.igraph.pagerank"
    assert parsed.call.args == ()
    assert tuple((item.name, item.alias) for item in parsed.call.yield_items) == (
        ("nodeId", None),
        ("pagerank", "pr"),
    )
    assert parsed.return_.items[0].expression.text == "nodeId"
    assert parsed.return_.items[1].expression.text == "pr"


def test_parse_call_with_options_map() -> None:
    parsed = _parse_query("CALL graphistry.cugraph.louvain({resolution: 1.0, directed: false})")

    assert isinstance(parsed.call, CallClause)
    assert parsed.call.procedure == "graphistry.cugraph.louvain"
    assert len(parsed.call.args) == 1
    assert parsed.call.args[0].text == "{resolution: 1.0, directed: false}"


def test_parse_call_without_return_synthesizes_row_sequence() -> None:
    parsed = _parse_query("CALL graphistry.degree()")

    assert isinstance(parsed.call, CallClause)
    assert len(parsed.row_sequence) == 1
    assert parsed.return_.items[0].expression.text == "*"


def test_parse_bare_call_without_parentheses() -> None:
    parsed = _parse_query("CALL graphistry.degree YIELD nodeId RETURN nodeId")

    assert isinstance(parsed.call, CallClause)
    assert parsed.call.procedure == "graphistry.degree"
    assert parsed.call.args == ()
    assert tuple((item.name, item.alias) for item in parsed.call.yield_items) == (("nodeId", None),)


def test_parse_where_clause() -> None:
    parsed = _parse_query("MATCH (a)-[r]->(b) WHERE a.team = b.team AND b.score >= 10 RETURN a")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 2
    pred0 = parsed.where.predicates[0]
    pred1 = parsed.where.predicates[1]
    assert not isinstance(pred0, WherePatternPredicate)
    assert not isinstance(pred1, WherePatternPredicate)
    left0 = cast(PropertyRef, pred0.left)
    left1 = cast(PropertyRef, pred1.left)
    assert left0.alias == "a"
    assert left0.property == "team"
    assert pred0.op == "=="
    assert left1.alias == "b"
    assert left1.property == "score"
    assert pred1.op == ">="


def test_parse_where_and_label_predicates_produces_structured_ast() -> None:
    # AND-joined bare label predicates must land in WhereClause.predicates (not .expr)
    # because Lark routes them to generic_where_clause via ambiguity resolution.
    parsed = _parse_query("MATCH (n) WHERE n:Admin AND n:Active RETURN n")

    assert parsed.where is not None
    assert parsed.where.expr_tree is None
    assert len(parsed.where.predicates) == 2
    p0, p1 = parsed.where.predicates
    assert isinstance(p0, WherePredicate) and isinstance(p0.left, LabelRef)
    assert p0.op == "has_labels"
    assert p0.left.alias == "n"
    assert set(p0.left.labels) == {"Admin"}
    assert isinstance(p1, WherePredicate) and isinstance(p1.left, LabelRef)
    assert p1.left.alias == "n"
    assert set(p1.left.labels) == {"Active"}


def test_parse_where_single_label_predicate_produces_structured_ast() -> None:
    parsed = _parse_query("MATCH (n) WHERE n:Admin RETURN n")

    assert parsed.where is not None
    assert parsed.where.expr_tree is None
    assert len(parsed.where.predicates) == 1
    p = parsed.where.predicates[0]
    assert isinstance(p, WherePredicate) and isinstance(p.left, LabelRef)
    assert p.op == "has_labels"
    assert p.left.alias == "n"
    assert set(p.left.labels) == {"Admin"}


def test_parse_where_non_label_expression_produces_raw_expr() -> None:
    # Non-label WHERE expressions land through a different grammar path
    # (`where_predicates` or raw expr); the contract under review is that
    # `generic_where_clause` never synthesizes fake has_labels predicates
    # from non-label text.  Assert no has_labels predicate is present.
    parsed = _parse_query("MATCH (n) WHERE n.name = 'alice' RETURN n")

    assert parsed.where is not None
    assert all(
        not (isinstance(p, WherePredicate) and p.op == "has_labels")
        for p in parsed.where.predicates
    )


def test_parse_where_xor_label_expression_stays_as_raw_expr() -> None:
    # XOR is not handled by generic_where_clause AND-split; must stay in .expr.
    parsed = _parse_query("MATCH (n) WHERE n:Admin XOR n:Active RETURN n")

    assert parsed.where is not None
    assert parsed.where.expr_tree is not None
    assert "XOR" in boolean_expr_to_text(parsed.where.expr_tree).upper()
    assert parsed.where.predicates == ()


def test_parse_where_triple_and_label_conjunction_through_generic_where_clause() -> None:
    # End-to-end coverage that a triple-AND bare-label WHERE still routes
    # through ``generic_where_clause`` and is lifted into structured
    # ``WhereClause.predicates`` by walking the parsed ``BooleanExpr``
    # AND-spine (#1194).  No text-level AND splitting is involved.
    parsed = _parse_query("MATCH (n) WHERE n:Admin AND n:Active AND n:Super RETURN n")

    assert parsed.where is not None
    assert parsed.where.expr_tree is None
    assert len(parsed.where.predicates) == 3
    aliases = [
        p.left.alias
        for p in parsed.where.predicates
        if isinstance(p, WherePredicate) and isinstance(p.left, LabelRef)
    ]
    assert aliases == ["n", "n", "n"]


# --- Unit tests for the label-lifting helpers (#1194 walker) ---


def test_match_bare_label_atom_accepts_alias_and_labels() -> None:
    from graphistry.compute.gfql.cypher.parser import _match_bare_label_atom

    assert _match_bare_label_atom("n:Admin") == ("n", ("Admin",))
    assert _match_bare_label_atom("b:Foo:Bar") == ("b", ("Foo", "Bar"))
    assert _match_bare_label_atom("  n:Admin  ") == ("n", ("Admin",))


def test_match_bare_label_atom_rejects_non_label_text() -> None:
    # ``fullmatch`` is load-bearing as the false-positive guard from #1125 —
    # text fragments that merely look label-shaped must not lift.
    from graphistry.compute.gfql.cypher.parser import _match_bare_label_atom

    assert _match_bare_label_atom(None) is None
    assert _match_bare_label_atom("") is None
    assert _match_bare_label_atom("n.prop = 1") is None
    assert _match_bare_label_atom("n:Admin AND extra") is None
    assert _match_bare_label_atom("'A:B'") is None  # quoted string fragment


def _bx_atom(text: str):  # type: ignore[no-untyped-def]
    from graphistry.compute.gfql.cypher.ast import BooleanExpr, SourceSpan

    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    return BooleanExpr(op="atom", span=span, atom_text=text, atom_span=span)


def _bx_branch(op, left, right=None):  # type: ignore[no-untyped-def]
    from graphistry.compute.gfql.cypher.ast import BooleanExpr, SourceSpan

    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    return BooleanExpr(op=op, span=span, left=left, right=right)


def test_lift_label_only_and_spine_walks_and_chain() -> None:
    from graphistry.compute.gfql.cypher.parser import _lift_label_only_and_spine

    # Left-associative AND: ((a AND b) AND c) — depth ordering preserves left-to-right.
    tree = _bx_branch(
        "and",
        _bx_branch("and", _bx_atom("n:Admin"), _bx_atom("n:Active")),
        _bx_atom("n:Super"),
    )
    assert _lift_label_only_and_spine(tree) == (
        ("n", ("Admin",)),
        ("n", ("Active",)),
        ("n", ("Super",)),
    )


def test_lift_label_only_and_spine_rejects_non_and_or_non_label() -> None:
    from graphistry.compute.gfql.cypher.parser import _lift_label_only_and_spine

    # OR root → reject (we only lift pure AND-spines).
    assert _lift_label_only_and_spine(
        _bx_branch("or", _bx_atom("n:Admin"), _bx_atom("n:Active"))
    ) is None
    # NOT root → reject.
    assert _lift_label_only_and_spine(_bx_branch("not", _bx_atom("n:Admin"))) is None
    # AND with a non-label leaf → reject (mixed predicates fall through).
    assert _lift_label_only_and_spine(
        _bx_branch("and", _bx_atom("n:Admin"), _bx_atom("n.prop = 1"))
    ) is None
    # Single-atom BooleanExpr → lifts as one predicate.
    assert _lift_label_only_and_spine(_bx_atom("n:Admin")) == (("n", ("Admin",)),)


def test_parse_where_null_predicates() -> None:
    parsed = _parse_query("MATCH (a)-[r]->(b) WHERE a.deleted IS NULL AND b.name IS NOT NULL RETURN a")

    assert parsed.where is not None
    pred0 = parsed.where.predicates[0]
    pred1 = parsed.where.predicates[1]
    assert not isinstance(pred0, WherePatternPredicate)
    assert not isinstance(pred1, WherePatternPredicate)
    assert pred0.op == "is_null"
    assert pred0.right is None
    assert pred1.op == "is_not_null"
    assert pred1.right is None


def test_parse_return_xor_precedence_expression() -> None:
    parsed = _parse_query("RETURN true OR true XOR false AND false AS result")

    assert parsed.return_.items[0].expression.text == "true OR true XOR false AND false"


def test_parse_return_searched_case_expression() -> None:
    parsed = _parse_query("RETURN CASE WHEN score > 1 THEN true ELSE false END AS result")

    assert parsed.return_.items[0].expression.text == "CASE WHEN score > 1 THEN true ELSE false END"
    assert parsed.return_.items[0].alias == "result"


def test_parse_return_simple_case_expression() -> None:
    parsed = _parse_query("RETURN CASE score WHEN 1 THEN 'one' ELSE 'other' END AS result")

    assert parsed.return_.items[0].expression.text == "CASE score WHEN 1 THEN 'one' ELSE 'other' END"
    assert parsed.return_.items[0].alias == "result"


@pytest.mark.parametrize(
    "query,expr_text",
    [
        ("RETURN 0x1 AS literal", "0x1"),
        ("RETURN -0x1 AS literal", "-0x1"),
        ("RETURN 0o7 AS literal", "0o7"),
        ("RETURN -0o7 AS literal", "-0o7"),
        ("RETURN .1e9 AS literal", ".1e9"),
        ("RETURN -.1e-5 AS literal", "-.1e-5"),
    ],
)
def test_parse_numeric_literal_forms(query: str, expr_text: str) -> None:
    parsed = _parse_query(query)

    assert parsed.return_.items[0].expression.text == expr_text


def test_parse_with_where_pipeline() -> None:
    parsed = _parse_query("UNWIND [true, false, null] AS a WITH a WHERE a IS NULL RETURN a")

    assert len(parsed.with_stages) == 1
    assert parsed.with_stages[0].where is not None
    assert parsed.with_stages[0].where.text == "a IS NULL"


def test_parse_match_after_with_reentry_shape() -> None:
    parsed = _parse_query("MATCH (a:A) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-->(b) RETURN a")

    assert len(parsed.matches) == 1
    assert len(parsed.with_stages) == 1
    assert len(parsed.reentry_matches) == 1
    assert parsed.reentry_where is None


def test_parse_match_with_unwind_then_reentry_shape() -> None:
    parsed = _parse_query(
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c"
    )

    assert len(parsed.matches) == 1
    assert len(parsed.with_stages) == 1
    assert parsed.with_stages[0].clause.items[0].expression.text == "collect(b1)"
    assert len(parsed.unwinds) == 1
    assert parsed.unwinds[0].expression.text == "bees"
    assert parsed.unwinds[0].alias == "b2"
    assert len(parsed.reentry_matches) == 1
    assert parsed.reentry_where is None


def test_parse_reentry_match_with_collect_unwind_then_reentry_shape() -> None:
    parsed = _parse_query(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH collect(distinct c) AS cs "
        "UNWIND cs AS c2 "
        "MATCH (c2)-[:T]->(d:D) "
        "WITH d "
        "RETURN d.id AS id"
    )

    assert len(parsed.matches) == 1
    assert len(parsed.with_stages) == 3
    assert parsed.with_stages[1].clause.items[0].expression.text == "collect(distinct c)"
    assert len(parsed.reentry_matches) == 2
    assert len(parsed.reentry_unwinds) == 1
    assert parsed.reentry_unwinds[0].expression.text == "cs"
    assert parsed.reentry_unwinds[0].alias == "c2"


def test_parse_reentry_match_with_multiple_where_stages() -> None:
    parsed = _parse_query(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WHERE c.id = 'c' "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "WITH d "
        "RETURN d.id AS id"
    )

    assert len(parsed.reentry_matches) == 2
    assert len(parsed.reentry_wheres) == 2
    assert parsed.reentry_wheres[0] is not None
    assert len(parsed.reentry_wheres[0].predicates) == 1
    assert parsed.reentry_wheres[1] is not None
    assert len(parsed.reentry_wheres[1].predicates) == 1


def test_parse_reentry_match_with_sparse_where_stages() -> None:
    parsed = _parse_query(
        "MATCH (a:A)-[:R]->(b:B) "
        "WITH b "
        "MATCH (b)-[:S]->(c:C) "
        "WITH c "
        "MATCH (c)-[:T]->(d:D) "
        "WHERE d.id = 'd' "
        "WITH d "
        "RETURN d.id AS id"
    )

    assert len(parsed.reentry_matches) == 2
    assert parsed.reentry_wheres[0] is None
    assert parsed.reentry_wheres[1] is not None
    assert len(parsed.reentry_wheres[1].predicates) == 1


def test_parse_rejects_multiple_reentry_unwinds() -> None:
    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher only supports one UNWIND after post-WITH MATCH",
    ):
        _parse_query(
            "MATCH (a:A)-[:R]->(b:B) "
            "WITH b "
            "MATCH (b)-[:S]->(c:C) "
            "UNWIND [c] AS c2 "
            "UNWIND [c2] AS c3 "
            "RETURN c3"
        )


def test_parse_rejects_match_after_reentry_unwind() -> None:
    with pytest.raises(
        GFQLSyntaxError,
        match="Cypher MATCH after post-WITH MATCH UNWIND is not yet supported",
    ):
        _parse_query(
            "MATCH (a:A)-[:R]->(b:B) "
            "WITH b "
            "MATCH (b)-[:S]->(c:C) "
            "WITH collect(distinct c) AS cs "
            "UNWIND cs AS c2 "
            "MATCH (c2)-[:T]->(d:D) "
            "MATCH (d)-[:Z]->(e) "
            "RETURN e"
        )


def test_parse_where_label_predicate() -> None:
    parsed = _parse_query("MATCH (a)-->(b) WHERE b:Foo:Bar RETURN b")

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 1
    predicate = parsed.where.predicates[0]
    assert not isinstance(predicate, WherePatternPredicate)
    assert predicate.op == "has_labels"
    left = cast(LabelRef, predicate.left)
    assert left.alias == "b"
    assert left.labels == ("Foo", "Bar")


def test_parse_reserved_keyword_labels_and_relationship_types() -> None:
    parsed = _parse_query("MATCH (n:Single)-[r:SINGLE]->(m:End) RETURN m:TYPE, n:Single")

    assert parsed.match is not None
    left = parsed.match.pattern[0]
    rel = parsed.match.pattern[1]
    right = parsed.match.pattern[2]
    assert isinstance(left, NodePattern)
    assert isinstance(rel, RelationshipPattern)
    assert isinstance(right, NodePattern)
    assert left.labels == ("Single",)
    assert rel.types == ("SINGLE",)
    assert right.labels == ("End",)
    assert parsed.return_.items[0].expression.text == "m:TYPE"
    assert parsed.return_.items[1].expression.text == "n:Single"


def test_parse_relationship_type_alternation() -> None:
    parsed = _parse_query("MATCH (a)-[r:KNOWS|HATES]->(b) RETURN r")

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.types == ("KNOWS", "HATES")


def test_parse_relationship_type_alternation_with_repeated_colon() -> None:
    parsed = _parse_query("MATCH (a)-[:T|:OTHER]->(b) RETURN b")

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.types == ("T", "OTHER")


@pytest.mark.parametrize(
    "query,direction,min_hops,max_hops,to_fixed_point,types",
    [
        ("MATCH (a)-[*2]->(b) RETURN b", "forward", 2, 2, False, ()),
        ("MATCH (a)-[:R*1..3]->(b) RETURN b", "forward", 1, 3, False, ("R",)),
        ("MATCH (a)<-[*4]-(b) RETURN b", "reverse", 4, 4, False, ()),
        ("MATCH (a)-[*]->(b) RETURN b", "forward", None, None, True, ()),
        ("MATCH (a)-[:R|:S*2..4]-(b) RETURN b", "undirected", 2, 4, False, ("R", "S")),
        ("MATCH (a)-[*0..]->(b) RETURN b", "forward", 0, None, True, ()),
        ("MATCH (a)-[*1..]->(b) RETURN b", "forward", 1, None, True, ()),
        ("MATCH (a)-[:R*0..]-(b) RETURN b", "undirected", 0, None, True, ("R",)),
        ("MATCH (a)<-[:R*2..]-(b) RETURN b", "reverse", 2, None, True, ("R",)),
        # #983: bounded zero-min ranges
        ("MATCH (a)-[*0..3]->(b) RETURN b", "forward", 0, 3, False, ()),
        ("MATCH (a)-[:R*0..5]->(b) RETURN b", "forward", 0, 5, False, ("R",)),
        ("MATCH (a)-[:R|S*0..2]-(b) RETURN b", "undirected", 0, 2, False, ("R", "S")),
    ],
)
def test_parse_variable_length_relationship_patterns(
    query: str,
    direction: str,
    min_hops: int | None,
    max_hops: int | None,
    to_fixed_point: bool,
    types: tuple[str, ...],
) -> None:
    parsed = _parse_query(query)

    assert parsed.match is not None
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.direction == direction
    assert rel.min_hops == min_hops
    assert rel.max_hops == max_hops
    assert rel.to_fixed_point is to_fixed_point
    assert rel.types == types


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (a)-[*-1]->(b) RETURN b",
        "MATCH (a)-[*-5..10]->(b) RETURN b",
        "MATCH (a)-[*1..-3]->(b) RETURN b",
    ],
)
def test_parse_negative_hop_bound_raises_syntax_error(query: str) -> None:
    """Negative hop bounds must raise GFQLSyntaxError specifically (#983)."""
    with pytest.raises(GFQLSyntaxError):
        _parse_query(query)


def test_parse_bound_variable_length_relationship_pattern_alias() -> None:
    parsed = _parse_query("MATCH p = (a)-[:R*2]->(b) RETURN b")

    assert parsed.match is not None
    assert parsed.match.pattern_aliases == ("p",)
    rel = parsed.match.pattern[1]
    assert isinstance(rel, RelationshipPattern)
    assert rel.types == ("R",)
    assert rel.min_hops == 2
    assert rel.max_hops == 2


def test_parse_return_pipeline_clauses() -> None:
    parsed = _parse_query(
        "MATCH (p:Person) RETURN DISTINCT p.name AS person_name ORDER BY person_name DESC, p.id ASC SKIP 1 LIMIT 2;"
    )

    assert parsed.return_.kind == "return"
    assert parsed.return_.distinct is True
    assert parsed.return_.items[0].expression.text == "p.name"
    assert parsed.return_.items[0].alias == "person_name"
    assert parsed.order_by is not None
    assert [item.expression.text for item in parsed.order_by.items] == ["person_name", "p.id"]
    assert [item.direction for item in parsed.order_by.items] == ["desc", "asc"]
    assert parsed.skip is not None and isinstance(parsed.skip.value, ExpressionText)
    assert parsed.skip.value.text == "1"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "2"
    assert parsed.trailing_semicolon is True


def test_parse_terminal_with_clause() -> None:
    parsed = _parse_query("MATCH (p) WITH p.name AS person_name ORDER BY person_name ASC LIMIT 5")

    assert parsed.with_stages == ()
    assert parsed.return_.kind == "with"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "person_name"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "5"


def test_parse_with_then_return_pipeline() -> None:
    parsed = _parse_query("UNWIND [1, 3, 2] AS ints WITH ints ORDER BY ints DESC LIMIT 2 RETURN ints")

    assert len(parsed.with_stages) == 1
    with_stage = parsed.with_stages[0]
    assert with_stage.clause.kind == "with"
    assert with_stage.clause.items[0].expression.text == "ints"
    assert with_stage.order_by is not None
    assert with_stage.order_by.items[0].expression.text == "ints"
    assert with_stage.order_by.items[0].direction == "desc"
    assert with_stage.limit is not None and isinstance(with_stage.limit.value, ExpressionText)
    assert with_stage.limit.value.text == "2"
    assert parsed.return_.kind == "return"
    assert parsed.return_.items[0].expression.text == "ints"


def test_parse_multiple_with_stages_and_long_order_directions() -> None:
    parsed = _parse_query("WITH 1 AS a, 2 AS b WITH a ORDER BY a ASCENDING WITH a ORDER BY a DESCENDING RETURN a")

    assert len(parsed.with_stages) == 3
    assert len(parsed.row_sequence) == 4
    assert parsed.with_stages[0].clause.items[0].expression.text == "1"
    assert parsed.with_stages[1].order_by is not None
    assert parsed.with_stages[1].order_by.items[0].direction == "asc"
    assert parsed.with_stages[2].order_by is not None
    assert parsed.with_stages[2].order_by.items[0].direction == "desc"
    assert parsed.return_.items[0].expression.text == "a"


def test_parse_interleaved_row_only_with_unwind_sequence() -> None:
    parsed = _parse_query(
        "WITH [0, 1] AS prows, [[2], [3, 4]] AS qrows "
        "UNWIND prows AS p "
        "UNWIND qrows[p] AS q "
        "WITH p, count(q) AS rng "
        "RETURN p "
        "ORDER BY rng"
    )

    assert parsed.match is None
    assert len(parsed.row_sequence) == 5
    assert parsed.return_.items[0].expression.text == "p"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "rng"


def test_parse_unwind_without_match() -> None:
    parsed = _parse_query("UNWIND [1, 2, 3] AS x RETURN x ORDER BY x")

    assert parsed.match is None
    assert len(parsed.unwinds) == 1
    assert parsed.unwinds[0].expression.text == "[1, 2, 3]"
    assert parsed.unwinds[0].alias == "x"
    assert parsed.return_.items[0].expression.text == "x"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "x"


def test_parse_match_then_unwind() -> None:
    parsed = _parse_query("MATCH (p) UNWIND p.vals AS v RETURN v")

    assert parsed.match is not None
    assert len(parsed.unwinds) == 1
    assert parsed.unwinds[0].expression.text == "p.vals"
    assert parsed.unwinds[0].alias == "v"


def test_parse_match_with_comma_connected_patterns() -> None:
    parsed = _parse_query("MATCH (a)-[:A]->(b), (b)-[:B]->(c) RETURN c")

    assert parsed.match is not None
    assert len(parsed.match.patterns) == 2
    assert cast(NodePattern, parsed.match.patterns[0][0]).variable == "a"
    assert cast(NodePattern, parsed.match.patterns[0][2]).variable == "b"
    assert cast(NodePattern, parsed.match.patterns[1][0]).variable == "b"
    assert cast(NodePattern, parsed.match.patterns[1][2]).variable == "c"


def test_parse_multiple_match_clauses() -> None:
    parsed = _parse_query("MATCH (a {name: 'A'}), (b {name: 'B'}) MATCH (a)-->(x)<--(b) RETURN x")

    assert len(parsed.matches) == 2
    assert len(parsed.matches[0].patterns) == 2
    assert len(parsed.matches[1].patterns) == 1
    assert cast(NodePattern, parsed.matches[0].patterns[0][0]).variable == "a"
    assert cast(NodePattern, parsed.matches[0].patterns[1][0]).variable == "b"
    assert cast(NodePattern, parsed.matches[1].patterns[0][2]).variable == "x"


def test_parse_optional_match_clause() -> None:
    parsed = _parse_query("OPTIONAL MATCH (n) RETURN n.exists IS NULL AS missing")

    assert len(parsed.matches) == 1
    assert parsed.matches[0].optional is True
    assert cast(NodePattern, parsed.matches[0].patterns[0][0]).variable == "n"


@pytest.mark.parametrize(
    "query,expr_text",
    [
        ("MATCH (n) WHERE (n)-[:R*]->() AND n.id <> 'a' RETURN n", "n.id <> 'a'"),
        ("MATCH (n) WHERE n.id <> 'a' AND (n)-[:R*]->() RETURN n", "n.id <> 'a'"),
        (
            "MATCH (n) WHERE n.id <> 'a' AND (n)-[:R*]->() AND n.kind = 'x' RETURN n",
            "n.id <> 'a' AND n.kind = 'x'",
        ),
        (
            "MATCH (n) WHERE n.kind = 'x' AND (n)-[:R*]->() AND n.id <> 'a' RETURN n",
            "n.kind = 'x' AND n.id <> 'a'",
        ),
        (
            "MATCH (n) WHERE (n)-[:R*]->() AND n.id <> 'a' AND n.kind = 'x' RETURN n",
            "n.id <> 'a' AND n.kind = 'x'",
        ),
        (
            "MATCH (n) WHERE n.id <> 'a' AND n.kind = 'x' AND (n)-[:R*]->() RETURN n",
            "n.id <> 'a' AND n.kind = 'x'",
        ),
        (
            "MATCH (n) WHERE (n)-[:R*]->() AND (n.id = 'b' OR n.id = 'c') RETURN n",
            "n.id = 'b' OR n.id = 'c'",
        ),
    ],
)
def test_parse_supports_where_pattern_predicate_and_expr_mix(query: str, expr_text: str) -> None:
    parsed = _parse_query(query)

    assert parsed.where is not None
    assert len(parsed.where.predicates) == 1
    assert isinstance(parsed.where.predicates[0], WherePatternPredicate)
    assert parsed.where.expr_tree is not None
    assert boolean_expr_to_text(parsed.where.expr_tree) == expr_text


@pytest.mark.parametrize(
    "query",
    [
        # OR-around-pattern remains slice 4 territory — still rejected at parse.
        "MATCH (n) WHERE (n)-[:R*]->() OR n.id = 'z' RETURN n",
        "MATCH (n) WHERE (n)-[:R]->() OR n.id = 'z' RETURN n",
    ],
)
def test_parse_rejects_mixed_where_pattern_predicates_as_unsupported(query: str) -> None:
    with pytest.raises(GFQLValidationError, match="mixed with generic row expressions"):
        parse_cypher(query)


@pytest.mark.parametrize(
    "query",
    [
        # NOT-pattern: parse succeeds (#1031 slice 2 plumbing); the inner
        # ``WherePatternPredicate`` is lifted with ``negated=True`` and
        # surfaces the lowering-stage gate when compiled.
        "MATCH (n) WHERE NOT (n)-[:R*]->() RETURN n",
        "MATCH (n) WHERE NOT (n)-[:R]->() RETURN n",
    ],
)
def test_parse_lifts_top_level_not_pattern_to_negated_predicate(query: str) -> None:
    parsed = _parse_query(query)
    assert parsed.where is not None
    pattern_preds = [
        p for p in parsed.where.predicates if isinstance(p, WherePatternPredicate)
    ]
    assert len(pattern_preds) == 1
    assert pattern_preds[0].negated is True


def test_parse_aggregate_projection_items() -> None:
    parsed = _parse_query(
        "MATCH (n) RETURN n.division AS division, count(*) AS cnt, max(n.age) AS max_age ORDER BY division, cnt DESC"
    )

    assert [item.expression.text for item in parsed.return_.items] == [
        "n.division",
        "count(*)",
        "max(n.age)",
    ]
    assert [item.alias for item in parsed.return_.items] == ["division", "cnt", "max_age"]
    assert parsed.order_by is not None
    assert [item.expression.text for item in parsed.order_by.items] == ["division", "cnt"]


def test_parse_top_level_projection_only() -> None:
    parsed = _parse_query("RETURN [1, 2, 3] AS xs LIMIT 1")

    assert parsed.match is None
    assert parsed.unwinds == ()
    assert parsed.return_.items[0].expression.text == "[1, 2, 3]"
    assert parsed.return_.items[0].alias == "xs"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "1"


def test_parse_top_level_quantifier_expression() -> None:
    parsed = _parse_query("RETURN none(x IN [true, false] WHERE x) AS result")

    assert parsed.match is None
    assert parsed.return_.items[0].expression.text == "none(x IN [true, false] WHERE x)"
    assert parsed.return_.items[0].alias == "result"


def test_parse_top_level_membership_and_null_expression() -> None:
    parsed = _parse_query("RETURN 3 IN [1, 2, 3] AS hit, null IS NULL AS empty")

    assert [item.expression.text for item in parsed.return_.items] == [
        "3 IN [1, 2, 3]",
        "null IS NULL",
    ]


def test_parse_top_level_list_comprehension_expression() -> None:
    parsed = _parse_query("RETURN [x IN [1, 2, 3] WHERE x > 1 | x + 10] AS vals")

    assert parsed.return_.items[0].expression.text == "[x IN [1, 2, 3] WHERE x > 1 | x + 10]"


def test_invalid_syntax_reports_line_and_column() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        _parse_query("MATCH (n RETURN n")

    exc = exc_info.value
    assert exc.code == ErrorCode.E107
    assert exc.context["line"] == 1
    assert exc.context["column"] >= 1
    assert "Invalid Cypher query syntax" in str(exc)


def test_multi_statement_rejected() -> None:
    with pytest.raises(GFQLSyntaxError) as exc_info:
        _parse_query("MATCH (n) RETURN n; MATCH (m) RETURN m")

    assert exc_info.value.code == ErrorCode.E107


def test_or_where_now_parses_after_earley_swap() -> None:
    # #1031 slice 1: Earley accepts OR in WHERE bodies that LALR(1) had to
    # reject due to FIRST-set state collapse.  The query parses cleanly,
    # produces a structured ``BooleanExpr(op="or", ...)`` tree on
    # ``expr_tree``, and ``predicates`` stays empty because OR is not a
    # ``where_predicates`` form.  Downstream consumers see the OR as a
    # single conjunct.
    parsed = _parse_query("MATCH (a)-[r]->(b) WHERE a.team = b.team OR a.name = 'Alice' RETURN a")

    assert parsed.where is not None
    assert parsed.where.predicates == ()
    assert parsed.where.expr_tree is not None
    assert parsed.where.expr_tree.op == "or"
    assert "OR" in boolean_expr_to_text(parsed.where.expr_tree).upper()
