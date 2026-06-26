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
from graphistry.compute.gfql.cypher.ast import BooleanExpr, GraphBinding, GraphConstructor, UseClause


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


def test_parse_graph_binding_with_trailing_semicolon_preserves_query_fields() -> None:
    parsed = parse_cypher(
        "GRAPH g1 = GRAPH { MATCH (a)-[r]->(b) } "
        "USE g1 MATCH (x) RETURN x;"
    )

    assert isinstance(parsed, CypherQuery)
    assert parsed.trailing_semicolon is True
    assert len(parsed.graph_bindings) == 1
    assert parsed.graph_bindings[0].name == "g1"
    assert parsed.use is not None
    assert parsed.use.ref == "g1"
    assert parsed.return_.items[0].expression.text == "x"


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


@pytest.mark.parametrize(
    "query",
    [
        "GRAPH { MATCH (a)-[r]->(b) CALL graphistry.degree.write() }",
        "GRAPH { }",
        "GRAPH { MATCH (a) RETURN a }",
        "GRAPH { UNWIND [1,2] AS x }",
    ],
)
def test_parse_rejects_invalid_graph_constructor_body(query: str) -> None:
    with pytest.raises(GFQLSyntaxError):
        parse_cypher(query)


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


def test_parse_optional_bound_pattern_preserves_alias_metadata() -> None:
    parsed = _parse_query("OPTIONAL MATCH p = (a)-[:R]->(b) RETURN b")

    assert parsed.match is not None
    assert parsed.match.optional is True
    assert parsed.match.pattern_aliases == ("p",)
    assert parsed.match.pattern_alias_kinds == ("pattern",)


def test_parse_optional_shortest_path_preserves_alias_metadata() -> None:
    parsed = _parse_query("OPTIONAL MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN path")

    assert parsed.match is not None
    assert parsed.match.optional is True
    assert parsed.match.pattern_aliases == ("path",)
    assert parsed.match.pattern_alias_kinds == ("shortestPath",)


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


@pytest.mark.parametrize(
    "query,expected_labels",
    [
        ("MATCH (n) WHERE n:Admin RETURN n", [{"Admin"}]),
        ("MATCH (n) WHERE n:Admin AND n:Active RETURN n", [{"Admin"}, {"Active"}]),
        ("MATCH (n) WHERE n:Admin AND n:Active AND n:Super RETURN n", [{"Admin"}, {"Active"}, {"Super"}]),
    ],
)
def test_parse_where_label_predicates_produce_structured_ast(
    query: str,
    expected_labels: list[set[str]],
) -> None:
    parsed = _parse_query(query)

    assert parsed.where is not None
    assert parsed.where.expr_tree is None
    assert len(parsed.where.predicates) == len(expected_labels)
    for predicate, labels in zip(parsed.where.predicates, expected_labels):
        assert isinstance(predicate, WherePredicate) and isinstance(predicate.left, LabelRef)
        assert predicate.op == "has_labels"
        assert predicate.left.alias == "n"
        assert set(predicate.left.labels) == labels


def test_parse_where_non_label_expression_produces_raw_expr() -> None:
    parsed = _parse_query("MATCH (n) WHERE n.name = 'alice' RETURN n")

    assert parsed.where is not None
    assert all(
        not (isinstance(p, WherePredicate) and p.op == "has_labels")
        for p in parsed.where.predicates
    )


def test_parse_where_xor_label_expression_stays_as_raw_expr() -> None:
    parsed = _parse_query("MATCH (n) WHERE n:Admin XOR n:Active RETURN n")

    assert parsed.where is not None
    assert parsed.where.expr_tree is not None
    assert "XOR" in boolean_expr_to_text(parsed.where.expr_tree).upper()
    assert parsed.where.predicates == ()


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE n:Admin OR n:Active RETURN n",
        "MATCH (n) WHERE NOT n:Admin RETURN n",
        "MATCH (n) WHERE 'A:B' = n.value RETURN n",
    ],
)
def test_parse_where_label_like_non_and_spines_stay_raw(query: str) -> None:
    parsed = _parse_query(query)

    assert parsed.where is not None
    assert parsed.where.expr_tree is not None
    assert parsed.where.predicates == ()


def test_label_lift_false_positive_boundaries() -> None:
    from graphistry.compute.gfql.cypher.ast import SourceSpan
    from graphistry.compute.gfql.cypher.parser import _lift_label_only_and_spine, _match_bare_label_atom

    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    label = BooleanExpr(op="atom", span=span, atom_text="n:Admin", atom_span=span)
    non_label = BooleanExpr(op="atom", span=span, atom_text="n.prop = 1", atom_span=span)

    assert _match_bare_label_atom(None) is None
    assert _match_bare_label_atom("'A:B'") is None
    assert _lift_label_only_and_spine(BooleanExpr(op="or", span=span, left=label, right=label)) is None
    assert _lift_label_only_and_spine(BooleanExpr(op="not", span=span, left=label)) is None
    assert _lift_label_only_and_spine(BooleanExpr(op="and", span=span, left=label, right=non_label)) is None


def test_label_lift_helper_positive_boundaries() -> None:
    from graphistry.compute.gfql.cypher.ast import SourceSpan
    from graphistry.compute.gfql.cypher.parser import _lift_label_only_and_spine, _match_bare_label_atom

    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)

    def atom(text: str) -> BooleanExpr:
        return BooleanExpr(op="atom", span=span, atom_text=text, atom_span=span)

    tree = BooleanExpr(
        op="and",
        span=span,
        left=BooleanExpr(op="and", span=span, left=atom("n:Admin"), right=atom("n:Active")),
        right=atom("n:Super"),
    )

    assert _match_bare_label_atom("") is None
    assert _match_bare_label_atom("n:Admin AND extra") is None
    assert _match_bare_label_atom("  b:Foo:Bar  ") == ("b", ("Foo", "Bar"))
    assert _lift_label_only_and_spine(atom("n:Admin")) == (("n", ("Admin",)),)
    assert _lift_label_only_and_spine(tree) == (
        ("n", ("Admin",)),
        ("n", ("Active",)),
        ("n", ("Super",)),
    )


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


def test_parse_ic4_style_return_side_case_expression() -> None:
    parsed = _parse_query(
        "MATCH (person:Person {id: $pid})-[:KNOWS]-(friend:Person), "
        "(friend)<-[:HAS_CREATOR]-(post:Post)-[:HAS_TAG]->(tag:Tag) "
        "WITH DISTINCT tag, post "
        "RETURN tag.name AS tagName, "
        "CASE WHEN 1275350400000 <= post.creationDate AND post.creationDate < 1306886400000 "
        "THEN post.id ELSE null END AS postId"
    )

    assert len(parsed.with_stages) == 1
    assert parsed.with_stages[0].clause.distinct is True
    assert parsed.with_stages[0].clause.items[0].expression.text == "tag"
    assert parsed.with_stages[0].clause.items[1].expression.text == "post"
    assert parsed.return_.items[1].expression.text == (
        "CASE WHEN 1275350400000 <= post.creationDate AND post.creationDate < 1306886400000 "
        "THEN post.id ELSE null END"
    )
    assert parsed.return_.items[1].alias == "postId"


def test_parse_rejects_return_case_missing_end() -> None:
    with pytest.raises(GFQLSyntaxError):
        _parse_query("RETURN CASE WHEN score > 1 THEN true ELSE false AS result")


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


def test_parse_shared_expression_and_page_rule_handlers_preserve_text() -> None:
    parsed = _parse_query(
        "UNWIND [1, 2] AS x WITH x WHERE x > 0 "
        "RETURN x + 1 AS y ORDER BY y DESC SKIP 1 LIMIT 2"
    )

    assert parsed.unwinds[0].expression.text == "[1, 2]"
    assert len(parsed.with_stages) == 1
    assert parsed.with_stages[0].where is not None
    assert parsed.with_stages[0].where.text == "x > 0"
    assert parsed.return_.items[0].expression.text == "x + 1"
    assert parsed.order_by is not None
    assert parsed.order_by.items[0].expression.text == "y"
    assert parsed.skip is not None and isinstance(parsed.skip.value, ExpressionText)
    assert parsed.skip.value.text == "1"
    assert parsed.limit is not None and isinstance(parsed.limit.value, ExpressionText)
    assert parsed.limit.value.text == "2"


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
        # #1236: OR-around-pattern now stays in expr_tree for lowering/runtime.
        "MATCH (n) WHERE (n)-[:R*]->() OR n.id = 'z' RETURN n",
        "MATCH (n) WHERE (n)-[:R]->() OR n.id = 'z' RETURN n",
    ],
)
def test_parse_supports_mixed_where_pattern_predicates_in_expr_tree(query: str) -> None:
    parsed = _parse_query(query)
    assert parsed.where is not None
    assert parsed.where.expr_tree is not None
    assert "OR" in boolean_expr_to_text(parsed.where.expr_tree).upper()
    assert parsed.where.predicates == ()

    def _has_pattern_leaf(node: BooleanExpr) -> bool:
        if node.op == "pattern":
            return True
        left_has = _has_pattern_leaf(node.left) if node.left is not None else False
        right_has = _has_pattern_leaf(node.right) if node.right is not None else False
        return left_has or right_has

    assert _has_pattern_leaf(parsed.where.expr_tree)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE n.txt = 'exists { shadow }' RETURN n",
        "MATCH (n) WHERE n.txt = 'not exists { shadow }' RETURN n",
        "MATCH (n) WHERE n.txt = 'not((a)-[:R]->(b))' RETURN n",
        "MATCH (n) WHERE n.txt = \"exists { shadow }\" RETURN n",
    ],
)
def test_parse_does_not_treat_pattern_existence_lexemes_inside_string_literals_as_unsupported(query: str) -> None:
    parsed = _parse_query(query)
    assert parsed.where is not None


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) // exists { shadow }\nRETURN n",
        "MATCH (n) /* not((a)-[:R]->(b)) */ RETURN n",
        "MATCH (n) WHERE n.txt = 'ok' // exists { shadow }\nRETURN n",
        "RETURN ['exists { shadow }', 'plain'] AS xs",
        "RETURN {k: 'not((a)-[:R]->(b))', v: 1} AS m",
        "RETURN CASE WHEN true THEN 'exists { shadow }' ELSE 'plain' END AS out",
        "RETURN \"contains // exists { shadow }\" AS out",
        "RETURN 'escaped \\'exists { shadow }\\'' AS out",
    ],
)
def test_parse_does_not_treat_pattern_existence_lexemes_inside_comments_or_literal_contexts_as_unsupported(
    query: str,
) -> None:
    parsed = parse_cypher(query)
    assert isinstance(parsed, CypherQuery)


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) WHERE exists { (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE not exists { (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE not((n)-[:R]->()) RETURN n",
        "MATCH (n) WHERE not((n:Admin)-[:R]->()) RETURN n",
        "MATCH (n) WHERE not((n)-[:R {w: 1}]->()) RETURN n",
        "MATCH (n) WHERE not((n)-[:R]->(:Admin)) RETURN n",
        "MATCH (n) WHERE not((n)<-[:R]-()) RETURN n",
        "MATCH (n) WHERE not((n)--()) RETURN n",
        "MATCH (n) WHERE not((n)-[:R*]->()) RETURN n",
        "MATCH (n) WHERE not((n)-[r:R]->()) RETURN n",
        "MATCH (n) WHERE exists/*inline*/{ (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE not/*inline*/exists/*inline*/{ (n)-[:R]->() } RETURN n",
        "MATCH (n) WHERE not/*inline*/((n)-[:R]->()) RETURN n",
        "MATCH (n) WHERE exists { (n)-[:R]->() } /* keep rejecting */ RETURN n",
        "// leading comment\nMATCH (n) WHERE not((n)-[:R]->()) RETURN n",
    ],
)
def test_parse_still_rejects_true_pattern_existence_expressions(query: str) -> None:
    with pytest.raises(GFQLValidationError, match="Pattern existence expressions"):
        _parse_query(query)


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


def test_parse_cypher_is_memoized_and_safe() -> None:
    # parse_cypher memoizes its (deterministic) result: repeated identical
    # queries return the SAME cached frozen AST (skips the ~15ms lark parse).
    from graphistry.compute.gfql.cypher.parser import _parse_cypher_cached

    q = "MATCH (a)-[e]->(b) WHERE a.val > 50 RETURN a.val"
    _parse_cypher_cached.cache_clear()
    first = parse_cypher(q)
    second = parse_cypher(q)
    assert first is second  # cache hit returns the identical object

    # Cached AST is immutable, so sharing it across callers is safe.
    with pytest.raises(Exception):
        first.where = None  # type: ignore[misc]  # frozen dataclass

    # Distinct query text is a distinct cache entry (no cross-contamination).
    other = parse_cypher("MATCH (a) RETURN a.val")
    assert other is not first

    # Clearing the cache and re-parsing yields an equal (value-identical) AST.
    _parse_cypher_cached.cache_clear()
    reparsed = parse_cypher(q)
    assert reparsed == first


def test_parse_cypher_invalid_input_not_cached() -> None:
    # Non-str / empty input must raise (and not poison the cache).
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("")
    with pytest.raises(GFQLSyntaxError):
        parse_cypher("   ")


# --- Single LALR(1) path (Earley removed) -------------------------------------
# After WHERE-grammar unification, parse_cypher uses one LALR(1) parser for every
# supported query -- including OR/XOR/NOT-in-WHERE and post-WITH WHERE, which the
# former dual grammar could only parse under Earley. These smoke-test that the
# whole corpus parses to a valid query on the sole LALR path.

_PARITY_QUERIES = [
    "MATCH (n) WHERE n.x = 50 RETURN n.__rid__",
    "MATCH (n) WHERE n.active = true AND n.x = 5 RETURN n.id",
    "MATCH (a)-[r:R]->(b) RETURN a.id, r.weight, b.id",
    "MATCH (n {a: 1, b: 2}) RETURN n",
    "MATCH (n) WHERE n.x IN [1, 2, 3] RETURN count(n)",
    "OPTIONAL MATCH (n)-[r]->(m) RETURN n, m",
    "MATCH (n) WHERE NOT (n)-[:R]->() RETURN n",
    "MATCH (n) RETURN n.first_name AS fn ORDER BY fn",
    # Formerly Earley-only constructs, now on the LALR path:
    "MATCH (n) WHERE n:Admin AND n.active = true RETURN n",       # label + property
    "MATCH (n) WHERE n.a > 3 OR n.b = 1 RETURN n",                # OR in WHERE
    "MATCH (n) WHERE n.x = 1 XOR n.y = 2 RETURN n",               # XOR in WHERE
    "MATCH (n)-[rel]->(x) WITH n, x WHERE n.animal = x.animal RETURN n, x",  # WITH-WHERE
]


@pytest.mark.parametrize("query", _PARITY_QUERIES)
def test_unified_lalr_parses_corpus(query: str) -> None:
    assert isinstance(parse_cypher(query), CypherQuery)


def test_lalr_is_the_only_parser() -> None:
    # The Earley parser and its fallback hook are gone.
    from graphistry.compute.gfql.cypher import parser as _p

    assert not hasattr(_p, "_parser")
    assert not hasattr(_p, "_lalr_tree_needs_earley")
    assert _p._parser_lalr().parse("MATCH (n) WHERE n.x = 1 OR n.y = 2 RETURN n") is not None


def test_unified_where_lifts_label_and_property_to_structured() -> None:
    # Post grammar-unification: every WHERE parses as the generic expr under the
    # sole LALR parser, and generic_where_clause lifts label + property back to
    # structured predicates.
    label = "MATCH (n) WHERE n:Admin AND n.active = true RETURN n"
    parsed = cast(CypherQuery, parse_cypher(label))
    assert parsed.where is not None
    assert parsed.where.expr_tree is None         # structured via the lift
    assert len(parsed.where.predicates) == 2


# --- Flat-AND spine lift in generic_where_clause -------------------------------
# A parenthesized / nested pure-AND of simple predicates lifts wholly to the
# structured (filter_dict) form instead of dropping to the where_rows engine.
# Anything with a non-liftable conjunct (OR/XOR/NOT, arithmetic) keeps the WHOLE
# clause on expr_tree -- full-lift-only, never a partial split.

def test_nested_pure_and_lifts_to_structured() -> None:
    w = cast(CypherQuery, parse_cypher(
        "MATCH (n) WHERE n.x = 1 AND (n.y = 2 AND n.z = 3) RETURN n"
    )).where
    assert w is not None
    assert w.expr_tree is None            # structured, not a tree
    assert len(w.predicates) == 3         # x, y, z all lifted


def test_and_with_or_stays_on_expr_tree() -> None:
    w = cast(CypherQuery, parse_cypher(
        "MATCH (n) WHERE n.x = 1 AND (n.y = 2 OR n.z = 3) RETURN n"
    )).where
    assert w is not None
    assert w.predicates == ()             # no partial split
    assert w.expr_tree is not None        # whole clause stays a tree


def test_not_in_and_spine_stays_on_expr_tree() -> None:
    w = cast(CypherQuery, parse_cypher(
        "MATCH (n) WHERE NOT n.x = 1 AND n.y = 2 RETURN n"
    )).where
    assert w is not None
    assert w.predicates == ()
    assert w.expr_tree is not None


# --- OR stays on the row engine (no OR->is_in optimization in this PR) ----------
# OR-of-equalities collapse to is_in is a deferred routing optimization; the
# parser-switch lift reproduces the old grammar's routing, under which any OR
# clause stays on expr_tree -> where_rows.
def test_or_clause_stays_on_expr_tree() -> None:
    for q in (
        "MATCH (n) WHERE n.x = 1 OR n.x = 2 OR n.x = 3 RETURN n",  # same column
        "MATCH (n) WHERE n.x = 1 OR n.y = 2 RETURN n",             # cross column
    ):
        w = cast(CypherQuery, parse_cypher(q)).where
        assert w is not None
        assert w.predicates == ()
        assert w.expr_tree is not None


# --- Lifted predicates keep absolute source spans ------------------------------
# Re-parsing each conjunct in isolation would collapse its span to column 1; the
# lift shifts spans back to absolute query coordinates (via the conjunct's
# atom_span) so downstream errors (e.g. E108) point at the real predicate, not
# the start of the query. Regression guard for the column-1 span bug.
def test_lifted_predicate_spans_are_absolute() -> None:
    q = "MATCH (n) WHERE n.x = 1 AND m.y = 2 RETURN n"
    w = cast(CypherQuery, parse_cypher(q)).where
    assert w is not None and w.expr_tree is None  # structured via the lift
    by_alias = {cast(PropertyRef, p.left).alias: p for p in w.predicates}
    # `n.x` / `m.y` sit at their true offsets in the query, not column 1.
    assert q[by_alias["n"].left.span.start_pos:].startswith("n.x")
    assert by_alias["n"].left.span.column == q.index("n.x") + 1
    assert q[by_alias["m"].left.span.start_pos:].startswith("m.y")
    assert by_alias["m"].left.span.column == q.index("m.y") + 1


def test_lifted_single_predicate_span_skips_where_keyword() -> None:
    # Single-predicate WHERE: the synthesized atom span must align with the
    # predicate text, not the WHERE keyword (the +len("WHERE ") off-by-six case).
    q = "MATCH (a)-[]->(b) WHERE a.x = b.y RETURN a"
    w = cast(CypherQuery, parse_cypher(q)).where
    assert w is not None and len(w.predicates) == 1
    left = cast(PropertyRef, w.predicates[0].left)
    right = w.predicates[0].right
    assert left.span.column == q.index("a.x") + 1
    assert isinstance(right, PropertyRef)
    assert right.span.column == q.index("b.y") + 1
