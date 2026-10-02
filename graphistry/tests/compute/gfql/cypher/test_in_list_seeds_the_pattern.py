"""``WHERE alias.prop IN [scalar literals]`` seeds the MATCH pattern as an ``is_in`` filter, the way
``alias.prop = literal`` already does, so the seeded hop lanes and the resident index see the seed
instead of a post-join row filter. Lists carrying null (three-valued verdict) and predicates under
OR / NOT stay in the row WHERE.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import ASTCall, ASTEdge, ASTNode
from graphistry.compute.predicates.is_in import IsIn


def _lowered(query: str, params: Optional[Dict[str, Any]] = None) -> List[Any]:
    from graphistry.compute.gfql.cypher import compile_cypher

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return list(compile_cypher(query, params=params).chain.chain)


def _where_rows_exprs(ops: List[Any]) -> List[str]:
    return [op.params["expr"] for op in ops if isinstance(op, ASTCall) and op.function == "where_rows"]


def _is_in_values(predicate: Any) -> List[Any]:
    assert isinstance(predicate, IsIn)
    return [getattr(option, "value", option) for option in predicate.options]


def test_a_literal_list_on_the_seed_alias_becomes_the_pattern_filter() -> None:
    ops = _lowered("MATCH (a)-[e]->(b) WHERE a.id IN [1, 2] RETURN b")
    seed = ops[0]
    assert isinstance(seed, ASTNode) and seed.filter_dict is not None
    assert _is_in_values(seed.filter_dict["id"]) == [1, 2]
    assert _where_rows_exprs(ops) == []
    rows = [op for op in ops if isinstance(op, ASTCall) and op.function == "rows"][0]
    assert not rows.params.get("alias_prefilters")


def test_a_parameter_list_is_a_literal_list() -> None:
    ops = _lowered("MATCH (a)-[e]->(b) WHERE a.id IN $ids RETURN b", params={"ids": ["x", "y"]})
    assert _is_in_values(ops[0].filter_dict["id"]) == ["x", "y"]
    assert _where_rows_exprs(ops) == []


def test_only_the_membership_conjunct_moves_out_of_an_and() -> None:
    ops = _lowered("MATCH (a)-[e]->(b) WHERE a.id IN [1, 2] AND b.kind = 'person' RETURN b")
    assert _is_in_values(ops[0].filter_dict["id"]) == [1, 2]
    exprs = _where_rows_exprs(ops)
    assert len(exprs) == 1 and "a.id" not in exprs[0] and "b.kind" in exprs[0]


def test_an_edge_alias_list_becomes_the_edge_match() -> None:
    ops = _lowered("MATCH (a)-[e]->(b) WHERE e.w IN [1, 3] RETURN b")
    edge = ops[1]
    assert isinstance(edge, ASTEdge) and edge.edge_match is not None
    assert _is_in_values(edge.edge_match["w"]) == [1, 3]


@pytest.mark.parametrize(
    "where",
    [
        "a.id IN [1, null]",
        "a.id IN [1] OR b.id IN [2]",
        "NOT a.id IN [1]",
        "a.id IN [[1, 2]]",
        "[a.id] IN [[1]]",
        "a.id IN b.tags",
        "a.ts IN [datetime('2020-03-04T00:00:00')]",
        "a.ts IN ['2020-03-04T00:00:00Z', '2020-01-01T00:00:00+02:00']",
    ],
    ids=["null element", "under OR", "under NOT", "nested list", "list-valued left side", "list held by another alias",
         "temporal constructor (lowers to zoned ISO text, compared as an instant)", "zoned ISO text"],
)
def test_three_valued_and_structural_forms_stay_in_the_row_where(where: str) -> None:
    ops = _lowered(f"MATCH (a)-[e]->(b) WHERE {where} RETURN b")
    assert all(not op.filter_dict for op in ops if isinstance(op, ASTNode))
    assert all(not op.edge_match for op in ops if isinstance(op, ASTEdge))
    exprs = _where_rows_exprs(ops)
    assert len(exprs) == 1 and " IN " in exprs[0]


def _graph(ids: List[Any]) -> tuple:
    rng = np.random.default_rng(11)
    n = len(ids)
    nodes = pd.DataFrame({"id": ids, "kind": rng.choice(["person", "company"], n)})
    edges = pd.DataFrame({
        "src": [ids[i] for i in rng.integers(0, n, 12 * n)],
        "dst": [ids[i] for i in rng.integers(0, n, 12 * n)],
        "w": rng.integers(0, 4, 12 * n),
    })
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    joined = (edges.merge(nodes.add_prefix("a_"), left_on="src", right_on="a_id")
              .merge(nodes.add_prefix("b_"), left_on="dst", right_on="b_id"))
    return g, joined


@pytest.mark.parametrize("ids", [list(range(1_000)), [f"v{i}" for i in range(1_000)]], ids=["int ids", "str ids"])
def test_seeded_results_equal_the_row_filter_results(ids: List[Any]) -> None:
    g, m = _graph(ids)
    seeds = [ids[i] for i in (4, 42, 420, 999)]

    def rows(mask: Any) -> List[Any]:
        return sorted(m[mask]["b_id"].tolist())

    cases = [
        (f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds!r} RETURN b", rows(m["src"].isin(seeds))),
        (f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds!r} AND b.kind = 'person' RETURN b",
         rows(m["src"].isin(seeds) & (m["b_kind"] == "person"))),
        (f"MATCH (a {{id: {seeds[0]!r}}})-[e]->(b) WHERE e.w IN [1, 3] RETURN b",
         rows((m["src"] == seeds[0]) & m["w"].isin([1, 3]))),
        (f"MATCH (a {{id: {seeds[0]!r}}})-[e]->(b) WHERE a.id IN {seeds[:2]!r} RETURN b", rows(m["src"] == seeds[0])),
        (f"MATCH (a {{id: {seeds[0]!r}}})-[e]->(b) WHERE a.id IN {seeds[1:3]!r} RETURN b", []),
        (f"MATCH (a)-[e]->(b) WHERE a.id IN {seeds[:2]!r} OR b.id IN {seeds[2:]!r} RETURN b",
         rows(m["src"].isin(seeds[:2]) | m["dst"].isin(seeds[2:]))),
    ]
    for query, expected in cases:
        assert sorted(g.gfql(query)._nodes["b.id"].tolist()) == expected, query
    assert sorted(g.gfql("MATCH (a)-[e]->(b) WHERE a.id IN $ids RETURN b", params={"ids": seeds})._nodes["b.id"].tolist()) \
        == rows(m["src"].isin(seeds))


def test_optional_match_keeps_its_null_row_with_a_seeded_list() -> None:
    nodes = pd.DataFrame({"id": [1, 2, 3]})
    edges = pd.DataFrame({"src": [1], "dst": [2]})
    g = graphistry.edges(edges, "src", "dst").nodes(nodes, "id")
    out = g.gfql("MATCH (a) WHERE a.id IN [1, 3] OPTIONAL MATCH (a)-[e]->(b) RETURN a.id AS a, b.id AS b ORDER BY a")._nodes
    assert out["a"].tolist() == [1, 3]
    assert out["b"].tolist()[0] == 2 and pd.isna(out["b"].tolist()[1])


def test_an_alias_outside_the_pattern_is_left_for_the_where_to_report() -> None:
    from graphistry.compute.ast import e_forward, n
    from graphistry.compute.gfql.cypher.ast import SourceSpan
    from graphistry.compute.gfql.cypher.where_membership import _literal_membership_seed

    targets = {"a": n(name="a"), "e": e_forward(name="e"), "b": n(name="b")}
    span = SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=0, end_pos=0)
    assert _literal_membership_seed("z.id IN [1]", span=span, alias_targets=targets, params=None) is None
    assert _literal_membership_seed("a.id IN [1]", span=span, alias_targets=targets, params=None) is not None
