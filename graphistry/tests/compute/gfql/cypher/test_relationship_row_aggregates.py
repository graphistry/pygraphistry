"""Cypher aggregates over relationship-pattern rows.

``MATCH (a)-[r]->(b)`` yields one row per matched path, so a node bound by several
edges appears several times. Aggregates whose group key and arguments reference any
node or edge alias run on those per-path binding rows. Each case is checked against
a brute-force path enumeration on seeded random multigraphs with parallel edges,
self-loops and null properties, using Cypher null rules (sum/avg/min/max/count(expr)
and collect skip nulls).
"""
import math
import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

ENGINES = ["pandas", pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars"))]
Path = Dict[str, Dict[Any, Any]]


def _graph(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    ids = [f"n{i}" for i in range(rng.randint(1, 6))]
    nodes = pd.DataFrame({
        "id": ids,
        "kind": [rng.choice(["A", "B"]) for _ in ids],
        "score": pd.Series([rng.choice([1.0, 2.0, 5.0, None]) for _ in ids], dtype="float64"),
    })
    m = rng.randint(0, 9)
    edges = pd.DataFrame({  # explicit dtypes so an edgeless graph keeps string endpoints
        "s": pd.Series([rng.choice(ids) for _ in range(m)], dtype="object"),
        "d": pd.Series([rng.choice(ids) for _ in range(m)], dtype="object"),
        "w": pd.Series([rng.choice([1.0, 2.0, 4.0, None]) for _ in range(m)], dtype="float64"),
        "eid": pd.Series(list(range(m)), dtype="int64"),
    })
    return nodes, edges


def _paths(nodes: pd.DataFrame, edges: pd.DataFrame, undirected: bool = False) -> List[Path]:
    by_id = {r["id"]: r for r in nodes.to_dict("records")}
    out: List[Path] = []
    for e in edges.to_dict("records"):
        out.append({"a": by_id[e["s"]], "r": e, "b": by_id[e["d"]]})
        if undirected and e["s"] != e["d"]:
            out.append({"a": by_id[e["d"]], "r": e, "b": by_id[e["s"]]})
    return out


def _vals(rows: Sequence[Path], alias: str, prop: str) -> List[Any]:
    return [p[alias][prop] for p in rows if not _is_null(p[alias][prop])]


def _is_null(v: Any) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _sum(xs: List[Any]) -> Any:
    return sum(xs) if xs else 0


def _avg(xs: List[Any]) -> Any:
    return sum(xs) / len(xs) if xs else None


def _grouped(paths: List[Path], key: Callable[[Path], Any], aggs: Sequence[Callable[[List[Path]], Any]]) -> List[tuple]:
    groups: Dict[Any, List[Path]] = defaultdict(list)
    for p in paths:
        groups[key(p)].append(p)
    rows = []
    for k, v in groups.items():
        rows.append((k if isinstance(k, tuple) else (k,)) + tuple(f(v) for f in aggs))
    return rows


A_ID: Callable[[Path], Any] = lambda p: p["a"]["id"]  # noqa: E731

CASES: List[Tuple[str, Callable[[pd.DataFrame, pd.DataFrame], List[tuple]], bool]] = [
    # (query, oracle, polars_may_decline)
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, count(*) AS c",
     lambda n, e: _grouped(_paths(n, e), A_ID, [len]), False),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, count(r) AS c",
     lambda n, e: _grouped(_paths(n, e), A_ID, [len]), True),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, count(r.w) AS c",
     lambda n, e: _grouped(_paths(n, e), A_ID, [lambda v: len(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, sum(r.w) AS s, min(r.w) AS lo, max(r.w) AS hi, avg(r.w) AS m",
     lambda n, e: _grouped(_paths(n, e), A_ID, [
         lambda v: _sum(_vals(v, "r", "w")),
         lambda v: min(_vals(v, "r", "w"), default=None),
         lambda v: max(_vals(v, "r", "w"), default=None),
         lambda v: _avg(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, collect(b.id) AS xs, collect(r.w) AS ws",
     lambda n, e: _grouped(_paths(n, e), A_ID, [
         lambda v: sorted(_vals(v, "b", "id")), lambda v: sorted(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, count(DISTINCT b) AS c, collect(DISTINCT b.kind) AS ks",
     lambda n, e: _grouped(_paths(n, e), A_ID, [
         lambda v: len({p["b"]["id"] for p in v}), lambda v: sorted(set(_vals(v, "b", "kind")))]), False),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, count(DISTINCT r) AS c",
     lambda n, e: _grouped(_paths(n, e), A_ID, [lambda v: len({p["r"]["eid"] for p in v})]), True),
    ("MATCH (a)-[r]->(b) RETURN a.id AS k, b.id AS k2, count(*) AS c, sum(r.w) AS s",
     lambda n, e: _grouped(_paths(n, e), lambda p: (p["a"]["id"], p["b"]["id"]),
                           [len, lambda v: _sum(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) RETURN b.kind AS k, sum(a.score) AS s, count(*) + 1 AS c1",
     lambda n, e: _grouped(_paths(n, e), lambda p: p["b"]["kind"],
                           [lambda v: _sum(_vals(v, "a", "score")), lambda v: len(v) + 1]), False),
    ("MATCH (a)-[r]-(b) RETURN a.id AS k, count(*) AS c, sum(r.w) AS s",
     lambda n, e: _grouped(_paths(n, e, undirected=True), A_ID,
                           [len, lambda v: _sum(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) WHERE r.w >= 2 RETURN a.id AS k, sum(r.w) AS s",
     lambda n, e: _grouped([p for p in _paths(n, e) if not _is_null(p["r"]["w"]) and p["r"]["w"] >= 2],
                           A_ID, [lambda v: _sum(_vals(v, "r", "w"))]), False),
    ("MATCH (a)-[r]->(b) WITH a, count(DISTINCT b) AS c RETURN a.id AS k, c",
     lambda n, e: _grouped(_paths(n, e), A_ID, [lambda v: len({p["b"]["id"] for p in v})]), False),
]


def _cell(v: Any) -> Any:
    if hasattr(v, "tolist") and not isinstance(v, (str, bytes)):
        v = v.tolist()
    if isinstance(v, (list, tuple)):
        return tuple(sorted((_cell(x) for x in v), key=repr))
    if _is_null(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return int(v)
    return v


def _rows(df: Any) -> List[tuple]:
    if not hasattr(df, "iloc"):
        df = df.to_pandas()
    return sorted((tuple(_cell(v) for v in r) for r in df.itertuples(index=False)), key=repr)


def _run(nodes: pd.DataFrame, edges: pd.DataFrame, query: str, engine: str) -> Any:
    g: Any
    if engine == "polars":
        g = graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    else:
        g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    return g.gfql(query, engine=engine)._nodes


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query, oracle, polars_may_decline", CASES, ids=[c[0] for c in CASES])
def test_relationship_row_aggregate_matches_path_oracle(engine: str, query: str, oracle: Any, polars_may_decline: bool) -> None:
    for seed in range(25):
        nodes, edges = _graph(seed)
        expected = sorted((tuple(_cell(v) for v in r) for r in oracle(nodes, edges)), key=repr)
        try:
            got = _rows(_run(nodes, edges, query, engine))
        except NotImplementedError:
            assert engine == "polars" and polars_may_decline, f"unexpected decline seed={seed}"
            continue
        assert got == expected, f"seed={seed}\nnodes={nodes.to_dict('records')}\nedges={edges.to_dict('records')}"


def test_whole_row_group_counts_paths_on_pandas() -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "kind": ["A", "B", "B"]})
    edges = pd.DataFrame({"s": ["a", "a", "a", "b"], "d": ["b", "b", "c", "c"]})
    out = _run(nodes, edges, "MATCH (a)-[r]->(b) RETURN a, count(*) AS c", "pandas")
    assert out.sort_values("a.id").to_dict("records") == [
        {"a.id": "a", "a.kind": "A", "c": 3},
        {"a.id": "b", "a.kind": "B", "c": 1},
    ]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (a)-[r]->(b) RETURN a.id AS k, a.kind + count(b) AS c",
    "OPTIONAL MATCH (a)-[r]->(b) RETURN a.id AS k, sum(r.w) AS s",
    "MATCH (a)-[r]->(b) RETURN a.id AS k, collect(b) AS xs",
])
def test_ambiguous_or_optional_shapes_keep_structured_errors(engine: str, query: str) -> None:
    nodes, edges = _graph(3)
    with pytest.raises(GFQLValidationError):
        _run(nodes, edges, query, engine)




# ---- unit pins for the per-path route's decline branches (each keeps the older lowering) ----

def _gate_inputs(text: str = "a.id", agg_text: Any = None, agg_func: str = "count") -> Any:
    from dataclasses import replace

    from graphistry.compute.ast import e_forward, n
    from graphistry.compute.gfql.cypher.ast import CypherQuery, ExpressionText, ReturnItem
    from graphistry.compute.gfql.cypher.lowering import _AggregateSpec
    from graphistry.compute.gfql.cypher.parser import parse_cypher

    query = parse_cypher("MATCH (a)-[r]->(b) RETURN a.id AS k, count(*) AS c")
    assert isinstance(query, CypherQuery)
    span = query.return_.items[0].span
    item = ReturnItem(ExpressionText(text, span), "k", span)
    query = replace(query, return_=replace(query.return_, items=(item,) + query.return_.items[1:]))
    spec = _AggregateSpec("agg", "c", agg_func, agg_text, False, 1, 1)
    targets = {"a": n(name="a"), "r": e_forward(name="r"), "b": n(name="b")}
    return query, [spec], [item], targets


def _gate(text: str = "a.id", agg_text: Any = None, agg_func: str = "count") -> bool:
    from graphistry.compute.gfql.cypher.aggregate_bindings import per_path_aggregate_bindings_apply

    query, specs, items, targets = _gate_inputs(text, agg_text, agg_func)
    return per_path_aggregate_bindings_apply(
        query, aggregate_specs=specs, non_aggregate_items=items, alias_targets=targets, params=None,
    )


def test_gate_engages_for_a_clean_relationship_aggregate() -> None:
    assert _gate("a.id", "r.w", "sum") is True


@pytest.mark.parametrize("text, agg_text, agg_func", [
    ("r", None, "count"),            # whole-row group on an edge alias
    ("a.id ~~ b.x", None, "count"),  # unanalyzable group key
    ("a.id", "r.w ~~ 1", "sum"),     # unanalyzable aggregate argument
    ("a.id", "b", "collect"),        # whole-entity collect keeps the entity path
])
def test_gate_declines_shapes_it_does_not_own(text: str, agg_text: Any, agg_func: str) -> None:
    assert _gate(text, agg_text, agg_func) is False


def test_mixed_ref_check_treats_unparseable_items_as_mixed() -> None:
    from graphistry.compute.gfql.cypher.aggregate_bindings import _clause_mixes_group_and_aggregate_refs

    query, _, _, targets = _gate_inputs("a.id ~~ b.x")
    assert _clause_mixes_group_and_aggregate_refs(query, alias_targets=targets, params=None) is True


@pytest.mark.parametrize("query, expected", [
    # variable-length 1..2 trails: a->b (2 parallel edges), a->b->c (x2), b->c, b->c->d, b->c->a, c->d, c->a, c->a->b (x2)
    ("MATCH (a)-[*1..2]->(b) RETURN a.id AS k, count(*) AS c", [("a", 4), ("b", 3), ("c", 4)]),
    ("MATCH (a)-[*1..2]->(b) RETURN a.id AS k, count(DISTINCT b) AS c", [("a", 2), ("b", 3), ("c", 3)]),
    ("MATCH (a)-[*1..2]->(b) RETURN a.id AS k, collect(b.id) AS xs",
     [("a", ("b", "b", "c", "c")), ("b", ("a", "c", "d")), ("c", ("a", "b", "b", "d"))]),
    ("MATCH (a {k:'X'})-[r]->(b) RETURN a.id AS k, sum(r.w) AS s", [("a", 3), ("c", 24)]),
])
def test_variable_length_and_filtered_relationship_aggregates(query: str, expected: List[tuple]) -> None:
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "k": ["X", "Y", "X", "Y"]})
    edges = pd.DataFrame({"s": ["a", "a", "b", "c", "c"], "d": ["b", "b", "c", "d", "a"], "w": [1.0, 2.0, 4.0, 8.0, 16.0]})
    assert _rows(_run(nodes, edges, query, "pandas")) == sorted(expected, key=repr)
