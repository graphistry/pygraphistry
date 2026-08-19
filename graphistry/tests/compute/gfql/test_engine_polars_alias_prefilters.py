"""#1804: the native polars bindings builder must HONOUR ``rows(alias_prefilters=...)``.

At master e6625ed28 the polars boundary called ``binding_rows_polars`` with
``binding_ops`` and ``attach_prop_aliases`` only, so a caller-set ``alias_prefilters``
was silently DISCARDED and the filtered-out rows came back (12 rows / max(a.id)=8 where
pandas and cuDF answer 5 / 2) — the worst silent-wrong class. The builder now applies
the specs natively per alias (``_apply_alias_prefilters_polars``) at the same points as
the pandas builder: seed alias, per-hop edge alias, endpoint alias, the node-cartesian
mode, and the single-entity pattern-apply shape. The flipped strict xfail lives in
``test_rewrite_param_discard.py``; this file pins per-surface VALUE parity against the
pandas oracle plus the typed decline.

DECLINE CONTRACT: a spec the polars lowering cannot serve raises a NotImplementedError
NAMING ``alias_prefilters`` — never a silent drop, never a pandas bridge. (On the Cypher
surface the lowering that emits these prefilters always keeps the equivalent post-join
filter, and that post-join op declines on the same shapes, so the earlier typed NIE does
not retire any previously-served Cypher query.)

Anti-vacuity: every filtered pin also asserts the UNFILTERED count, so a prefilter that
stops narrowing (the master defect) fails on the number, not on schema noise.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import ASTObject, e_forward, n, rows, serialize_binding_ops
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason

ENGINES = ("pandas", "cudf", "polars")

NODES = pd.DataFrame({
    "id": np.arange(1, 13, dtype=np.int64),
    "kind": ["seed", "mid", "mid", "end", "end", "tail",
             "tail", "reverse", "seed", "noise", "noise", "noise"],
    # float column: the polars searchAny lowering declines float stringification
    # (repr diverges from pandas in the exponent regime), giving the typed-NIE pin below.
    "score": np.linspace(0.0, 1.1, 12),
})
EDGES = pd.DataFrame({
    "src": [1, 1, 1, 2, 2, 3, 4, 5, 6, 8, 4, 5],
    "dst": [2, 2, 3, 4, 5, 5, 6, 6, 7, 1, 2, 3],
    "type": ["A", "A", "A", "B", "B", "B", "C", "C", "D", "REV", "B", "B"],
    "weight": np.arange(10, 22, dtype=np.int64),
})
ALL_EDGES = len(EDGES)

MIDDLE: List[ASTObject] = [n(name="a"), e_forward(name="r"), n(name="b")]
BOPS = serialize_binding_ops(MIDDLE)


def _graph(engine: str) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(
            pl.from_pandas(EDGES), "src", "dst")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(NODES), "id").edges(
            cudf.from_pandas(EDGES), "src", "dst")
    return graphistry.nodes(NODES, "id").edges(EDGES, "src", "dst")


@lru_cache(maxsize=None)
def _engine_skip_reason(engine: str) -> Optional[str]:
    def smoke() -> None:
        # NOT the shape under test: no alias_prefilters anywhere in the smoke.
        _graph(engine).gfql([n({"id": 1}), e_forward(), n()], engine=engine)

    return engine_skip_reason(engine, smoke)


def _require(engine: str) -> None:
    reason = _engine_skip_reason(engine)
    if reason is not None:
        pytest.skip(f"engine {engine!r} unavailable here ({reason}) — NOT evidence of passing")


def _run(engine: str, ops: List[ASTObject]) -> pd.DataFrame:
    out = _graph(engine).gfql(list(ops), engine=engine)._nodes
    assert out is not None
    return out.to_pandas() if hasattr(out, "to_pandas") else out


def _pairs(frame: pd.DataFrame, cols: List[str]) -> List[tuple]:
    return sorted(map(tuple, frame[cols].values.tolist()))


# Each case: (prefilters, hand-computed filtered row count, columns whose values must
# match pandas). Counts are hand-derived from the 12-edge fixture, so a prefilter that
# silently stops narrowing (master's polars defect: 12 rows back) fails on the number.
PREFILTER_CASES = {
    "seed_expr": ({"a": [{"kind": "expr", "text": "a.id < 3"}]}, 5, ["a.id", "b"]),
    "endpoint_expr": ({"b": [{"kind": "expr", "text": "b.kind = 'end'"}]}, 3, ["a.id", "b"]),
    "edge_expr": ({"r": [{"kind": "expr", "text": "r.weight >= 20"}]}, 2, ["a.id", "b", "r.weight"]),
    "seed_search_any": ({"a": [{"kind": "search_any", "term": "seed"}]}, 3, ["a.id", "b"]),
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", sorted(PREFILTER_CASES))
def test_connected_prefilter_narrows_and_matches_pandas(engine: str, case: str) -> None:
    _require(engine)
    prefilters, expected_rows, value_cols = PREFILTER_CASES[case]
    unfiltered = _run(engine, [rows(binding_ops=BOPS)])
    filtered = _run(engine, [rows(binding_ops=BOPS, alias_prefilters=prefilters)])
    assert len(unfiltered) == ALL_EDGES  # anti-vacuity: the unnarrowed bag is the full bag
    assert len(filtered) == expected_rows, (
        f"[{engine}/{case}] prefilter narrowed to {len(filtered)} rows, expected "
        f"{expected_rows} (a discarded prefilter returns {ALL_EDGES})"
    )
    if engine != "pandas":
        oracle = _run("pandas", [rows(binding_ops=BOPS, alias_prefilters=prefilters)])
        assert _pairs(filtered, value_cols) == _pairs(oracle, value_cols), (
            f"[{engine}/{case}] filtered VALUES diverge from the pandas oracle"
        )


@pytest.mark.parametrize("engine", ENGINES)
def test_cartesian_prefilter_narrows_and_matches_pandas(engine: str) -> None:
    """Disconnected MATCH (a), (b): the node-cartesian mode must honour prefilters too.

    12x12 cross-product; ``a.id < 3`` keeps 2x12 = 24 rows.
    """
    _require(engine)
    cart = serialize_binding_ops([n(name="a"), n(name="b")])
    pref = {"a": [{"kind": "expr", "text": "a.id < 3"}]}
    unfiltered = _run(engine, [rows(binding_ops=cart)])
    filtered = _run(engine, [rows(binding_ops=cart, alias_prefilters=pref)])
    assert len(unfiltered) == 144
    assert len(filtered) == 24
    assert sorted(filtered["a"].unique().tolist()) == [1, 2]


@pytest.mark.parametrize("engine", ENGINES)
def test_single_entity_prefilter_narrows_and_keeps_the_layout(engine: str) -> None:
    """The single named-Node shape (the EXISTS-pipeline left table) honours prefilters
    WITHOUT changing its column layout between the filtered and unfiltered spellings."""
    _require(engine)
    single = serialize_binding_ops([n(name="a")])
    pref = {"a": [{"kind": "expr", "text": "a.id < 3"}]}
    unfiltered = _run(engine, [rows(binding_ops=single)])
    filtered = _run(engine, [rows(binding_ops=single, alias_prefilters=pref)])
    assert len(unfiltered) == len(NODES)
    assert len(filtered) == 2
    assert sorted(filtered["a"].tolist()) == [1, 2]
    assert sorted(map(str, filtered.columns)) == sorted(map(str, unfiltered.columns))


def test_polars_unlowerable_prefilter_declines_typed_naming_the_feature() -> None:
    """A spec polars cannot lower raises a typed NIE NAMING alias_prefilters.

    searchAny over an explicit FLOAT column is the deterministic decline (float
    stringification diverges from the pandas kernel, same gate as the post-join
    ``search_any`` op). The error must name the feature and the alias — never a
    silent drop, never a raw polars exception.
    """
    _require("polars")
    pref = {"a": [{"kind": "search_any", "term": "1", "columns": ["score"]}]}
    with pytest.raises(NotImplementedError, match=r"alias_prefilters.*'a'"):
        _run("polars", [rows(binding_ops=BOPS, alias_prefilters=pref)])
