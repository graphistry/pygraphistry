"""#1739: HAS_<Label> destination disambiguation on DUPLICATE-id node tables must hold
on the single-hop grouped-aggregate fast path, on every engine.

THE DEFECT (red at master e6625ed28). The Cypher lowering of an LDBC-style multi-label
graph collides node ids across label tables (the same id carries a Tag row and a Forum
row). ``MATCH (post:Post {id})-[:HAS_TAG]->(tag) RETURN tag.title, count(post)`` serves via
``_execute_single_hop_grouped_aggregate_fast_path``, whose lanes join the ``tag`` property
lookup by node id:

* the polars lanes joined WITHOUT any dedup or label narrowing, so the colliding id
  attached BOTH rows' properties and the aggregate emitted an extra group
  (``[{f4,1},{t4,1}]`` where the oracle says ``[{t4,1}]``);
* the pandas/cuDF lane deduped ``keep='first'`` — the oracle answer only when the
  label-true row happens to come first in the frame (order-dependent, not narrowing).

THE ORACLE is hand-computed and matches the pandas ROW PIPELINE
(``_gfql_disambiguate_has_edge_destination_nodes``): when the candidate destination ids of
a forward ``HAS_<Label>``-typed hop COLLIDE and the destination alias carries no label
filter, the destination domain narrows to the ``label__<Label>``-true rows. Unique
candidate ids (even on a duplicate-id table) do NOT narrow, and a labeled destination
filter disables narrowing — both sides pinned below so the fix cannot over-narrow.

ENGINES: pandas + polars run everywhere this lane runs; cudf skips with a stated reason
where the GPU stack is absent (never silently green). polars-gpu is exercised out of band
(see test_rewrite_param_discard's COVERAGE BOUNDARY note).

The grouped property column is ``title``, not ``name``: a node property literally named
``name`` trips a PRE-EXISTING, unrelated cuDF crash in the fast path's pandas-API lane
(``grouped.size().reset_index(name=...)`` collides with the column; ValueError at master
e6625ed28 too) — do not rename it back, or the cudf params test that quirk instead of
the #1739 narrowing.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd
import pytest

import graphistry
from graphistry.tests.compute.gfql.polars_test_utils import engine_skip_reason

ENGINES = ("pandas", "cudf", "polars")

# Node id 400 collides (Tag row 't4' + Forum row 'f4'); 500 collides too but is never a
# destination, so it proves the CANDIDATE-scoped probe (a full-table probe alone must not
# narrow the 500 rows' queries). Edge 601-HAS_TAG->400 is the colliding hop.
_BASE: Dict[str, List[Any]] = {
    "id": [600, 601, 201, 300, 400, 400, 500, 500],
    "label__Post": [True, True, False, False, False, False, False, False],
    "label__Tag": [False, False, True, False, True, False, True, False],
    "label__Forum": [False, False, False, True, False, True, False, True],
    "title": [None, None, "t1", "f-node", "t4", "f4", "t5", "f5"],
}
EDGES = pd.DataFrame({
    "src": [600, 600, 601],
    "dst": [201, 300, 400],
    "type": ["HAS_TAG", "HAS_TAG", "HAS_TAG"],
})

Q_COLLIDING = (
    "MATCH (post:Post {id: 601})-[:HAS_TAG]->(tag) "
    "RETURN tag.title AS tagName, count(post) AS c ORDER BY tagName"
)
Q_UNIQUE_CANDIDATES = (
    "MATCH (post:Post {id: 600})-[:HAS_TAG]->(tag) "
    "RETURN tag.title AS tagName, count(post) AS c ORDER BY tagName"
)
Q_LABELED_DEST = (
    "MATCH (post:Post {id: 601})-[:HAS_TAG]->(tag:Forum) "
    "RETURN tag.title AS tagName, count(post) AS c ORDER BY tagName"
)


def _nodes(order: str) -> pd.DataFrame:
    """``tag_first`` = the fixture as written (label-true row first for id 400);
    ``forum_first`` swaps the two id-400 rows — the order that unmasked the pandas
    lane's keep='first' accident at master."""
    df = pd.DataFrame(_BASE)
    if order == "forum_first":
        idx = list(range(len(df)))
        idx[4], idx[5] = idx[5], idx[4]
        df = df.iloc[idx].reset_index(drop=True)
    return df


def _graph(engine: str, nodes: pd.DataFrame) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(
            pl.from_pandas(EDGES), "src", "dst")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(EDGES), "src", "dst")
    return graphistry.nodes(nodes, "id").edges(EDGES, "src", "dst")


@lru_cache(maxsize=None)
def _engine_skip_reason(engine: str) -> Optional[str]:
    def smoke() -> None:
        # NOT the shape under test (no HAS_ type, no duplicate destination).
        _graph(engine, _nodes("tag_first")).gfql(
            "MATCH (a:Post {id: 600}) RETURN a.id AS i", engine=engine)

    return engine_skip_reason(engine, smoke)


def _require(engine: str) -> None:
    reason = _engine_skip_reason(engine)
    if reason is not None:
        pytest.skip(f"engine {engine!r} unavailable here ({reason}) — NOT evidence of passing")


def _records(engine: str, nodes: pd.DataFrame, query: str) -> List[Dict[str, Any]]:
    out = _graph(engine, nodes).gfql(query, engine=engine)._nodes
    if hasattr(out, "collect"):
        out = out.collect()
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("order", ["tag_first", "forum_first"])
def test_colliding_destination_narrows_to_the_label_row(engine: str, order: str) -> None:
    """The colliding hop answers the label-narrowed row set — in BOTH frame orders.

    Red at master: polars emitted ``[{f4,1},{t4,1}]`` in both orders; pandas emitted
    ``[{f4,1}]`` in ``forum_first`` (keep='first' kept the Forum row). One group, the
    Tag row, is the only answer the disambiguation contract allows.
    """
    _require(engine)
    assert _records(engine, _nodes(order), Q_COLLIDING) == [{"tagName": "t4", "c": 1}]


@pytest.mark.parametrize("engine", ENGINES)
def test_unique_candidate_ids_do_not_narrow(engine: str) -> None:
    """ANTI-OVER-NARROWING: unique candidate ids keep every destination row.

    Seeded from post 600 the candidate destinations are {201, 300} — unique, even though
    the TABLE has colliding ids (400, 500). The pandas oracle does not narrow here, so
    the non-Tag destination ``f-node`` must survive; a mutant that narrows on the
    full-table probe (or unconditionally by label) drops it and fails this test.
    """
    _require(engine)
    assert _records(engine, _nodes("tag_first"), Q_UNIQUE_CANDIDATES) == [
        {"tagName": "f-node", "c": 1},
        {"tagName": "t1", "c": 1},
    ]


@pytest.mark.parametrize("engine", ENGINES)
def test_labeled_destination_filter_disables_narrowing(engine: str) -> None:
    """An explicit destination label wins over the edge type's implied label.

    ``->(tag:Forum)`` filters the destination itself, so the HAS_TAG narrowing must NOT
    apply (pandas oracle: ``_gfql_node_filter_has_label`` short-circuits) and the Forum
    row is the answer.
    """
    _require(engine)
    assert _records(engine, _nodes("tag_first"), Q_LABELED_DEST) == [{"tagName": "f4", "c": 1}]


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_fast_path_still_serves_the_colliding_shape(engine: str) -> None:
    """ANTI-VACUITY: the fix narrows INSIDE the fast path rather than declining it.

    Every value test above passes equally if the fast path silently retires and the
    query re-serves on the canonical route (or, on polars, would NIE) — so pin the
    engagement itself, via the public trace rather than a monkeypatch (see
    engagement.py on why patching private names fails open).
    """
    _require(engine)
    from graphistry.tests.compute.gfql.engagement import assert_fast_path

    assert_fast_path(
        _graph(engine, _nodes("tag_first")), Q_COLLIDING,
        "single_hop_grouped_aggregate", served=True, engine=engine,
    )
