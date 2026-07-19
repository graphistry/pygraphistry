"""Regression: openCypher NULL ordering in the general row-pipeline ORDER BY.

openCypher orders NULL as the LARGEST value:
  - ``ORDER BY x ASC``  -> NULLs LAST
  - ``ORDER BY x DESC`` -> NULLs FIRST

The general row pipeline previously hardcoded NULLs-last for every key (pandas/cuDF
``sort_values`` defaulting ``na_position='last'``; polars ``order_by_polars`` passing
``nulls_last=True``), so any ``ORDER BY <k> DESC`` over a column containing NULLs was
mis-ordered on BOTH pandas and polars and ``DESC ... LIMIT k`` silently returned the wrong
top-k (it dropped the NULL group, which should rank first). Oracle here is HAND-DERIVED
openCypher truth, NOT parity-vs-parent (the parity suites had no NULL in the ordered key,
which is how this survived).
"""
import pandas as pd
import pytest

import graphistry

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover - depends on test env
    _HAS_CUDF = False

_ENGINES = ["pandas", "polars"] + (["cudf"] if _HAS_CUDF else [])

# city groups: A:3 (0,3,7)  B:2 (1,4)  None:2 (2,5)  C:1 (6)
_NODES = pd.DataFrame(
    {"id": [0, 1, 2, 3, 4, 5, 6, 7], "city": ["A", "B", None, "A", "B", None, "C", "A"]}
)
_EDGES = pd.DataFrame({"s": [0, 1], "d": [1, 2], "eid": [0, 1]})


def _g():
    return graphistry.nodes(_NODES, "id").edges(_EDGES, "s", "d").bind(edge="eid")


def _recs(nodes):
    """Rows as list-of-dicts with NaN/pd.NA normalized to None (engine-agnostic)."""
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    out = []
    for rec in nodes.to_dict("records"):
        out.append({k: (None if pd.isna(v) else v) for k, v in rec.items()})
    return out


@pytest.mark.parametrize("engine", _ENGINES)
def test_grouped_desc_limit_puts_null_group_first(engine):
    # openCypher DESC: NULL largest -> [None(2), C(1), B(2), A(3)]; LIMIT 2 -> [None, C].
    r = _g().gfql(
        "MATCH (m) RETURN m.city AS city, count(m) AS c ORDER BY city DESC LIMIT 2",
        engine=engine,
    )._nodes
    assert _recs(r) == [{"city": None, "c": 2}, {"city": "C", "c": 1}]


@pytest.mark.parametrize("engine", _ENGINES)
def test_grouped_asc_limit_keeps_null_group_last(engine):
    # openCypher ASC: NULLs last -> [A(3), B(2), C(1), None(2)]; LIMIT 2 -> [A, B].
    r = _g().gfql(
        "MATCH (m) RETURN m.city AS city, count(m) AS c ORDER BY city ASC LIMIT 2",
        engine=engine,
    )._nodes
    assert _recs(r) == [{"city": "A", "c": 3}, {"city": "B", "c": 2}]


@pytest.mark.parametrize("engine", _ENGINES)
def test_scalar_desc_orders_nulls_first(engine):
    # DESC over scalar column: NULL rows first, then C, B, A (stable within group by id).
    r = _g().gfql(
        "MATCH (m) RETURN m.id AS id, m.city AS city ORDER BY city DESC",
        engine=engine,
    )._nodes
    cities = [rec["city"] for rec in _recs(r)]
    assert cities == [None, None, "C", "B", "B", "A", "A", "A"]


@pytest.mark.parametrize("engine", _ENGINES)
def test_scalar_asc_orders_nulls_last(engine):
    r = _g().gfql(
        "MATCH (m) RETURN m.id AS id, m.city AS city ORDER BY city ASC",
        engine=engine,
    )._nodes
    cities = [rec["city"] for rec in _recs(r)]
    assert cities == ["A", "A", "A", "B", "B", "C", None, None]
