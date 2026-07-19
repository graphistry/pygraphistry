"""Regression: CASE with branches of different dtypes must be engine-consistent.

pandas coerces the two CASE branches to a common type, but cuDF's `.where` raises
`TypeError: cudf does not support mixed types`, which surfaced as a hard
`GFQLTypeError` (e.g. `CASE WHEN path IS NULL THEN -1 ELSE length(path) END` over an
UNREACHABLE shortestPath: the hops branch is an object/null column, the other is int
-1). The evaluator now unifies the branch dtypes and retries, so cuDF returns the same
value pandas does instead of raising. This general repro uses an all-null object column
(same mechanism, no shortestPath dependency).
"""
import pandas as pd
import pytest

import graphistry

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover
    _HAS_CUDF = False

_ENGINES = ["pandas"] + (["cudf"] if _HAS_CUDF else [])

_N = pd.DataFrame({"id": [0, 1, 2], "x": pd.Series([None, None, None], dtype="object")})
_E = pd.DataFrame({"s": [0], "d": [1], "eid": [0]})


@pytest.mark.parametrize("engine", _ENGINES)
def test_case_mixed_dtype_branches_do_not_raise(engine):
    # int THEN branch vs all-null object ELSE branch: cuDF `.where` used to raise
    # "cudf does not support mixed types"; must now return -1 for every row like pandas.
    g = graphistry.nodes(_N, "id").edges(_E, "s", "d").bind(edge="eid")
    r = g.gfql(
        "MATCH (m) RETURN m.id AS id, CASE WHEN m.x IS NULL THEN -1 ELSE m.x END AS r ORDER BY id",
        engine=engine,
    )._nodes
    if hasattr(r, "to_pandas"):
        r = r.to_pandas()
    # value-equal across engines (cuDF coerces to float -1.0; pandas keeps int -1)
    assert [rec["id"] for rec in r.to_dict("records")] == [0, 1, 2]
    assert all(float(rec["r"]) == -1.0 for rec in r.to_dict("records"))
