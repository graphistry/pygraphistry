import operator
from typing import Any, Callable, Dict, Iterable, List, Set, Tuple

import pandas as pd
import pytest

from graphistry.compute.predicates.comparison import ge, gt, le, lt


OpCase = Tuple[str, Callable[[Any], Any], Callable[[Any, Any], Any]]
Backend = str

OPS: Tuple[OpCase, ...] = (
    ("gt", gt, operator.gt),
    ("lt", lt, operator.lt),
    ("ge", ge, operator.ge),
    ("le", le, operator.le),
)

_cudf_runtime_ok: bool | None = None


SAFE_SCALAR_CASES: Tuple[Dict[str, Any], ...] = (
    {
        "id": "numeric_scalar_numeric_rhs",
        "values": [-1, 0, 2, None],
        "rhs": 0,
        "dtype": "float64",
        "backends": {"pandas", "cudf"},
    },
    {
        "id": "string_scalar_string_rhs",
        "values": ["aa", "te", "text", None],
        "rhs": "te",
        "dtype": "object",
        "backends": {"pandas", "cudf"},
    },
    {
        "id": "numeric_scalar_string_rhs_incomparable",
        "values": [0, 1, 2, None],
        "rhs": "1",
        "dtype": "float64",
        "backends": {"pandas", "cudf"},
    },
    {
        "id": "string_scalar_numeric_rhs_incomparable",
        "values": ["0", "2", "10", None],
        "rhs": 1,
        "dtype": "object",
        "backends": {"pandas", "cudf"},
    },
    {
        "id": "bool_scalar_string_rhs_incomparable",
        "values": [True, False, None, True],
        "rhs": "x",
        "dtype": "object",
        "backends": {"pandas", "cudf"},
    },
    {
        # pandas-only mixed object row domain: exercises per-row TypeError fallback.
        "id": "mixed_object_scalar_string_rhs",
        "values": ["text", 0, None, 3.5, "aa"],
        "rhs": "te",
        "dtype": "object",
        "backends": {"pandas"},
    },
)


def _safe_row_compare(cell: Any, rhs: Any, py_op: Callable[[Any, Any], Any]) -> bool:
    if pd.isna(cell):
        return False
    try:
        out = py_op(cell, rhs)
    except TypeError:
        return False
    if pd.isna(out):
        return False
    return bool(out)


def _to_backend_series(values: Iterable[Any], dtype: str, backend: Backend):
    s_pd = pd.Series(list(values), dtype=dtype)
    if backend == "pandas":
        return s_pd
    if backend == "cudf":
        cudf = pytest.importorskip("cudf")
        global _cudf_runtime_ok
        if _cudf_runtime_ok is None:
            try:
                _ = cudf.Series([1, 2, 3])
                _cudf_runtime_ok = True
            except Exception:
                _cudf_runtime_ok = False
        if not _cudf_runtime_ok:
            pytest.skip("cudf installed but runtime is unavailable (no usable GPU/VRAM)")
        return cudf.from_pandas(s_pd)
    raise AssertionError(f"Unexpected backend: {backend}")


def _mask_to_bool_list(mask: Any) -> List[bool]:
    if hasattr(mask, "to_pandas"):
        mask = mask.to_pandas()
    out: List[bool] = []
    for v in mask.tolist():
        out.append(False if pd.isna(v) else bool(v))
    return out


@pytest.mark.parametrize("backend", ["pandas", "cudf"])
@pytest.mark.parametrize("case", SAFE_SCALAR_CASES, ids=[c["id"] for c in SAFE_SCALAR_CASES])
@pytest.mark.parametrize("op_name,pred_factory,py_op", OPS, ids=[o[0] for o in OPS])
def test_comparison_safe_scalar_conformance_matrix(
    backend: Backend,
    case: Dict[str, Any],
    op_name: str,
    pred_factory: Callable[[Any], Any],
    py_op: Callable[[Any, Any], Any],
) -> None:
    del op_name
    allowed_backends: Set[str] = case["backends"]
    if backend not in allowed_backends:
        pytest.skip(f"{case['id']} is scoped to backends={sorted(allowed_backends)}")

    s = _to_backend_series(case["values"], case["dtype"], backend)
    predicate = pred_factory(case["rhs"])
    actual = _mask_to_bool_list(predicate(s))
    expected = [_safe_row_compare(v, case["rhs"], py_op) for v in case["values"]]
    assert actual == expected


@pytest.mark.parametrize("case", [c for c in SAFE_SCALAR_CASES if c["backends"] == {"pandas", "cudf"}], ids=[c["id"] for c in SAFE_SCALAR_CASES if c["backends"] == {"pandas", "cudf"}])
@pytest.mark.parametrize("op_name,pred_factory,_", OPS, ids=[o[0] for o in OPS])
def test_comparison_safe_scalar_cudf_pandas_parity(
    case: Dict[str, Any],
    op_name: str,
    pred_factory: Callable[[Any], Any],
    _: Callable[[Any, Any], Any],
) -> None:
    del op_name
    s_pd = _to_backend_series(case["values"], case["dtype"], "pandas")
    s_cudf = _to_backend_series(case["values"], case["dtype"], "cudf")
    predicate = pred_factory(case["rhs"])

    pd_mask = _mask_to_bool_list(predicate(s_pd))
    cudf_mask = _mask_to_bool_list(predicate(s_cudf))
    assert cudf_mask == pd_mask


@pytest.mark.parametrize(
    "op_name,pred_factory,expected",
    (
        ("gt", gt, [False, True, False]),
        ("lt", lt, [True, False, False]),
        ("ge", ge, [False, True, False]),
        ("le", le, [True, False, False]),
    ),
    ids=["gt", "lt", "ge", "le"],
)
@pytest.mark.parametrize("backend", ["pandas", "cudf"])
def test_comparison_temporal_datetime_conformance(
    op_name: str,
    pred_factory: Callable[[Any], Any],
    expected: List[bool],
    backend: Backend,
) -> None:
    del op_name
    s_pd = pd.Series(
        [
            pd.Timestamp("2024-01-01 00:00:00"),
            pd.Timestamp("2024-01-03 00:00:00"),
            pd.NaT,
        ],
        dtype="datetime64[ns]",
    )
    s = _to_backend_series(s_pd.tolist(), "datetime64[ns]", backend)
    rhs = pd.Timestamp("2024-01-02 00:00:00+00:00")

    actual = _mask_to_bool_list(pred_factory(rhs)(s))
    assert actual == expected
