"""Version-stable id-set membership (``Expr.is_in`` against a ``Series``) for the native
polars GFQL engine.

polars changed the ``is_in`` right-hand-side contract in 1.28.0:

* ``< 1.28``: a bare ``Series`` RHS is set membership. A List-typed RHS is matched ROW-WISE and
  must have the column's length, so the length-1 ``ids.implode()`` raises
  ``ComputeError: shapes don't match: expected N elements in 'is_in' comparison, got 1``
  (#2082 — RAPIDS 25.02 pins polars 1.21 through cudf-polars).
* ``>= 1.28``: a bare same-dtype ``Series`` RHS emits ``DeprecationWarning`` ("ambiguous ...
  use implode"; #1938 item 5) and the imploded RHS is the supported spelling.

ONE helper so the spelling cannot drift per call site. Boundary pinned by the
plans/gfql-2082-polars121 sweep over polars 1.21.0..1.35.2: the imploded RHS fails through
1.27.1 and passes from 1.28.0; the bare RHS passes on every version and warns from 1.28.0.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from packaging.version import Version

if TYPE_CHECKING:
    import polars as pl

IMPLODED_RHS_FLOOR = Version("1.28.0")


def imploded_rhs_supported(polars_version: str) -> bool:
    """True when ``is_in(ids.implode())`` is the correct, warning-free spelling."""
    return Version(polars_version) >= IMPLODED_RHS_FLOOR


@lru_cache(maxsize=1)
def _installed_polars_implodes() -> bool:
    import polars as pl
    return imploded_rhs_supported(pl.__version__)


def is_in_ids(expr: "pl.Expr", ids: "pl.Series") -> "pl.Expr":
    """``expr`` is a member of the id set ``ids`` — spelled for the installed polars.

    Set semantics on every supported version: nulls in ``expr`` are null (callers keep their
    own ``fill_null(False)``), a null in ``ids`` matches nothing, an empty ``ids`` matches nothing.
    """
    if _installed_polars_implodes():
        return expr.is_in(ids.implode())
    return expr.is_in(ids)
