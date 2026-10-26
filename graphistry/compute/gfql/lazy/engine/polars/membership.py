"""Version-stable id-set membership (``Expr.is_in`` against a ``Series``) for the native
polars GFQL engine.

polars changed the ``is_in`` right-hand-side contract in 1.28.0:

* ``< 1.28``: a bare ``Series`` RHS is set membership. A List-typed RHS is matched ROW-WISE and
  must have the column's length, so the length-1 ``ids.implode()`` raises
  ``ComputeError: shapes don't match: expected N elements in 'is_in' comparison, got 1``
  (RAPIDS 25.02 pins polars 1.21 through cudf-polars).
* ``>= 1.28``: a bare same-dtype ``Series`` RHS emits ``DeprecationWarning`` ("ambiguous ...
  use implode") and the imploded RHS is the supported spelling.

ONE helper so the spelling cannot drift per call site. Boundary pinned by a per-release sweep
over polars 1.21.0..1.35.2: the imploded RHS fails through 1.27.1 and passes from 1.28.0; the
bare RHS passes on every release and warns from 1.28.0.

The ``< 1.28`` branch is NOT dead code even though the ``polars`` extra declares
``polars>=1.29`` (setup.py): the supported RAPIDS 25.02 environment pins polars 1.21 through
cudf-polars, and the polars CPU engine must run there too.
"""
from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from graphistry.compute.gfql.cache_registry import register_process_singleton

if TYPE_CHECKING:
    import polars as pl

IMPLODED_RHS_FLOOR = Version("1.28.0")


def imploded_rhs_supported(polars_version: str) -> bool:
    """True when ``is_in(ids.implode())`` is the correct, warning-free spelling.

    An unparseable version falls back to the bare-Series RHS, which is CORRECT on every
    release 1.21..1.35 and merely warns from 1.28 -- the safe side of the branch."""
    try:
        return Version(polars_version) >= IMPLODED_RHS_FLOOR
    except InvalidVersion:
        return False


@lru_cache(maxsize=1)
def _installed_polars_implodes() -> bool:
    import polars as pl
    return imploded_rhs_supported(pl.__version__)


register_process_singleton(
    _installed_polars_implodes,
    "is_in RHS spelling for the installed polars; a function of the environment, not of caller input")


def id_set(ids: "pl.Series") -> "pl.Series":
    """``ids`` as the ``is_in`` right-hand side for the installed polars: one List row
    (``implode()``) on >= 1.28, the flat Series below. Hoist it when the same set is tested
    against several columns."""
    if _installed_polars_implodes():
        return ids.implode()
    return ids


def is_in_ids(expr: "pl.Expr", ids: "pl.Series") -> "pl.Expr":
    """``expr`` is a member of the id set ``ids`` — spelled for the installed polars.

    Set semantics on every supported version: nulls in ``expr`` are null (callers keep their
    own ``fill_null(False)``), a null in ``ids`` matches nothing, an empty ``ids`` matches nothing.
    """
    return expr.is_in(id_set(ids))
