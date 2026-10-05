"""Exceptions for GFQL index building."""
from __future__ import annotations

from typing import Literal, Tuple

from typing_extensions import TypeGuard

INDEX_SUPPORT_ISSUE_URL = "https://github.com/graphistry/pygraphistry/issues/2141"
NotYetImplementedKind = Literal["edge_prop"]
NOT_YET_IMPLEMENTED_KINDS: Tuple[NotYetImplementedKind, ...] = ("edge_prop",)


class GfqlIndexUnsupportedError(ValueError):
    """The DATA cannot support this index (duplicate node ids, an unindexable
    property dtype). Distinct from a caller mistake — a missing column, an unknown
    kind, unbound edges — which stays a plain ``ValueError`` and must propagate.

    Subclasses ``ValueError`` so existing ``except ValueError`` callers keep
    working; the convenience builders catch only THIS type, so a real failure is
    never silently skipped.
    """


class GfqlIndexNotImplementedError(GfqlIndexUnsupportedError, NotImplementedError):
    """Index support GFQL does not have yet: an edge property index, or a property
    column type other than null-free integers. Tracked in ``INDEX_SUPPORT_ISSUE_URL``."""


def is_not_yet_implemented_kind(kind: object) -> TypeGuard[NotYetImplementedKind]:
    return isinstance(kind, str) and kind in NOT_YET_IMPLEMENTED_KINDS


def not_implemented_kind_error(kind: NotYetImplementedKind, supported: Tuple[str, ...]) -> GfqlIndexNotImplementedError:
    return GfqlIndexNotImplementedError(
        f"GFQL does not support a {kind!r} index yet. Supported kinds: {supported}. "
        f"Tracked in {INDEX_SUPPORT_ISSUE_URL}"
    )
