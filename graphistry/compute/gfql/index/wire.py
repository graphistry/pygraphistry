"""JSON wire protocol for GFQL index DDL + the executor that applies an op.

DDL ops are top-level program types (peers of Chain/Let), following the GFQL AST
JSON convention ``{"type": ClassName, ...fields}``. They round-trip via
``to_json``/``from_json`` and are dispatched by ``index_op_from_json``.

    {"type": "CreateIndex", "kind": "edge_out_adj", "column": null, "name": null, "replace": false}
    {"type": "DropIndex",   "name": "edge_out_adj:src", "missing_ok": true}   # true = IF EXISTS (no-op when missing; default false = raise)
    {"type": "DropIndex",   "kind": "edge_in_adj", "column": "dst", "missing_ok": true}
    {"type": "ShowIndexes"}

``index_policy`` rides in the request envelope (peer of ``engine``), handled by
``gfql(index_policy=...)`` — it is NOT part of these op shapes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union, cast

from .registry import ALL_KINDS
from .types import IndexKind


@dataclass(frozen=True)
class CreateIndex:
    kind: IndexKind
    column: Optional[str] = None
    name: Optional[str] = None
    replace: bool = False

    def to_json(self) -> Dict[str, Any]:
        return {"type": "CreateIndex", "kind": self.kind, "column": self.column,
                "name": self.name, "replace": self.replace}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "CreateIndex":
        kind = d.get("kind")
        if kind not in ALL_KINDS:
            raise ValueError(f"CreateIndex.kind must be one of {ALL_KINDS}, got {kind!r}")
        return CreateIndex(kind=cast(IndexKind, kind), column=d.get("column"), name=d.get("name"),
                           replace=bool(d.get("replace", False)))


@dataclass(frozen=True)
class DropIndex:
    name: Optional[str] = None
    kind: Optional[IndexKind] = None
    column: Optional[str] = None
    missing_ok: bool = False  # IF EXISTS semantics: True = dropping a missing index is a no-op

    def to_json(self) -> Dict[str, Any]:
        return {"type": "DropIndex", "name": self.name, "kind": self.kind,
                "column": self.column, "missing_ok": self.missing_ok}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "DropIndex":
        return DropIndex(name=d.get("name"), kind=d.get("kind"), column=d.get("column"),
                         missing_ok=bool(d.get("missing_ok", False)))


@dataclass(frozen=True)
class ShowIndexes:
    def to_json(self) -> Dict[str, Any]:
        return {"type": "ShowIndexes"}

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "ShowIndexes":
        return ShowIndexes()


INDEX_OP_TYPES = ("CreateIndex", "DropIndex", "ShowIndexes")
IndexOp = Union[CreateIndex, DropIndex, ShowIndexes]


def is_index_op(obj: Any) -> bool:
    return isinstance(obj, (CreateIndex, DropIndex, ShowIndexes))


def is_index_op_json(d: Any) -> bool:
    return isinstance(d, dict) and d.get("type") in INDEX_OP_TYPES


def index_op_from_json(d: Dict[str, Any]) -> IndexOp:
    t = d.get("type")
    if t == "CreateIndex":
        return CreateIndex.from_json(d)
    if t == "DropIndex":
        return DropIndex.from_json(d)
    if t == "ShowIndexes":
        return ShowIndexes.from_json(d)
    raise ValueError(f"Not a GFQL index op: type={t!r}")


def apply_index_op(g: Any, op: IndexOp, *, engine: Any = "auto") -> Any:
    """Execute a DDL op against a Plottable's index registry.

    CreateIndex/DropIndex -> new Plottable; ShowIndexes -> pandas DataFrame.
    """
    from .api import create_index, drop_index, show_indexes, get_registry, _is_resident_index_valid

    if isinstance(op, CreateIndex):
        if not op.replace:
            reg = get_registry(g)
            if reg.has(op.kind) and _is_resident_index_valid(g, op.kind, engine):
                return g  # valid resident index reuse
        return create_index(g, op.kind, column=op.column, name=op.name, engine=engine)
    if isinstance(op, DropIndex):
        kind = op.kind
        if kind is None and op.name is not None:
            # Resolve a (possibly custom) index NAME to its kind by searching the
            # registry's index.name — NOT by splitting the name on ':' (that only
            # recovered the default ``kind:col`` name → a custom name silently no-op'd).
            reg = get_registry(g)
            kind = next((k for k, ix in reg.indexes.items()
                         if getattr(ix, "name", None) == op.name), None)
            if kind is None:
                if op.missing_ok:
                    return g  # IF EXISTS semantics: dropping a missing index is a no-op
                raise ValueError(
                    f"DROP GFQL INDEX: no resident index named {op.name!r} "
                    f"(resident: {sorted(getattr(ix, 'name', k) for k, ix in reg.indexes.items())})"
                )
        if kind is not None and not op.missing_ok and not get_registry(g).has(kind):
            raise ValueError(f"DROP GFQL INDEX: no resident index of kind {kind!r}")
        return drop_index(g, kind)
    if isinstance(op, ShowIndexes):
        return show_indexes(g)
    raise ValueError(f"Unknown index op: {op!r}")
