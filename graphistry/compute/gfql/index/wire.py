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
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, cast

from typing_extensions import TypeGuard

from .registry import ALL_KINDS
from .errors import is_not_yet_implemented_kind, not_implemented_kind_error
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
        if is_not_yet_implemented_kind(kind):
            raise not_implemented_kind_error(kind, ALL_KINDS)
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


if TYPE_CHECKING:
    from graphistry.compute.ast import ASTCall


def is_index_op(obj: object) -> TypeGuard[IndexOp]:
    return isinstance(obj, (CreateIndex, DropIndex, ShowIndexes))


def is_index_op_json(d: object) -> TypeGuard[Dict[str, Any]]:
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
    from .api import (
        create_index, drop_index, show_indexes, get_registry,
        _is_resident_index_valid, resolve_index_engine,
    )

    from .registry import NODE_PROP, EDGE_PROP, PROPERTY_ROLES

    if isinstance(op, CreateIndex):
        if not op.replace:
            reg = get_registry(g)
            if op.kind in (NODE_PROP, EDGE_PROP):
                # Property indexes are keyed by COLUMN, not kind: reuse only the
                # index for THIS column, and only while it is still valid.
                if op.column is not None and reg.get_property_valid(
                    "nodes" if op.kind == NODE_PROP else "edges", op.column,
                    g._nodes if op.kind == NODE_PROP else g._edges,
                    resolve_index_engine(engine, g),
                ) is not None:
                    return g
            elif reg.has(op.kind) and _is_resident_index_valid(g, op.kind, engine):
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
            column = op.column
            if kind is None:
                for prop_kind, role in PROPERTY_ROLES:
                    prop_col = next((c for c, ix in reg.property_indexes(role).items() if ix.name == op.name), None)
                    if prop_col is not None:
                        kind, column = prop_kind, prop_col
                        break
            if kind is None:
                if op.missing_ok:
                    return g  # IF EXISTS semantics: dropping a missing index is a no-op
                resident = sorted(
                    [getattr(ix, 'name', k) for k, ix in reg.indexes.items()]
                    + [ix.name or c for _, role in PROPERTY_ROLES for c, ix in reg.property_indexes(role).items()]
                )
                raise ValueError(
                    f"DROP GFQL INDEX: no resident index named {op.name!r} "
                    f"(resident: {resident})"
                )
            return drop_index(g, kind, column=column)
        if kind is not None and not op.missing_ok:
            reg = get_registry(g)
            if kind in (NODE_PROP, EDGE_PROP):
                props = reg.property_indexes("nodes" if kind == NODE_PROP else "edges")
                is_resident = op.column in props if op.column is not None else bool(props)
            else:
                is_resident = reg.has(kind)
            if not is_resident:
                raise ValueError(f"DROP GFQL INDEX: no resident index of kind {kind!r}")
        return drop_index(g, kind, column=op.column)
    if isinstance(op, ShowIndexes):
        return show_indexes(g, engine=engine)
    raise ValueError(f"Unknown index op: {op!r}")


def index_op_to_call(op: IndexOp) -> "ASTCall":
    """The ``call()`` op equivalent of an index DDL op, so DDL can sit in a chain or a ``let`` binding.

    ``CreateIndex`` -> ``call('create_index', ...)``, ``DropIndex`` by kind -> ``call('drop_index', ...)``.
    ``ShowIndexes`` answers a table, not a graph, and a drop by NAME has no method form; both are
    rejected with the standalone form they do have.
    """
    from graphistry.compute.ast import ASTCall
    from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

    if isinstance(op, CreateIndex):
        params: Dict[str, Any] = {"kind": op.kind}
        if op.column is not None:
            params["column"] = op.column
        if op.name is not None:
            params["name"] = op.name
        return ASTCall("create_index", params)
    if isinstance(op, DropIndex):
        if op.name is not None:
            raise GFQLTypeError(
                ErrorCode.E201,
                "DropIndex by name cannot sit in a chain; drop by kind (DropIndex(kind=...)) or run g.gfql(DropIndex(name=...)) on its own",
                field="chain", value="DropIndex",
            )
        params = {}
        if op.kind is not None:
            params["kind"] = op.kind
        if op.column is not None:
            params["column"] = op.column
        return ASTCall("drop_index", params)
    raise GFQLTypeError(
        ErrorCode.E201,
        "ShowIndexes answers a table, not a graph, so it cannot sit in a chain; call g.show_indexes() or g.gfql(ShowIndexes()) on its own",
        field="chain", value="ShowIndexes",
    )
