from typing import Dict, List, Optional, Sequence, TypeVar

from graphistry.models.collections import (
    CollectionIntersection,
    CollectionExprInput,
    CollectionSet,
)
from graphistry.utils.json import JSONVal

CollectionDict = TypeVar("CollectionDict", CollectionSet, CollectionIntersection)


def _apply_collection_metadata(collection: CollectionDict, **metadata: Optional[str]) -> CollectionDict:
    value = metadata.get("id")
    if value is not None:
        collection["id"] = value
    value = metadata.get("name")
    if value is not None:
        collection["name"] = value
    value = metadata.get("description")
    if value is not None:
        collection["description"] = value
    value = metadata.get("node_color")
    if value is not None:
        collection["node_color"] = value
    value = metadata.get("edge_color")
    if value is not None:
        collection["edge_color"] = value
    return collection


def _wrap_gfql_expr(expr: CollectionExprInput) -> Dict[str, JSONVal]:

    from graphistry.compute.ast import ASTObject, from_json as ast_from_json
    from graphistry.compute.chain import Chain

    def _normalize_op(op: object) -> Dict[str, JSONVal]:
        if isinstance(op, ASTObject):
            return op.to_json()
        if isinstance(op, dict):
            return ast_from_json(op, validate=True).to_json()
        raise TypeError("Collection GFQL operations must be AST objects or dictionaries")

    def _normalize_ops(raw: object) -> List[Dict[str, JSONVal]]:
        if isinstance(raw, Chain):
            return _normalize_ops(raw.to_json().get("chain", []))
        if isinstance(raw, ASTObject):
            return [raw.to_json()]
        if isinstance(raw, list):
            if len(raw) == 0:
                raise ValueError("Collection GFQL operations list cannot be empty")
            return [_normalize_op(op) for op in raw]
        if isinstance(raw, dict):
            if raw.get("type") == "Chain" and "chain" in raw:
                return _normalize_ops(raw.get("chain"))
            if raw.get("type") == "gfql_chain" and "gfql" in raw:
                return _normalize_ops(raw.get("gfql"))
            if "chain" in raw:
                return _normalize_ops(raw.get("chain"))
            if "gfql" in raw:
                return _normalize_ops(raw.get("gfql"))
            return [_normalize_op(raw)]
        raise TypeError("Collection expr must be an AST object, chain, list, or dict")

    return {"type": "gfql_chain", "gfql": _normalize_ops(expr)}


def collection_set(
    *,
    expr: CollectionExprInput,
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    node_color: Optional[str] = None,
    edge_color: Optional[str] = None,
) -> CollectionSet:
    """Build a collection dict for a GFQL-defined set."""
    collection: CollectionSet = {"type": "set", "expr": _wrap_gfql_expr(expr)}
    return _apply_collection_metadata(
        collection,
        id=id,
        name=name,
        description=description,
        node_color=node_color,
        edge_color=edge_color,
    )


def collection_intersection(
    *,
    sets: Sequence[str],
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    node_color: Optional[str] = None,
    edge_color: Optional[str] = None,
) -> CollectionIntersection:
    """Build a collection dict for an intersection of set IDs."""
    collection: CollectionIntersection = {
        "type": "intersection",
        "expr": {
            "type": "intersection",
            "sets": list(sets),
        },
    }
    return _apply_collection_metadata(
        collection,
        id=id,
        name=name,
        description=description,
        node_color=node_color,
        edge_color=edge_color,
    )
