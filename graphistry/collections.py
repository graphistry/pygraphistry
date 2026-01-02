from typing import List, Optional, Sequence, TypeVar, cast

from graphistry.models.collections import (
    CollectionIntersection,
    CollectionSet,
    GFQLChainInput,
    GFQLChainWire,
    GFQLWireOp,
)

CollectionDict = TypeVar("CollectionDict", CollectionSet, CollectionIntersection)


def _apply_collection_metadata(collection: CollectionDict, **metadata: Optional[str]) -> CollectionDict:
    collection.update({key: value for key, value in metadata.items() if value is not None})
    return collection


def _wrap_gfql_expr(expr: GFQLChainInput) -> GFQLChainWire:

    from graphistry.compute.ast import ASTObject
    from graphistry.compute.chain import Chain

    if isinstance(expr, dict):
        expr_type = expr.get("type")
        if expr_type == "gfql_chain" and "gfql" in expr:
            return expr
        if expr_type == "Chain" and "chain" in expr:
            return {"type": "gfql_chain", "gfql": expr.get("chain", [])}
        if "gfql" in expr:
            return {"type": "gfql_chain", "gfql": expr.get("gfql")}
        if "chain" in expr:
            return {"type": "gfql_chain", "gfql": expr.get("chain")}
        return {"type": "gfql_chain", "gfql": [expr]}

    if isinstance(expr, Chain):
        return {"type": "gfql_chain", "gfql": expr.to_json().get("chain", [])}

    if isinstance(expr, ASTObject):
        return {"type": "gfql_chain", "gfql": [expr.to_json()]}

    if isinstance(expr, list):
        ops: List[GFQLWireOp] = []
        for op in expr:
            if isinstance(op, ASTObject):
                ops.append(op.to_json())
            else:
                ops.append(cast(GFQLWireOp, op))
        return {"type": "gfql_chain", "gfql": ops}

    return {"type": "gfql_chain", "gfql": [cast(GFQLWireOp, expr)]}


def collection_set(
    *,
    expr: GFQLChainInput,
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
