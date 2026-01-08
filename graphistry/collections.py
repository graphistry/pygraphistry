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

    from graphistry.compute.ast import ASTLet, ASTObject, from_json as ast_from_json
    from graphistry.compute.chain import Chain

    def _normalize_op(op: object) -> Dict[str, JSONVal]:
        if isinstance(op, ASTLet):
            raise TypeError("Collection GFQL does not support Let/DAG expressions")
        if isinstance(op, ASTObject):
            return op.to_json()
        if isinstance(op, dict):
            if op.get("type") == "Let" or "bindings" in op:
                raise TypeError("Collection GFQL does not support Let/DAG expressions")
            parsed = ast_from_json(op, validate=True)
            if isinstance(parsed, ASTLet):
                raise TypeError("Collection GFQL does not support Let/DAG expressions")
            return parsed.to_json()
        raise TypeError("Collection GFQL operations must be AST objects or dictionaries")

    def _normalize_ops_value(raw: object) -> List[Dict[str, JSONVal]]:
        if isinstance(raw, list):
            if len(raw) == 0:
                raise ValueError("Collection GFQL operations list cannot be empty")
            return [_normalize_op(op) for op in raw]
        return [_normalize_op(raw)]

    if isinstance(expr, dict):
        expr_type = expr.get("type")
        if expr_type == "gfql_chain" and "gfql" in expr:
            gfql_ops = expr.get("gfql")
            return {"type": "gfql_chain", "gfql": _normalize_ops_value(gfql_ops)}
        if expr_type == "Chain" and "chain" in expr:
            return {"type": "gfql_chain", "gfql": _normalize_ops_value(expr.get("chain"))}
        if "gfql" in expr:
            gfql_ops = expr.get("gfql")
            return {"type": "gfql_chain", "gfql": _normalize_ops_value(gfql_ops)}
        if "chain" in expr:
            return {"type": "gfql_chain", "gfql": _normalize_ops_value(expr.get("chain"))}
        return {"type": "gfql_chain", "gfql": _normalize_ops_value(expr)}

    if isinstance(expr, Chain):
        chain_json = expr.to_json()
        return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}

    if isinstance(expr, ASTObject):
        return {"type": "gfql_chain", "gfql": [expr.to_json()]}

    if isinstance(expr, list):
        return {"type": "gfql_chain", "gfql": _normalize_ops_value(expr)}

    raise TypeError("Collection expr must be an AST object, chain, list, or dict")


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
