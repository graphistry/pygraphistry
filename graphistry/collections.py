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

    from graphistry.compute.ast import ASTObject
    from graphistry.compute.chain import Chain

    def _gfql_chain_from_ops(ops: Sequence[ASTObject]) -> Dict[str, JSONVal]:
        chain_json = Chain(list(ops)).to_json()
        return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}

    def _gfql_chain_from_wire_ops(ops: List[Dict[str, JSONVal]]) -> Dict[str, JSONVal]:
        chain = Chain.from_json({"chain": ops}, validate=True)
        chain_json = chain.to_json()
        return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}

    def _list_to_wire_ops(raw: Sequence[object]) -> List[Dict[str, JSONVal]]:
        ops: List[Dict[str, JSONVal]] = []
        for op in raw:
            if isinstance(op, ASTObject):
                ops.append(op.to_json())
            elif isinstance(op, dict):
                ops.append(op)
            else:
                raise TypeError("Collection GFQL operations must be AST objects or dictionaries")
        return ops

    def _wrap_ops_value(raw: object) -> List[Dict[str, JSONVal]]:
        if isinstance(raw, list):
            return _list_to_wire_ops(raw)
        if isinstance(raw, (ASTObject, dict)):
            return _list_to_wire_ops([raw])
        raise TypeError("Collection GFQL operations must be a list, AST object, or dictionary")

    if isinstance(expr, dict):
        expr_type = expr.get("type")
        if expr_type == "gfql_chain" and "gfql" in expr:
            gfql_ops = expr.get("gfql")
            return _gfql_chain_from_wire_ops(_wrap_ops_value(gfql_ops))
        if expr_type == "Chain" and "chain" in expr:
            chain = Chain.from_json(expr, validate=True)
            chain_json = chain.to_json()
            return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}
        if "gfql" in expr:
            gfql_ops = expr.get("gfql")
            return _gfql_chain_from_wire_ops(_wrap_ops_value(gfql_ops))
        if "chain" in expr:
            chain = Chain.from_json(expr, validate=True)
            chain_json = chain.to_json()
            return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}
        return _gfql_chain_from_wire_ops([expr])

    if isinstance(expr, Chain):
        chain_json = expr.to_json()
        return {"type": "gfql_chain", "gfql": chain_json.get("chain", [])}

    if isinstance(expr, ASTObject):
        return {"type": "gfql_chain", "gfql": [expr.to_json()]}

    if isinstance(expr, list):
        ops_ast = [op for op in expr if isinstance(op, ASTObject)]
        if len(ops_ast) == len(expr):
            return _gfql_chain_from_ops(ops_ast)
        return _gfql_chain_from_wire_ops(_list_to_wire_ops(expr))

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
