from typing import Optional, Sequence, TypeVar

from graphistry.models.collections import (
    CollectionIntersection,
    CollectionExprInput,
    CollectionSet,
)

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
    from graphistry.compute.ast import normalize_gfql_to_wire
    collection: CollectionSet = {"type": "set", "expr": {"type": "gfql_chain", "gfql": normalize_gfql_to_wire(expr)}}
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
