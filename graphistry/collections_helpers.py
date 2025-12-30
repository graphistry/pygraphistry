from typing import Any, Dict, List, Optional, Sequence, TypedDict, Union


class CollectionSet(TypedDict, total=False):
    """Collection definition for a GFQL-defined set."""
    type: str
    id: str
    name: str
    description: str
    node_color: str
    edge_color: str
    expr: Any


class CollectionIntersection(TypedDict, total=False):
    """Collection definition for an intersection of sets."""
    type: str
    id: str
    name: str
    description: str
    node_color: str
    edge_color: str
    expr: Dict[str, Any]


Collection = Union[CollectionSet, CollectionIntersection]
CollectionsInput = Union[str, Collection, List[Collection], Dict[str, Any], List[Dict[str, Any]]]


def _apply_collection_metadata(
    collection: Dict[str, Any],
    id: Optional[str],
    name: Optional[str],
    description: Optional[str],
    node_color: Optional[str],
    edge_color: Optional[str],
) -> Dict[str, Any]:
    if id is not None:
        collection["id"] = id
    if name is not None:
        collection["name"] = name
    if description is not None:
        collection["description"] = description
    if node_color is not None:
        collection["node_color"] = node_color
    if edge_color is not None:
        collection["edge_color"] = edge_color
    return collection


def collection_set(
    *,
    expr: Any,
    id: Optional[str] = None,
    name: Optional[str] = None,
    description: Optional[str] = None,
    node_color: Optional[str] = None,
    edge_color: Optional[str] = None,
) -> CollectionSet:
    """Build a collection dict for a GFQL-defined set."""
    collection: Dict[str, Any] = {"type": "set", "expr": expr}
    return _apply_collection_metadata(collection, id, name, description, node_color, edge_color)  # type: ignore[return-value]


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
    collection: Dict[str, Any] = {
        "type": "intersection",
        "expr": {
            "type": "intersection",
            "sets": list(sets),
        },
    }
    return _apply_collection_metadata(collection, id, name, description, node_color, edge_color)  # type: ignore[return-value]
