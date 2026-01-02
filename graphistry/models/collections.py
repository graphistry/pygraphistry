from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING, Union
from typing_extensions import Literal, NotRequired, Required, TypedDict

from graphistry.utils.json import JSONVal

if TYPE_CHECKING:
    from graphistry.compute.ast import ASTObject
    from graphistry.compute.chain import Chain


GFQLWireOp = Dict[str, JSONVal]


class GFQLChainWire(TypedDict):
    type: Literal["gfql_chain"]
    gfql: List[GFQLWireOp]


class GFQLChainJSON(TypedDict):
    type: Literal["Chain"]
    chain: List[GFQLWireOp]


GFQLOpInput = Union["ASTObject", GFQLWireOp]
GFQLChainInput = Union[
    "Chain",
    "ASTObject",
    GFQLChainWire,
    GFQLChainJSON,
    List[GFQLOpInput],
    GFQLWireOp,
]


class IntersectionExpr(TypedDict):
    type: Literal["intersection"]
    sets: List[str]


class CollectionBase(TypedDict, total=False):
    id: str
    name: str
    description: str
    node_color: str
    edge_color: str


class CollectionSet(CollectionBase):
    type: NotRequired[Literal["set"]]
    expr: Required[GFQLChainInput]


class CollectionIntersection(CollectionBase):
    type: NotRequired[Literal["intersection"]]
    expr: Required[IntersectionExpr]


Collection = Union[CollectionSet, CollectionIntersection]
CollectionsInput = Union[str, Collection, List[Collection]]
