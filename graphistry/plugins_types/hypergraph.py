from typing import TYPE_CHECKING, TypedDict
from graphistry.Engine import DataframeLike

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

class HypergraphResult(TypedDict):
    entities: DataframeLike
    events: DataframeLike
    edges: DataframeLike
    nodes: DataframeLike
    graph: 'Plottable'
