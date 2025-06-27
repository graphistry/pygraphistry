from typing import TypedDict
from graphistry.Engine import DataframeLike
from graphistry.Plottable import Plottable

class HypergraphResult(TypedDict):
    entities: DataframeLike
    events: DataframeLike
    edges: DataframeLike
    nodes: DataframeLike
    graph: Plottable
