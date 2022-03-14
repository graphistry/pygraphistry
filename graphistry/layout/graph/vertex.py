# DEPRECRATED: Non-vector operators over non-vectorized data

from graphistry.layout.utils import Rectangle
from .vertexBase import VertexBase


class Vertex(VertexBase):
    """
       Vertex class enhancing a VertexBase with graph-related features.

       **Attributes**
            component (GraphBase): the component of connected vertices that contains this vertex. By default, a vertex belongs no component but when it is added in a graph, c points to the connected component in this graph.
            data (object) : an object associated with the vertex.

    """

    def __init__(self, data = None):
        super().__init__()
        # by default, a new vertex belongs to its own component
        # but when the vertex is added to a graph, component points to the
        # connected component where it belongs.
        self.component = None
        self.data = data
        self.__index = None
        self.view = Rectangle()

    @property
    def index(self):
        from .graphBase import GraphBase
        if self.__index:
            return self.__index
        elif isinstance(self.component, GraphBase):
            self.__index = self.component.verticesPoset.index(self)
            return self.__index
        else:
            return None

    def __lt__(self, v):
        return 0

    def __gt__(self, v):
        return 0

    def __le__(self, v):
        return 0

    def __ge__(self, v):
        return 0

    def __getstate__(self):
        return (self.index, self.data)

    def __setstate__(self, state):
        self.__index, self.data = state
        self.component = None
        self.e = []

    def __str__(self):
        return self.data
