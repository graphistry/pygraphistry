# DEPRECRATED: Non-vector operators over non-vectorized data
from typing import Optional

class LayoutVertex(object):
    """
    The Sugiyama layout adds new attributes to vertices.
    These attributes are stored in an internal _sugimyama_vertex_attr object.

    Attributes:
        layer (int): layer number
        dummy (0/1): whether the vertex is a dummy
        pos (int): the index of the vertex within the layer
        x (list(float)): the list of computed horizontal coordinates of the vertex
        bar (float): the current barycenter of the vertex
    """

    def __init__(self, layer: Optional[int] = None, is_dummy = 0):
        self.layer = layer  # layer number
        self.dummy = is_dummy
        self.root = None
        self.align = None
        self.sink = None
        self.shift = None
        self.X = None
        self.pos = None
        self.x = 0
        self.bar = None
        self.nvs = None

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.layer, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s
