# DEPRECRATED: Non-vector operators over non-vectorized data

class EdgeBase(object):
    """

    Base class for edges.

    **Attributes**
          - degree (int): degree of the edge (number of unique vertices).
          - v (list[Vertex]): list of vertices associated with this edge.
    """
    degree: int
    """Is 0 if a loop, otherwise 1."""

    def __init__(self, x, y):
        # a bit odd, I know
        self.degree = 0 if x == y else 1
        self.v = (x, y)
