# DEPRECRATED: Non-vector operators over non-vectorized data

from .edgeBase import EdgeBase


class Edge(EdgeBase):
    """
        A graph edge.

       **Attributes**
         - data (object): an optional payload
         - w (int): an optional weight associated with the edge (default 1) used by Dijkstra to find min-flow paths.
         - feedback (bool):  whether the Tarjan algorithm has inverted this edge to de-cycle the graph.
    """
    feedback: bool
    data: object
    w: int

    def __init__(self, x, y, w = 1, data = None, connect = False):
        """
        Creates a new edge.
        :param x: source vertex
        :param y: target vertex
        :param w: optional weight
        :param data: optional data
        :param connect: whether the edge should be added to the component.
        """
        super().__init__(x, y)
        self.w = w
        self.data = data
        self.feedback = False
        if connect and (x.component is None or y.component is None):
            c = x.component or y.component
            c.add_edge(self)

    def attach(self):
        """
            Attach this edge to the edge collections of the vertices.
        """
        if self not in self.v[0].e:
            self.v[0].e.append(self)
        if self not in self.v[1].e:
            self.v[1].e.append(self)

    def detach(self):
        """
            Removes this edge from the edge collections of the vertices.
        """
        if self.degree == 1:
            assert self in self.v[0].e
            assert self in self.v[1].e
            self.v[0].e.remove(self)
            self.v[1].e.remove(self)
        else:
            if self in self.v[0].e:
                self.v[0].e.remove(self)
            assert self not in self.v[0].e
        return [self]

    def __lt__(self, v):
        return 0

    def __gt__(self, v):
        return 0

    def __le__(self, v):
        return 0

    def __ge__(self, v):
        return 0

    def __getstate__(self):
        xi, yi = (self.v[0].index, self.v[1].index)
        return (xi, yi, self.w, self.data, self.feedback)

    def __setstate__(self, state):
        xi, yi, self.w, self.data, self.feedback = state
        self._v = [xi, yi]
        self.degree = 0 if xi == yi else 1

    def __str__(self):
        return f"""{self.v[0].data}->{self.v[1].data}"""
