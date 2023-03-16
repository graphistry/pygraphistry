# DEPRECRATED: Non-vector operators over non-vectorized data

class VertexBase(object):
    """
        Base class for vertices.

        **Attributes**
            e (list[Edge]): list of edges associated with this vertex.

    """

    def __init__(self):
        # will hold list of edges for this vertex (adjacency list)
        self.e = []

    def degree(self):
        """
        degree() : degree of the vertex (number of edges).
        """
        return len(self.e)

    def e_in(self):
        """
        e_in() : list of edges directed toward this vertex.
        """
        return list(filter((lambda e: e.v[1] == self), self.e))

    def e_out(self):
        """
        e_out(): list of edges directed outward this vertex.
        """
        return list(filter((lambda e: e.v[0] == self), self.e))

    def e_dir(self, dir):
        """
        either e_in, e_out or all edges depending on provided direction parameter (>0 means outward).
        """
        if dir > 0:
            return self.e_out()
        if dir < 0:
            return self.e_in()
        return self.e

    def neighbors(self, direction = 0):
        """
            Returns the neighbors of this vertex. List of neighbor vertices in all directions (default) or in filtered f_io direction (>0 means outward).

            :param direction:
                - 0: parent and children
                - -1: parents
                - +1: children
            :return: list of vertices
        """
        arr = []
        if direction <= 0:
            arr += [e.v[0] for e in self.e_in()]
        if direction >= 0:
            arr += [e.v[1] for e in self.e_out()]
        return arr

    def e_to(self, y):
        """
        returns the Edge from this vertex directed toward vertex v.
        """
        for e in self.e_out():
            if e.v[1] == y:
                return e
        return None

    def e_from(self, x):
        """
         returns the Edge from vertex v directed toward this vertex.
        """
        for e in self.e_in():
            if e.v[0] == x:
                return e
        return None

    def e_with(self, v):
        """
        return the Edge with both this vertex and vertex v
        """
        for e in self.e:
            if v in e.v:
                return e
        return None

    def detach(self):
        """
        removes this vertex from all its edges and returns this list of edges.
        """
        E = self.e[:]
        for e in E:
            e.detach()
        assert self.degree() == 0
        return E
