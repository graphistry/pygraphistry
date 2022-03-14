# DEPRECRATED: Non-vector operators over non-vectorized data

from .rectangle import Rectangle
from .layoutVertex import LayoutVertex


class DummyVertex(LayoutVertex):
    """
    A DummyVertex is used for edges that span over several layers, it's inserted in every inner layer.

    **Attributes**
        - view (viewclass): since a DummyVertex is acting as a Vertex, it must have a view.
        - ctrl (list[_sugiyama_attr]): the list of associated dummy vertices.
    """

    def __init__(self, r = None):
        self.view = Rectangle()
        self.control_vertices = None
        super().__init__(r, is_dummy = 1)

    def neighbors(self, direction: int):
        """
            Reflect the Vertex method and returns the list of adjacent vertices (possibly dummy) in the given direction.
            :param direction: +1 for the next layer (children) and -1 (parents) for the previous
        """
        # assert direction == +1 or direction == -1
        assert isinstance(self.layer, int)
        v = self.control_vertices.get(int(self.layer) + direction)
        return [v] if v is not None else []

    def inner(self, direction):
        """
         True if a neighbor in the given direction is *dummy*.
        """
        assert direction == +1 or direction == -1
        try:
            return any([x.dummy == 1 for x in self.neighbors(direction)])
        except KeyError:
            return False
        except AttributeError:
            return False

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.layer, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s
