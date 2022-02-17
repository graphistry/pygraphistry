from .rectangle import Rectangle
from .layoutVertex import LayoutVertex


class DummyVertex(LayoutVertex):
    """
    A DummyVertex is used for edges that span over several layers, it's inserted in every inner layer.

    **Attributes**
        - view (viewclass): since a DummyVertex is acting as a Vertex, it must have a view.
        - ctrl (list[_sugiyama_attr]): the list of associated dummy vertices.
    """

    def __init__(self, r = None ):
        self.view = Rectangle()
        self.ctrl = None
        super().__init__(r, is_dummy = 1)

    def neighbors(self, dir):
        """
            Reflect the Vertex method and returns the list of adjacent vertices (possibly dummy) in the given direction.
        """
        assert dir == +1 or dir == -1
        v = self.ctrl.get(self.layer + dir, None)
        return [v] if v is not None else []

    def inner(self, dir):
        """
         True if a neighbor in the given direction is *dummy*.
        """
        assert dir == +1 or dir == -1
        try:
            return any([x.dummy == 1 for x in self.neighbors(dir)])
        except KeyError:
            return False
        except AttributeError:
            return False

    def __str__(self):
        s = "(%3d,%3d) x=%s" % (self.layer, self.pos, str(self.x))
        if self.dummy:
            s = "[d] %s" % s
        return s
