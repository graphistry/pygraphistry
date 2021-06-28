from .PlotterBase import PlotterBase
from .gremlin import GremlinMixin, CosmosMixin

class Plotter(CosmosMixin, GremlinMixin, PlotterBase, object):

    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
