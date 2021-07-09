from .PlotterBase import PlotterBase
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin

class Plotter(CosmosMixin, NeptuneMixin, GremlinMixin, PlotterBase, object):  # type: ignore

    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
