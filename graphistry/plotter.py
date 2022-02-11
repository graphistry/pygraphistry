from .PlotterBase import PlotterBase
from .compute import ComputeMixin
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin


class Plotter(  # type: ignore
    CosmosMixin,
    NeptuneMixin,
    GremlinMixin,
    LayoutsMixin,
    FeatureMixin,
    ComputeMixin,
    PlotterBase,
    object,
):  # type: ignore
    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        FeatureMixin.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
