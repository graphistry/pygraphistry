from .PlotterBase import PlotterBase
from .compute import ComputeMixin
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin, has_dependancy as has_featurize
from .dgl_utils import DGLGraphMixin, has_dependancy as has_dgl

mixins = (
    [CosmosMixin, NeptuneMixin, GremlinMixin, LayoutsMixin]
    + [DGLGraphMixin if has_dgl and has_featurize else []]
    + [FeatureMixin if has_featurize else []] +
    [ComputeMixin, PlotterBase, object]
)

class Plotter(  # type: ignore
    *mixins
):  # type: ignore
    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        if has_dgl:
            DGLGraphMixin.__init__(self, *args, **kwargs)
        if has_featurize:
            FeatureMixin.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
