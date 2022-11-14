from .PlotterBase import PlotterBase
from .compute.ComputeMixin import ComputeMixin 
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin  # type: ignore
from .dgl_utils import DGLGraphMixin  # type: ignore
from .umap_utils import UMAPMixin  # type: ignore
from .text_utils import SearchToGraphMixin  # type: ignore
from .motif_utils import MotifMixin # type: ignore

mixins = ([
    CosmosMixin, NeptuneMixin, GremlinMixin, LayoutsMixin, 
    SearchToGraphMixin, DGLGraphMixin, UMAPMixin, FeatureMixin,
    ComputeMixin, PlotterBase, MotifMixin, object
])


class Plotter(  # type: ignore
    *mixins  # type: ignore
):  # type: ignore
    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        FeatureMixin.__init__(self, *args, **kwargs)
        DGLGraphMixin.__init__(self, *args, **kwargs)
        UMAPMixin.__init__(self, *args, **kwargs)
        SearchToGraphMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
        MotifMixin.__init__(self, *args, **kwargs)

