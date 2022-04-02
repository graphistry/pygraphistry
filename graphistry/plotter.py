from .PlotterBase import PlotterBase
from .compute import ComputeMixin
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin, has_min_dependancy as has_featurize
#from .dgl_utils import DGLGraphMixin, has_dependancy as has_dgl
from .umap_utils import UMAPMixin, has_dependancy as has_umap

mixins = (
    [CosmosMixin, NeptuneMixin, GremlinMixin, LayoutsMixin]
    #+ ([DGLGraphMixin] if has_dgl and has_featurize else [])  # noqa: W503
    + ([UMAPMixin] if has_umap else [])  # noqa: W503
    + [FeatureMixin, ComputeMixin, PlotterBase, object]  # noqa: W503
)


class Plotter(  # type: ignore
    *mixins
):  # type: ignore4
    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        FeatureMixin.__init__(self, *args, **kwargs)
        # if has_dgl:
        #     DGLGraphMixin.__init__(self, *args, **kwargs)
        if has_umap:
            UMAPMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
