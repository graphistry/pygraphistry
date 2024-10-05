from .PlotterBase import PlotterBase
from .compute.ComputeMixin import ComputeMixin 
from .gremlin import GremlinMixin, CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin  # type: ignore
from .dgl_utils import DGLGraphMixin  # type: ignore
from .umap_utils import UMAPMixin  # type: ignore
from .embed_utils import HeterographEmbedModuleMixin  # type: ignore
from .text_utils import SearchToGraphMixin  # type: ignore
from .compute.conditional import ConditionalMixin  # type: ignore
from .compute.cluster import ClusterMixin  # type: ignore


mixins = ([
    CosmosMixin, NeptuneMixin, GremlinMixin,
    HeterographEmbedModuleMixin,
    SearchToGraphMixin,
    DGLGraphMixin, ClusterMixin,
    UMAPMixin,
    FeatureMixin, ConditionalMixin,
    LayoutsMixin,
    ComputeMixin, PlotterBase, object
])


class Plotter(  # type: ignore
    *mixins  # type: ignore
):  # type: ignore
    """
    Main Plotter class for Graphistry.

    This class represents a graph in Graphistry and serves as the primary interface for plotting and analyzing graphs.
    It inherits from multiple mixins, allowing it to extend its functionality with additional graph computation, layouts, conditional formatting, and more.

    Inherits:
        - :py:class:`graphistry.PlotterBase.PlotterBase`: Base class for plotting graphs.
        - :py:class:`graphistry.compute.ComputeMixin`: Enables computation-related functions like degree calculations.
        - :py:class:`graphistry.layouts.LayoutsMixin`: Provides methods for controlling graph layouts.
        - :py:class:`graphistry.compute.conditional.ConditionalMixin`: Adds support for conditional graph operations.
        - :py:class:`graphistry.feature_utils.FeatureMixin`: Adds feature engineering capabilities.
        - :py:class:`graphistry.umap_utils.UMAPMixin`: Integrates UMAP for dimensionality reduction.
        - :py:class:`graphistry.compute.cluster.ClusterMixin`: Enables clustering-related functionalities.
        - :py:class:`graphistry.dgl_utils.DGLGraphMixin`: Integrates deep graph learning with DGL.
        - :py:class:`graphistry.text_utils.SearchToGraphMixin`: Supports converting search results into graphs.
        - :py:class:`graphistry.embed_utils.HeterographEmbedModuleMixin`: Adds heterograph embedding capabilities.
        - :py:class:`graphistry.gremlin.GremlinMixin`: Provides Gremlin query support for graph databases.
        - :py:class:`graphistry.gremlin.CosmosMixin`: Integrates with Azure Cosmos DB.
        - :py:class:`graphistry.gremlin.NeptuneMixin`: Integrates with AWS Neptune DB.

    Attributes:
        All attributes are inherited from the mixins and base classes.

    """
    def __init__(self, *args, **kwargs):
        PlotterBase.__init__(self, *args, **kwargs)
        ComputeMixin.__init__(self, *args, **kwargs)
        LayoutsMixin.__init__(self, *args, **kwargs)
        ConditionalMixin.__init__(self, *args, **kwargs)
        FeatureMixin.__init__(self, *args, **kwargs)
        UMAPMixin.__init__(self, *args, **kwargs)
        ClusterMixin.__init__(self, *args, **kwargs)
        DGLGraphMixin.__init__(self, *args, **kwargs)
        SearchToGraphMixin.__init__(self, *args, **kwargs)
        HeterographEmbedModuleMixin.__init__(self, *args, **kwargs)
        GremlinMixin.__init__(self, *args, **kwargs)
        CosmosMixin.__init__(self, *args, **kwargs)
        NeptuneMixin.__init__(self, *args, **kwargs)
