from .PlotterBase import PlotterBase
from .compute.ComputeMixin import ComputeMixin 
from .gremlin import CosmosMixin, NeptuneMixin
from .layouts import LayoutsMixin
from .feature_utils import FeatureMixin
from .dgl_utils import DGLGraphMixin
from .umap_utils import UMAPMixin
from .embed_utils import HeterographEmbedModuleMixin
from .text_utils import SearchToGraphMixin
from .compute.conditional import ConditionalMixin
from .compute.cluster import ClusterMixin


# NOTE: Cooperative mixins must call:
#       super().__init__(*a, **kw) in their __init__ method
#       to pass along args/kwargs to the next mixin in the chain
class Plotter(
    CosmosMixin, NeptuneMixin,
    HeterographEmbedModuleMixin,
    SearchToGraphMixin,
    DGLGraphMixin, ClusterMixin,
    UMAPMixin,
    FeatureMixin, ConditionalMixin,
    LayoutsMixin,
    ComputeMixin, PlotterBase
):
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
