from typing import Optional
import warnings

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
from .plugins.kusto import KustoMixin
from .plugins.spanner import SpannerMixin
from .client_session import AuthManagerProtocol
# NOTE: Cooperative mixins must call:
#       super().__init__(*a, **kw) in their __init__ method
#       to pass along args/kwargs to the next mixin in the chain
class Plotter(
    KustoMixin, SpannerMixin,
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

    Implements the :py:class:`graphistry.Plottable.Plottable` interface.

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
        - :py:class:`graphistry.plugins.kusto.KustoMixin`: Integrates with Azure Kusto DB.
        - :py:class:`graphistry.plugins.spanner.SpannerMixin`: Integrates with Google Spanner DB.

    Attributes:
        All attributes are inherited from the mixins and base classes.


    Session Binding:
        A Plottable's state is tied to the client used to create it through two attributes:
          - _pygraphistry: Reference to the `GraphistryClient` that created this plottable
          - session: The `ClientSession` (self._pygraphistry.session)

        See: :py:class:`graphistry.pygraphistry.GraphistryClient` for more details.
        
        This binding ensures that authentication state, server configuration, and other
        session-specific settings are preserved when plotting. The session reference is
        particularly important during plot() operations where token refresh may occur.
    
    Concurrency:
        Each plottable inherits the concurrency constraints of its parent client. A plottable
        should only be used within the same concurrency context as the client that created it.
        
        To transfer a plottable between clients, use client.set_client_for(plottable).
    """

    def __init__(self, *args, pygraphistry: Optional[AuthManagerProtocol] = None, **kwargs) -> None:
        from .pygraphistry import PyGraphistry
        if pygraphistry is None:
            warnings.warn(
                "Initializing Plotter without PyGraphistryClient,"
                "defaulting to global PyGraphistry instance.",
                UserWarning
            )
            self._pygraphistry = PyGraphistry
        else:
            self._pygraphistry = pygraphistry
        self.session = self._pygraphistry.session

        super().__init__(*args, **kwargs)
