from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from typing_extensions import Literal
import pandas as pd

from graphistry.plugins_types.cugraph_types import CuGraphKind
from graphistry.Engine import Engine, EngineAbstract


if TYPE_CHECKING:
    try:
        from umap import UMAP
    except:
        UMAP = Any
    try:
        from sklearn.pipeline import Pipeline
    except:
        Pipeline = Any
else:
    UMAP = Any
    Pipeline = Any

class Plottable(object):

    _edges : Any
    _nodes : Any
    _source : Optional[str]
    _destination : Optional[str]
    _node : Optional[str]
    _edge : Optional[str]
    _edge_title : Optional[str]
    _edge_label : Optional[str]
    _edge_color : Optional[str]
    _edge_source_color : Optional[str]
    _edge_destination_color : Optional[str]
    _edge_size : Optional[str]
    _edge_weight : Optional[str]
    _edge_icon : Optional[str]
    _edge_opacity : Optional[str]
    _point_title : Optional[str]
    _point_label : Optional[str]
    _point_color : Optional[str]
    _point_size : Optional[str]
    _point_weight : Optional[str]
    _point_icon : Optional[str]
    _point_opacity : Optional[str]
    _point_x : Optional[str]
    _point_y : Optional[str]
    _height : int
    _render : bool
    _url_params : dict
    _name : Optional[str]
    _description : Optional[str]
    _style : Optional[dict]
    _complex_encodings : dict
    _bolt_driver : Any
    _tigergraph : Any

    _node_embedding : Optional[pd.DataFrame]
    _node_encoder : Optional[Any]
    _node_features : Optional[pd.DataFrame]
    _node_target : Optional[pd.DataFrame]

    _edge_embedding : Optional[pd.DataFrame]
    _edge_encoder : Optional[Any]
    _edge_features : Optional[pd.DataFrame]
    _edge_target : Optional[pd.DataFrame]

    _weighted_adjacency: Optional[Any]
    _weighted_adjacency_nodes : Optional[Any]
    _weighted_adjacency_edges : Optional[Any]
    _weighted_edges_df : Optional[pd.DataFrame]
    _weighted_edges_df_from_nodes : Optional[pd.DataFrame]
    _weighted_edges_df_from_edges : Optional[pd.DataFrame]
    _xy: Optional[pd.DataFrame]

    _umap : Optional[UMAP]
    _umap_params: Optional[Dict[str, Any]]
    _umap_fit_kwargs: Optional[Dict[str, Any]]
    _umap_transform_kwargs: Optional[Dict[str, Any]]

    _adjacency : Optional[Any]
    _entity_to_index : Optional[dict]
    _index_to_entity : Optional[dict]

    DGL_graph: Optional[Any]
    
    # embed utils
    _relation : Optional[str]
    _use_feat: bool
    _triplets: Optional[List]  # actually torch.Tensor too
    _kg_embed_dim: int
    

    def __init__(self, *args, **kwargs):
        #raise RuntimeError('should not happen')
        None

    def nodes(
        self, nodes: Union[Callable, Any], node: Optional[str] = None,
        *args, **kwargs
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def edges(
        self, edges: Union[Callable, Any], source: Optional[str] = None, destination: Optional[str] = None, edge: Optional[str] = None,
        *args, **kwargs
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def pipe(self, graph_transform: Callable, *args, **kwargs) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def bind(self, source=None, destination=None, node=None, edge=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None, edge_size=None, edge_opacity=None, edge_icon=None,
             edge_source_color=None, edge_destination_color=None,
             point_title=None, point_label=None, point_color=None, point_weight=None, point_size=None, point_opacity=None, point_icon=None,
             point_x=None, point_y=None):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def copy(self):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    # ### compute

    def get_indegrees(self, col: str = 'degree_in') -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def get_outdegrees(self, col: str = 'degree_out') -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def get_degrees(
        self,
        col: str = "degree",
        degree_in: str = "degree_in",
        degree_out: str = "degree_out",
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def materialize_nodes(self, reuse: bool = True, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def get_topological_levels(
        self,
        level_col: str = 'level',
        allow_cycles: bool = True,
        warn_cycles: bool = True,
        remove_self_loops: bool = True
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def drop_nodes(
        self,
        nodes: Any
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def keep_nodes(
        self,
        nodes: Union[List, Any]
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def prune_self_edges(
        self
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def collapse(
        self,
        node: Union[str, int],
        attribute: Union[str, int],
        column: Union[str, int],
        self_edges: bool = False,
        unwrap: bool = False,
        verbose: bool = False
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def hop(self,
        nodes: Optional[pd.DataFrame],
        hops: Optional[int] = 1,
        to_fixed_point: bool = False,
        direction: str = 'forward',
        edge_match: Optional[dict] = None,
        source_node_match: Optional[dict] = None,
        destination_node_match: Optional[dict] = None,
        source_node_query: Optional[str] = None,
        destination_node_query: Optional[str] = None,
        edge_query: Optional[str] = None,
        return_as_wave_front: bool = False,
        target_wave_front: Optional[pd.DataFrame] = None
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def filter_nodes_by_dict(self, filter_dict: Optional[dict] = None) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def filter_edges_by_dict(self, filter_dict: Optional[dict] = None) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    # FIXME python recursive typing issues
    def chain(self, ops: List[Any]) -> 'Plottable':
        """
        ops is List[ASTObject]
        """
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def to_igraph(self, 
        directed: bool = True,
        use_vids: bool = False,
        include_nodes: bool = True,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None
    ) -> Any:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def from_igraph(self,
        ig,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes: bool = True, load_edges: bool = True,
        merge_if_existing: bool = True
    ):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def compute_igraph(self,
        alg: str, out_col: Optional[str] = None, directed: Optional[bool] = None, use_vids: bool = False, params: dict = {}, stringify_rich_types: bool = True
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def layout_igraph(self,
        layout: str,
        directed: Optional[bool] = None,
        use_vids: bool = False,
        bind_position: bool = True,
        x_out_col: str = 'x',
        y_out_col: str = 'y',
        play: Optional[int] = 0,
        params: dict = {}
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def from_cugraph(self,
        G,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes: bool = True, load_edges: bool = True,
        merge_if_existing: bool = True
    ):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def to_cugraph(self, 
        directed: bool = True,
        include_nodes: bool = True,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        kind : CuGraphKind = 'Graph'
    ) -> Any:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def compute_cugraph(self,
        alg: str, out_col: Optional[str] = None, params: dict = {},
        kind : CuGraphKind = 'Graph', directed = True,
        G: Optional[Any] = None
    ):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def layout_cugraph(self,
        layout: str = 'force_atlas2', params: dict = {},
        kind : CuGraphKind = 'Graph', directed = True,
        G: Optional[Any] = None,
        bind_position: bool = True,
        x_out_col: str = 'x',
        y_out_col: str = 'y',
        play: Optional[int] = 0
    ):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def from_networkx(self, G: Any) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def networkx2pandas(self, G: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return pd.DataFrame(), pd.DataFrame()

    def layout_settings(
        self,

        play: Optional[int] = None,

        locked_x: Optional[bool] = None,
        locked_y: Optional[bool] = None,
        locked_r: Optional[bool] = None,

        left: Optional[float] = None,
        top: Optional[float] = None,
        right: Optional[float] = None,
        bottom: Optional[float] = None,

        lin_log: Optional[bool] = None,
        strong_gravity: Optional[bool] = None,
        dissuade_hubs: Optional[bool] = None,

        edge_influence: Optional[float] = None,
        precision_vs_speed: Optional[float] = None,
        gravity: Optional[float] = None,
        scaling_ratio: Optional[float] = None
    ):
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def encode_axis(self, rows=[]) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def settings(self, height: Optional[float] = None, url_params: Dict[str, Any] = {}, render: Optional[bool] = None) -> 'Plottable':
        """Specify iframe height and add URL parameter dictionary.

        The library takes care of URI component encoding for the dictionary.

        :param height: Height in pixels.
        :type height: int

        :param url_params: Dictionary of querystring parameters to append to the URL.
        :type url_params: dict

        :param render: Whether to render the visualization using the native notebook environment (default True), or return the visualization URL
        :type render: bool

        """

        res = self.copy()
        res._height = height or self._height
        res._url_params = dict(self._url_params, **url_params)
        res._render = self._render if render is None else render
        return res
