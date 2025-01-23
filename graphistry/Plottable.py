from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union
from typing_extensions import Literal
import pandas as pd

from graphistry.models.ModelDict import ModelDict
from graphistry.models.compute.chain_remote import FormatType, OutputTypeAll, OutputTypeDf, OutputTypeGraph
from graphistry.models.compute.dbscan import DBSCANEngine
from graphistry.models.compute.umap import UMAPEngineConcrete
from graphistry.plugins_types.cugraph_types import CuGraphKind
from graphistry.Engine import Engine, EngineAbstract
from graphistry.utils.json import JSONVal


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


RenderModesConcrete = Literal["g", "url", "ipython", "databricks", "browser"]
RENDER_MODE_CONCRETE_VALUES: Set[RenderModesConcrete] = set(["g", "url", "ipython", "databricks", "browser"])
RenderModes = Union[Literal["auto"], RenderModesConcrete]
RENDER_MODE_VALUES: Set[RenderModes] = set(["auto", "g", "url", "ipython", "databricks", "browser"])

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
    _render : RenderModesConcrete
    _url_params : dict
    _name : Optional[str]
    _description : Optional[str]
    _style : Optional[dict]
    _complex_encodings : dict
    _bolt_driver : Any
    _tigergraph : Any
    _spannergraph: Any
    
    _dataset_id: Optional[str]
    _url: Optional[str]
    _nodes_file_id: Optional[str]
    _edges_file_id: Optional[str]

    _node_embedding : Optional[pd.DataFrame]
    _node_encoder : Optional[Any]
    _node_features : Optional[pd.DataFrame]
    _node_features_raw: Optional[pd.DataFrame]
    _node_target : Optional[pd.DataFrame]

    _edge_embedding : Optional[pd.DataFrame]
    _edge_encoder : Optional[Any]
    _edge_features : Optional[pd.DataFrame]
    _edge_features_raw: Optional[pd.DataFrame]
    _edge_target : Optional[pd.DataFrame]

    _weighted_adjacency: Optional[Any]
    _weighted_adjacency_nodes : Optional[Any]
    _weighted_adjacency_edges : Optional[Any]
    _weighted_edges_df : Optional[pd.DataFrame]
    _weighted_edges_df_from_nodes : Optional[pd.DataFrame]
    _weighted_edges_df_from_edges : Optional[pd.DataFrame]
    _xy: Optional[pd.DataFrame]

    _umap : Optional[UMAP]
    _umap_engine: Optional[UMAPEngineConcrete]
    _umap_params: Optional[Union[ModelDict, Dict[str, Any]]]
    _umap_fit_kwargs: Optional[Dict[str, Any]]
    _umap_transform_kwargs: Optional[Dict[str, Any]]

    # extra umap
    _n_components: int
    _metric: str
    _n_neighbors: int
    _min_dist: float
    _spread: float
    _local_connectivity: int
    _repulsion_strength: float
    _negative_sample_rate: float
    _suffix: str

    _dbscan_engine: Optional[DBSCANEngine]
    _dbscan_params: Optional[ModelDict]
    _dbscan_nodes: Optional[Any]  # fit model
    _dbscan_edges: Optional[Any]  # fit model

    _adjacency : Optional[Any]
    _entity_to_index : Optional[dict]
    _index_to_entity : Optional[dict]

    DGL_graph: Optional[Any]
    
    # embed utils
    _relation : Optional[str]
    _use_feat: bool
    _triplets: Optional[List]  # actually torch.Tensor too
    _kg_embed_dim: int

    # layout
    _partition_offsets: Optional[Dict[str, Dict[int, float]]]  # from gib


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
    def chain(self, ops: Union[Any, List[Any]]) -> 'Plottable':
        """
        ops is Union[List[ASTObject], Chain]
        """
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def chain_remote(
        self: 'Plottable',
        chain: Union[Any, Dict[str, JSONVal]],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        output_type: OutputTypeGraph = "all",
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: Optional[Literal["pandas", "cudf"]] = None
    ) -> 'Plottable':
        """
        chain is Union[List[ASTObject], Chain]
        """
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def chain_remote_shape(
        self: 'Plottable',
        chain: Union[Any, Dict[str, JSONVal]],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: Optional[Literal["pandas", "cudf"]] = None
    ) -> pd.DataFrame:
        """
        chain is Union[List[ASTObject], Chain]
        """
        if 1 + 1:
            raise RuntimeError('should not happen')
        return pd.DataFrame({})

    def python_remote_g(
        self: 'Plottable',
        code: str,
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = 'parquet',
        output_type: Optional[OutputTypeAll] = 'all',
        engine: Literal["pandas", "cudf"] = "cudf",
        run_label: Optional[str] = None,
        validate: bool = True
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def python_remote_table(
        self: 'Plottable',
        code: str,
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = 'parquet',
        output_type: Optional[OutputTypeDf] = 'table',
        engine: Literal["pandas", "cudf"] = "cudf",
        run_label: Optional[str] = None,
        validate: bool = True
    ) -> pd.DataFrame:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return pd.DataFrame({})

    def python_remote_json(
        self: 'Plottable',
        code: str,
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        engine: Literal["pandas", "cudf"] = "cudf",
        run_label: Optional[str] = None,
        validate: bool = True
    ) -> Any:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return {}

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
    
    def fa2_layout(
        self,
        fa2_params: Optional[Dict[str, Any]] = None,
        circle_layout_params: Optional[Dict[str, Any]] = None,
        singleton_layout: Optional[Callable[['Plottable', Union[Tuple[float, float, float, float], Any]], 'Plottable']] = None,
        partition_key: Optional[str] = None,
        engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

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

    def scene_settings(
        self,
        menu: Optional[bool] = None,
        info: Optional[bool] = None,
        show_arrows: Optional[bool] = None,
        point_size: Optional[float] = None,
        edge_curvature: Optional[float] = None,
        edge_opacity: Optional[float] = None,
        point_opacity: Optional[float] = None,
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def encode_axis(self, rows=[]) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def settings(self,
        height: Optional[float] = None,
        url_params: Dict[str, Any] = {},
        render: Optional[Union[bool, RenderModes]] = None
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def to_cudf(self) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
    
    def to_pandas(self) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def protocol(self, v: Optional[str] = None) -> str:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return ''
    
    def server(self, v: Optional[str] = None) -> str:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return ''
    
    def client_protocol_hostname(self, v: Optional[str] = None) -> str:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return ''
    
    def base_url_server(self, v: Optional[str] = None) -> str:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return ''
    
    def base_url_client(self, v: Optional[str] = None) -> str:
        if 1 + 1:
            raise RuntimeError('should not happen')
        return ''

    def upload(
        self,
        memoize: bool = True,
        validate: bool = True
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def plot(
        self,
        graph=None,
        nodes=None,
        name=None,
        description=None,
        render: Optional[Union[bool, RenderModes]] = "auto",
        skip_upload=False, as_files=False, memoize=True,
        extra_html="", override_html_style=None, validate: bool = True
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self
