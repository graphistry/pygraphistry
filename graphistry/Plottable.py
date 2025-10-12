from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple, Union, Protocol, overload
from typing_extensions import Literal, runtime_checkable
import pandas as pd

from graphistry.models.ModelDict import ModelDict
from graphistry.models.compute.chain_remote import FormatType, OutputTypeAll, OutputTypeDf, OutputTypeGraph
from graphistry.models.compute.dbscan import DBSCANEngine
from graphistry.models.compute.umap import UMAPEngineConcrete
from graphistry.models.compute.features import GraphEntityKind
from graphistry.plugins_types.cugraph_types import CuGraphKind
from graphistry.plugins_types.embed_types import ProtoSymbolic, XSymbolic, YSymbolic
from graphistry.plugins_types.graphviz_types import EdgeAttr, Format, GraphAttr, NodeAttr, Prog
from graphistry.plugins_types.hypergraph import HypergraphResult
from graphistry.plugins_types.umap_types import UMAPEngine
from graphistry.privacy import Mode as PrivacyMode, Privacy, ModeAction
from graphistry.Engine import EngineAbstract
from graphistry.utils.json import JSONVal
from graphistry.client_session import ClientSession, AuthManagerProtocol

if TYPE_CHECKING:
    try:
        from umap import UMAP
    except:
        UMAP = Any
    try:
        from sklearn.pipeline import Pipeline
    except:
        Pipeline = Any
    try:
        from cugraph import Graph
        from cugraph import MultiGraph
        from cugraph import BiPartiteGraph
    except:
        Graph = Any
        MultiGraph = Any
        BiPartiteGraph = Any
else:
    UMAP = Any
    Pipeline = Any
    Graph = Any
    MultiGraph = Any
    BiPartiteGraph = Any


RenderModesConcrete = Literal["g", "url", "ipython", "databricks", "browser"]
RENDER_MODE_CONCRETE_VALUES: Set[RenderModesConcrete] = set(["g", "url", "ipython", "databricks", "browser"])
RenderModes = Union[Literal["auto"], RenderModesConcrete]
RENDER_MODE_VALUES: Set[RenderModes] = set(["auto", "g", "url", "ipython", "databricks", "browser"])

@runtime_checkable
class Plottable(Protocol):
    session: ClientSession
    _pygraphistry: AuthManagerProtocol

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
    _privacy : Optional[Privacy]
    _name : Optional[str]
    _description : Optional[str]
    _style : Optional[dict]
    _complex_encodings : dict
    _bolt_driver : Any
    _tigergraph : Any

    _dataset_id: Optional[str]
    _url: Optional[str]
    _nodes_file_id: Optional[str]
    _edges_file_id: Optional[str]

    _node_embedding : Optional[pd.DataFrame]
    _node_encoder : Optional[Any]
    _node_features : Optional[pd.DataFrame]
    _node_features_raw: Optional[pd.DataFrame]
    _node_target : Optional[pd.DataFrame]
    _node_target_raw : Optional[pd.DataFrame]

    _edge_embedding : Optional[pd.DataFrame]
    _edge_encoder : Optional[Any]
    _edge_features : Optional[pd.DataFrame]
    _edge_features_raw: Optional[pd.DataFrame]
    _edge_target : Optional[pd.DataFrame]
    _edge_target_raw : Optional[pd.DataFrame]

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

    # Collapse operation internal column names (generated dynamically)
    _collapse_node_col: Optional[str]
    _collapse_src_col: Optional[str]
    _collapse_dst_col: Optional[str]

    _adjacency : Optional[Any]
    _entity_to_index : Optional[Dict]
    _index_to_entity : Optional[Dict]

    # DGL
    DGL_graph: Optional[Any]
    _dgl_graph: Optional[Any]
    
    # KG embeddings
    _relation : Optional[str]
    _use_feat: bool
    _triplets: Optional[List]  # actually torch.Tensor too
    _kg_embed_dim: int

    # layout
    _partition_offsets: Optional[Dict[str, Dict[int, float]]]  # from gib


    def reset_caches(self) -> None:
        ...

    def addStyle(
        self,
        fg: Optional[Dict[str, Any]] = None,
        bg: Optional[Dict[str, Any]] = None,
        page: Optional[Dict[str, Any]] = None,
        logo: Optional[Dict[str, Any]] = None,
    ) -> 'Plottable':
        ...

    def style(
        self,
        fg: Optional[Dict[str, Any]] = None,
        bg: Optional[Dict[str, Any]] = None,
        page: Optional[Dict[str, Any]] = None,
        logo: Optional[Dict[str, Any]] = None,
    ) -> 'Plottable':
        ...
    
    def encode_point_color(
        self,
        column: str,
        palette: Optional[List[str]] = ...,
        as_categorical: Optional[bool] = ...,
        as_continuous: Optional[bool] = ...,
        categorical_mapping: Optional[Dict[Any, Any]] = ...,
        default_mapping: Optional[str] = ...,
        for_default: bool = True,
        for_current: bool = False,
    ) -> "Plottable":
        ...
    
    def encode_edge_color(
        self,
        column: str,
        palette: Optional[List[str]] = ...,
        as_categorical: Optional[bool] = ...,
        as_continuous: Optional[bool] = ...,
        categorical_mapping: Optional[Dict[Any, Any]] = ...,
        default_mapping: Optional[str] = ...,
        for_default: bool = True,
        for_current: bool = False,
    ) -> "Plottable":
        ...

    def encode_point_size(
        self,
        column: str,
        categorical_mapping: Optional[Dict[Any, Union[int, float]]] = ...,
        default_mapping: Optional[Union[int, float]] = ...,
        for_default: bool = True,
        for_current: bool = False,
    ) -> "Plottable":
        ...


    def encode_point_icon(
        self,
        column: str,
        categorical_mapping: Optional[Dict[Any, str]] = ...,
        continuous_binning: Optional[List[Any]] = ...,
        default_mapping: Optional[str] = ...,
        comparator: Optional[Callable[[Any, Any], int]] = ...,
        for_default: bool = True,
        for_current: bool = False,
        as_text: bool = False,
        blend_mode: Optional[str] = ...,
        style: Optional[Dict[str, Any]] = ...,
        border: Optional[Dict[str, Any]] = ...,
        shape: Optional[str] = ...,
    ) -> "Plottable":
        ...


    def encode_edge_icon(
        self,
        column: str,
        categorical_mapping: Optional[Dict[Any, str]] = ...,
        continuous_binning: Optional[List[Any]] = ...,
        default_mapping: Optional[str] = ...,
        comparator: Optional[Callable[[Any, Any], int]] = ...,
        for_default: bool = True,
        for_current: bool = False,
        as_text: bool = False,
        blend_mode: Optional[str] = ...,
        style: Optional[Dict[str, Any]] = ...,
        border: Optional[Dict[str, Any]] = ...,
        shape: Optional[str] = ...,
    ) -> "Plottable":
        ...


    def encode_point_badge(
        self,
        column: str,
        position: str = "TopRight",
        categorical_mapping: Optional[Dict[Any, Any]] = ...,
        continuous_binning: Optional[List[Any]] = ...,
        default_mapping: Optional[Any] = ...,
        comparator: Optional[Callable[[Any, Any], int]] = ...,
        color: Optional[str] = ...,
        bg: Optional[str] = ...,
        fg: Optional[str] = ...,
        for_current: bool = False,
        for_default: bool = True,
        as_text: Optional[bool] = ...,
        blend_mode: Optional[str] = ...,
        style: Optional[Dict[str, Any]] = ...,
        border: Optional[Dict[str, Any]] = ...,
        shape: Optional[str] = ...,
    ) -> "Plottable":
        ...

    def encode_edge_badge(
        self,
        column: str,
        position: str = "TopRight",
        categorical_mapping: Optional[Dict[Any, Any]] = ...,
        continuous_binning: Optional[List[Any]] = ...,
        default_mapping: Optional[Any] = ...,
        comparator: Optional[Callable[[Any, Any], int]] = ...,
        color: Optional[str] = ...,
        bg: Optional[str] = ...,
        fg: Optional[str] = ...,
        for_current: bool = False,
        for_default: bool = True,
        as_text: Optional[bool] = ...,
        blend_mode: Optional[str] = ...,
        style: Optional[Dict[str, Any]] = ...,
        border: Optional[Dict[str, Any]] = ...,
        shape: Optional[str] = ...,
    ) -> "Plottable":
        ...

    def name(self, name: str) -> 'Plottable':
        ...
    
    def description(self, description: str) -> 'Plottable':
        ...

    def nodes(
        self, nodes: Union[Callable, Any], node: Optional[str] = None,
        *args: Any, **kwargs: Any
    ) -> 'Plottable':
        ...

    def edges(
        self, edges: Union[Callable, Any], source: Optional[str] = None, destination: Optional[str] = None, edge: Optional[str] = None,
        *args: Any, **kwargs: Any
    ) -> 'Plottable':
        ...

    def pipe(self, graph_transform: Callable, *args: Any, **kwargs: Any) -> 'Plottable':
        ...

    def graph(self, ig: Any) -> 'Plottable':
        ...

    def bind(
        self,
        source: Optional[str] = None,
        destination: Optional[str] = None,
        node: Optional[str] = None,
        edge: Optional[str] = None,
        edge_title: Optional[str] = None,
        edge_label: Optional[str] = None,
        edge_color: Optional[str] = None,
        edge_weight: Optional[str] = None,
        edge_size: Optional[str] = None,
        edge_opacity: Optional[str] = None,
        edge_icon: Optional[str] = None,
        edge_source_color: Optional[str] = None,
        edge_destination_color: Optional[str] = None,
        point_title: Optional[str] = None,
        point_label: Optional[str] = None,
        point_color: Optional[str] = None,
        point_weight: Optional[str] = None,
        point_size: Optional[str] = None,
        point_opacity: Optional[str] = None,
        point_icon: Optional[str] = None,
        point_x: Optional[str] = None,
        point_y: Optional[str] = None,
        dataset_id: Optional[str] = None,
        url: Optional[str] = None,
        nodes_file_id: Optional[str] = None,
        edges_file_id: Optional[str] = None,
    ) -> 'Plottable':
        ...

    def copy(self) -> 'Plottable':
        ...

    # ### ComputeMixin

    def get_indegrees(self, col: str = 'degree_in') -> 'Plottable':
        ...

    def get_outdegrees(self, col: str = 'degree_out') -> 'Plottable':
        ...

    def get_degrees(
        self,
        col: str = "degree",
        degree_in: str = "degree_in",
        degree_out: str = "degree_out",
    ) -> 'Plottable':
        ...

    def materialize_nodes(self, reuse: bool = True, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> 'Plottable':
        ...

    def get_topological_levels(
        self,
        level_col: str = 'level',
        allow_cycles: bool = True,
        warn_cycles: bool = True,
        remove_self_loops: bool = True
    ) -> 'Plottable':
        ...
    
    def drop_nodes(self, nodes: Any) -> 'Plottable':
        ...

    def keep_nodes(self, nodes: Union[List, Any]) -> 'Plottable':
        ...

    def prune_self_edges(self) -> 'Plottable':
        ...

    def collapse(
        self,
        node: Union[str, int],
        attribute: Union[str, int],
        column: Union[str, int],
        self_edges: bool = False,
        unwrap: bool = False,
        verbose: bool = False
    ) -> 'Plottable':
        ...

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
        ...

    def filter_nodes_by_dict(self, filter_dict: Optional[dict] = None) -> 'Plottable':
        ...

    def filter_edges_by_dict(self, filter_dict: Optional[dict] = None) -> 'Plottable':
        ...

    # FIXME python recursive typing issues
    def chain(self, ops: Union[Any, List[Any]]) -> 'Plottable':
        """
        ops is Union[List[ASTObject], Chain]
        """
        ...

    @overload
    def hypergraph(
        self,
        raw_events: Optional[Any] = None,
        *,
        entity_types: Optional[List[str]] = None,
        opts: dict = {},
        drop_na: bool = True,
        drop_edge_attrs: bool = False,
        verbose: bool = True,
        direct: bool = False,
        engine: str = 'pandas',
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        from_edges: bool = False,
        return_as: Literal['graph'] = 'graph'
    ) -> 'Plottable':
        ...

    @overload
    def hypergraph(
        self,
        raw_events: Optional[Any] = None,
        *,
        entity_types: Optional[List[str]] = None,
        opts: dict = {},
        drop_na: bool = True,
        drop_edge_attrs: bool = False,
        verbose: bool = True,
        direct: bool = False,
        engine: str = 'pandas',
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        from_edges: bool = False,
        return_as: Literal['all']
    ) -> HypergraphResult:
        ...

    @overload
    def hypergraph(
        self,
        raw_events: Optional[Any] = None,
        *,
        entity_types: Optional[List[str]] = None,
        opts: dict = {},
        drop_na: bool = True,
        drop_edge_attrs: bool = False,
        verbose: bool = True,
        direct: bool = False,
        engine: str = 'pandas',
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        from_edges: bool = False,
        return_as: Literal['entities', 'events', 'edges', 'nodes']
    ) -> Any:
        ...

    def hypergraph(
        self,
        raw_events: Optional[Any] = None,
        *,
        entity_types: Optional[List[str]] = None,
        opts: dict = {},
        drop_na: bool = True,
        drop_edge_attrs: bool = False,
        verbose: bool = True,
        direct: bool = False,
        engine: str = 'pandas',
        npartitions: Optional[int] = None,
        chunksize: Optional[int] = None,
        from_edges: bool = False,
        return_as: Literal['graph', 'all', 'entities', 'events', 'edges', 'nodes'] = 'graph'
    ) -> Union['Plottable', HypergraphResult, Any]:
        ...

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
        engine: Optional[Literal["pandas", "cudf"]] = None,
        validate: bool = True,
        persist: bool = False
    ) -> 'Plottable':
        """
        chain is Union[List[ASTObject], Chain]
        """
        ...

    def chain_remote_shape(
        self: 'Plottable',
        chain: Union[Any, Dict[str, JSONVal]],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: Optional[Literal["pandas", "cudf"]] = None,
        validate: bool = True,
        persist: bool = False
    ) -> pd.DataFrame:
        """
        chain is Union[List[ASTObject], Chain]
        """
        ...

    def gfql_remote(
        self: 'Plottable',
        chain: Union[Any, Dict[str, JSONVal]],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        output_type: OutputTypeGraph = "all",
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: Optional[Literal["pandas", "cudf"]] = None,
        validate: bool = True,
        persist: bool = False
    ) -> 'Plottable':
        """
        chain is Union[List[ASTObject], Chain]
        """
        ...

    def gfql_remote_shape(
        self: 'Plottable',
        chain: Union[Any, Dict[str, JSONVal]],
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        format: Optional[FormatType] = None,
        df_export_args: Optional[Dict[str, Any]] = None,
        node_col_subset: Optional[List[str]] = None,
        edge_col_subset: Optional[List[str]] = None,
        engine: Optional[Literal["pandas", "cudf"]] = None,
        validate: bool = True,
        persist: bool = False
    ) -> pd.DataFrame:
        """
        chain is Union[List[ASTObject], Chain]
        """
        ...

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
        ...

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
        ...

    def python_remote_json(
        self: 'Plottable',
        code: str,
        api_token: Optional[str] = None,
        dataset_id: Optional[str] = None,
        engine: Literal["pandas", "cudf"] = "cudf",
        run_label: Optional[str] = None,
        validate: bool = True
    ) -> Any:
        ...

    def to_igraph(self, 
        directed: bool = True,
        include_nodes: bool = True,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        use_vids: bool = False
    ) -> Any:
        ...

    def from_igraph(self,
        ig: Any,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes: bool = True, load_edges: bool = True,
        merge_if_existing: bool = True
    ) -> 'Plottable':
        ...
    
    def compute_igraph(self,
        alg: str, out_col: Optional[str] = None, directed: Optional[bool] = None, use_vids: bool = False, params: dict = {}, stringify_rich_types: bool = True
    ) -> 'Plottable':
        ...
    
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
        ...

    def from_cugraph(self,
        G,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        load_nodes: bool = True, load_edges: bool = True,
        merge_if_existing: bool = True
    ) -> 'Plottable':
        ...

    def to_cugraph(self, 
        directed: bool = True,
        include_nodes: bool = True,
        node_attributes: Optional[List[str]] = None,
        edge_attributes: Optional[List[str]] = None,
        kind : CuGraphKind = 'Graph'
    ) -> Union[Graph, MultiGraph, BiPartiteGraph]:
        ...

    def compute_cugraph(self,
        alg: str, out_col: Optional[str] = None, params: dict = {},
        kind : CuGraphKind = 'Graph', directed = True,
        G: Optional[Any] = None
    ) -> 'Plottable':
        ...

    def layout_cugraph(self,
        layout: str = 'force_atlas2', params: dict = {},
        kind : CuGraphKind = 'Graph', directed = True,
        G: Optional[Any] = None,
        bind_position: bool = True,
        x_out_col: str = 'x',
        y_out_col: str = 'y',
        play: Optional[int] = 0
    ) -> 'Plottable':
        ...

    def layout_graphviz(self,
        prog: Prog = 'dot',
        args: Optional[str] = None,
        directed: bool = True,
        strict: bool = False,
        graph_attr: Optional[Dict[GraphAttr, Any]] = None,
        node_attr: Optional[Dict[NodeAttr, Any]] = None,
        edge_attr: Optional[Dict[EdgeAttr, Any]] = None,
        skip_styling: bool = False,
        render_to_disk: bool = False,  # unsafe in server settings
        path: Optional[str] = None,
        format: Optional[Format] = None
    ) -> 'Plottable':
        ...

    def from_networkx(self, G: Any) -> 'Plottable':
        ...

    def networkx_checkoverlap(self, g: Any) -> None:
        ...
    
    def networkx2pandas(self, G: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...
    
    def fa2_layout(
        self,
        fa2_params: Optional[Dict[str, Any]] = None,
        circle_layout_params: Optional[Dict[str, Any]] = None,
        singleton_layout: Optional[Callable[['Plottable', Union[Tuple[float, float, float, float], Any]], 'Plottable']] = None,
        partition_key: Optional[str] = None,
        engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
    ) -> 'Plottable':
        ...

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
    ) -> 'Plottable':
        ...

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
        ...

    def encode_axis(self, rows: List[Dict] = []) -> 'Plottable':
        ...

    def settings(self,
        height: Optional[int] = None,
        url_params: Dict[str, Any] = {},
        render: Optional[Union[bool, RenderModes]] = None
    ) -> 'Plottable':
        ...

    def privacy(self, mode: Optional[PrivacyMode] = None, notify: Optional[bool] = None, invited_users: Optional[List[str]] = None, message: Optional[str] = None, mode_action: Optional[ModeAction] = None) -> 'Plottable':
        ...

    def to_cudf(self) -> 'Plottable':
        ...
    
    def to_pandas(self) -> 'Plottable':
        ...

    def protocol(self, v: Optional[str] = None) -> str:
        ...
    
    def server(self, v: Optional[str] = None) -> str:
        ...
    
    def client_protocol_hostname(self, v: Optional[str] = None) -> str:
        ...
    
    def base_url_server(self, v: Optional[str] = None) -> str:
        ...
    
    def base_url_client(self, v: Optional[str] = None) -> str:
        ...

    @property
    def url(self) -> Optional[str]:
        ...

    def upload(
        self,
        memoize: bool = True,
        erase_files_on_fail=True,
        validate: bool = True
    ) -> 'Plottable':
        ...

    def plot(
        self,
        graph: Optional[Any] = None,
        nodes: Optional[Any] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
        render: Optional[Union[bool, RenderModes]] = "auto",
        skip_upload: bool = False,
        as_files: bool = False,
        memoize: bool = True,
        erase_files_on_fail: bool = True,
        extra_html: str = "",
        override_html_style: Optional[str] = None,
        validate: bool = True
    ) -> Any:
        ...

    def pandas2igraph(self, edges: pd.DataFrame, directed: bool = True) -> Any:
        ...

    def igraph2pandas(self, ig: Any) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def infer_labels(self) -> 'Plottable':
        ...

    
    @overload
    def transform(self, df: pd.DataFrame, 
                  y: Optional[pd.DataFrame] = None, 
                  kind: str = 'nodes', 
                  min_dist: Union[str, float, int] = 'auto', 
                  n_neighbors: int = 7,
                  merge_policy: bool = False,
                  sample: Optional[int] = None, 
                  *,
                  return_graph: Literal[True] = True,
                  scaled: bool = True,
                  verbose: bool = False) -> 'Plottable':
        ...

    @overload
    def transform(self, df: pd.DataFrame, 
                  y: Optional[pd.DataFrame] = None, 
                  kind: str = 'nodes', 
                  min_dist: Union[str, float, int] = 'auto', 
                  n_neighbors: int = 7,
                  merge_policy: bool = False,
                  sample: Optional[int] = None, 
                  *,
                  return_graph: Literal[False],
                  scaled: bool = True,
                  verbose: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def transform(self, df: pd.DataFrame, 
                  y: Optional[pd.DataFrame] = None, 
                  kind: str = 'nodes', 
                  min_dist: Union[str, float, int] = 'auto', 
                  n_neighbors: int = 7,
                  merge_policy: bool = False,
                  sample: Optional[int] = None, 
                  *,
                  return_graph: bool = True,
                  scaled: bool = True,
                  verbose: bool = False) -> Union[Tuple[pd.DataFrame, pd.DataFrame], 'Plottable']:
        ...


    @overload
    def transform_umap(self, df: pd.DataFrame,
                    y: Optional[pd.DataFrame] = None,
                    kind: GraphEntityKind = 'nodes',
                    min_dist: Union[str, float, int] = 'auto',
                    n_neighbors: int = 7,
                    merge_policy: bool = False,
                    sample: Optional[int] = None,
                    *,
                    return_graph: Literal[True] = True,
                    fit_umap_embedding: bool = True,
                    umap_transform_kwargs: Dict[str, Any] = {}
    ) -> 'Plottable':
        ...

    @overload
    def transform_umap(self, df: pd.DataFrame,
                    y: Optional[pd.DataFrame] = None,
                    kind: GraphEntityKind = 'nodes',
                    min_dist: Union[str, float, int] = 'auto',
                    n_neighbors: int = 7,
                    merge_policy: bool = False,
                    sample: Optional[int] = None,
                    *,
                    return_graph: Literal[False],
                    fit_umap_embedding: bool = True,
                    umap_transform_kwargs: Dict[str, Any] = {}
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ...

    def transform_umap(self, df: pd.DataFrame,
                    y: Optional[pd.DataFrame] = None,
                    kind: GraphEntityKind = 'nodes',
                    min_dist: Union[str, float, int] = 'auto',
                    n_neighbors: int = 7,
                    merge_policy: bool = False,
                    sample: Optional[int] = None,
                    *,
                    return_graph: bool = True,
                    fit_umap_embedding: bool = True,
                    umap_transform_kwargs: Dict[str, Any] = {}
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], 'Plottable']:
        ...

    def umap_lazy_init(
        self,
        res: "Plottable",
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        engine: UMAPEngine = "auto",
        suffix: str = "",
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
    ) -> "Plottable":
        ...


    def umap_fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.DataFrame] = None,
        umap_fit_kwargs: Dict[str, Any] = {},
    ) -> "Plottable":
        ...

    @overload
    def umap(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        kind: GraphEntityKind = "nodes",
        scale: float = 1.0,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        suffix: str = "",
        play: Optional[int] = 0,
        encode_position: bool = True,
        encode_weight: bool = True,
        dbscan: bool = False,
        engine: UMAPEngine = "auto",
        feature_engine: str = "auto",
        inplace: Literal[False] = False,
        memoize: bool = True,
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
        **featurize_kwargs: Any,
    ) -> "Plottable":
        ...

    @overload
    def umap(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        kind: GraphEntityKind = "nodes",
        scale: float = 1.0,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        suffix: str = "",
        play: Optional[int] = 0,
        encode_position: bool = True,
        encode_weight: bool = True,
        dbscan: bool = False,
        engine: UMAPEngine = "auto",
        feature_engine: str = "auto",
        inplace: Literal[True] = ...,
        memoize: bool = True,
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
        **featurize_kwargs: Any,
    ) -> None:
        ...

    def umap(
        self,
        X: XSymbolic = None,
        y: YSymbolic = None,
        kind: GraphEntityKind = "nodes",
        scale: float = 1.0,
        n_neighbors: int = 12,
        min_dist: float = 0.1,
        spread: float = 0.5,
        local_connectivity: int = 1,
        repulsion_strength: float = 1,
        negative_sample_rate: int = 5,
        n_components: int = 2,
        metric: str = "euclidean",
        suffix: str = "",
        play: Optional[int] = 0,
        encode_position: bool = True,
        encode_weight: bool = True,
        dbscan: bool = False,
        engine: UMAPEngine = "auto",
        feature_engine: str = "auto",
        inplace: bool = False,
        memoize: bool = True,
        umap_kwargs: Dict[str, Any] = {},
        umap_fit_kwargs: Dict[str, Any] = {},
        umap_transform_kwargs: Dict[str, Any] = {},
        **featurize_kwargs: Any,
    ) -> Optional["Plottable"]:
        ...


    def filter_weighted_edges(
        self,
        scale: float = 1.0,
        index_to_nodes_dict: Optional[Dict] = None,
        inplace: bool = False,
        kind: GraphEntityKind = "nodes",
    ) -> Optional["Plottable"]:
        ...


    def search_graph(
        self,
        query: str,
        scale: float = 0.5,
        top_n: int = 100,
        thresh: float = 5000,
        broader: bool = False,
        inplace: bool = False,
    ) -> 'Plottable':
        ...

    def search(
        self,
        query: str,
        cols=None,
        thresh: float = 5000,
        fuzzy: bool = True,
        top_n: int = 10,
    ):
        ...

    def embed(
        self,
        relation:str,
        proto: ProtoSymbolic = 'DistMult',
        embedding_dim: int = 32,
        use_feat: bool = False,
        X: XSymbolic = None,
        epochs: int = 2,
        batch_size: int = 32,
        train_split: Union[float, int] = 0.8,
        sample_size: int = 1000, 
        num_steps: int = 50,
        lr: float = 1e-2,
        inplace: Optional[bool] = False,
        device: Optional['str'] = "cpu",
        evaluate: bool = True,
        *args,
        **kwargs,
    ) -> 'Plottable':
        ...
