"""Type definitions for GFQL call() operations.

Provides TypedDict classes and Literal types for type-safe call() usage.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict, Union


# Literal type of all valid call method names
CallMethodName = Literal[
    'collapse',
    'compute_cugraph',
    'compute_igraph',
    'description',
    'drop_nodes',
    'encode_edge_color',
    'encode_point_color',
    'encode_point_icon',
    'encode_point_size',
    'fa2_layout',
    'filter_edges_by_dict',
    'filter_nodes_by_dict',
    'get_degrees',
    'get_indegrees',
    'get_outdegrees',
    'get_topological_levels',
    'group_in_a_box_layout',
    'hop',
    'hypergraph',
    'keep_nodes',
    'layout_cugraph',
    'layout_graphviz',
    'layout_igraph',
    'materialize_nodes',
    'name',
    'prune_self_edges',
    'umap'
]


# TypedDict parameter classes for each method
# Using total=False to make all fields optional (matching safelist behavior)

class HopParams(TypedDict, total=False):
    """Parameters for hop() traversal operation."""
    nodes: Any
    hops: int
    to_fixed_point: bool
    direction: Literal['forward', 'reverse', 'undirected']
    source_node_match: Dict[str, Any]
    edge_match: Dict[str, Any]
    destination_node_match: Dict[str, Any]
    source_node_query: str
    edge_query: str
    destination_node_query: str
    return_as_wave_front: bool
    target_wave_front: Any
    engine: str


class UmapParams(TypedDict, total=False):
    """Parameters for UMAP dimensionality reduction."""
    X: Union[None, str, List[str]]
    y: Union[None, str, List[str]]
    kind: Literal['nodes', 'edges']
    scale: Union[int, float]
    n_neighbors: int
    min_dist: Union[int, float]
    spread: Union[int, float]
    local_connectivity: int
    repulsion_strength: Union[int, float]
    negative_sample_rate: int
    n_components: int
    metric: str
    suffix: str
    play: int
    encode_position: bool
    encode_weight: bool
    dbscan: bool
    engine: Literal['auto', 'umap_learn', 'cuml']
    feature_engine: str
    inplace: bool
    memoize: bool
    umap_kwargs: Dict[str, Any]
    umap_fit_kwargs: Dict[str, Any]
    umap_transform_kwargs: Dict[str, Any]


class HypergraphParams(TypedDict, total=False):
    """Parameters for hypergraph transformation."""
    entity_types: List[str]
    opts: Dict[str, Any]
    drop_na: bool
    drop_edge_attrs: bool
    verbose: bool
    direct: bool
    engine: Literal['pandas', 'cudf', 'dask', 'auto']
    npartitions: int
    chunksize: int
    from_edges: bool
    return_as: Literal['graph', 'entities', 'events', 'edges', 'nodes']


class GetDegreesParams(TypedDict, total=False):
    """Parameters for get_degrees operation."""
    col_in: str
    col_out: str
    col: str
    engine: str


class FilterEdgesByDictParams(TypedDict, total=False):
    """Parameters for filter_edges_by_dict operation."""
    filter_dict: Dict[str, Any]


class FilterNodesByDictParams(TypedDict, total=False):
    """Parameters for filter_nodes_by_dict operation."""
    filter_dict: Dict[str, Any]


class ComputeCugraphParams(TypedDict, total=False):
    """Parameters for compute_cugraph GPU algorithms."""
    alg: str  # Required in safelist
    out_col: str
    params: Dict[str, Any]
    kind: str
    directed: bool
    G: None


class ComputeIgraphParams(TypedDict, total=False):
    """Parameters for compute_igraph algorithms."""
    alg: str  # Required in safelist
    out_col: str
    directed: bool
    use_vids: bool
    params: Dict[str, Any]


class EncodePointColorParams(TypedDict, total=False):
    """Parameters for encode_point_color operation."""
    column: str  # Required in safelist
    palette: List[str]
    as_categorical: bool
    as_continuous: bool
    categorical_mapping: Dict[str, str]
    default_mapping: str


class EncodeEdgeColorParams(TypedDict, total=False):
    """Parameters for encode_edge_color operation."""
    column: str  # Required in safelist
    palette: List[str]
    as_categorical: bool
    as_continuous: bool
    categorical_mapping: Dict[str, str]
    default_mapping: str


class EncodePointSizeParams(TypedDict, total=False):
    """Parameters for encode_point_size operation."""
    column: str  # Required in safelist
    categorical_mapping: Dict[str, Union[int, float]]
    default_mapping: Union[int, float]


class EncodePointIconParams(TypedDict, total=False):
    """Parameters for encode_point_icon operation."""
    column: str  # Required in safelist
    categorical_mapping: Dict[str, str]
    continuous_binning: List[Any]
    default_mapping: str
    as_text: bool


class LayoutIgraphParams(TypedDict, total=False):
    """Parameters for layout_igraph operation."""
    layout: str  # Required in safelist
    directed: bool
    use_vids: bool
    bind_position: bool
    x_out_col: str
    y_out_col: str
    params: Dict[str, Any]
    play: int


class LayoutCugraphParams(TypedDict, total=False):
    """Parameters for layout_cugraph GPU layouts."""
    layout: str
    params: Dict[str, Any]
    kind: str
    directed: bool
    G: None
    bind_position: bool
    x_out_col: str
    y_out_col: str
    play: int


class LayoutGraphvizParams(TypedDict, total=False):
    """Parameters for layout_graphviz operation."""
    prog: str
    args: str
    directed: bool
    strict: bool
    graph_attr: Dict[str, Any]
    node_attr: Dict[str, Any]
    edge_attr: Dict[str, Any]
    x_out_col: str
    y_out_col: str
    bind_position: bool


class Fa2LayoutParams(TypedDict, total=False):
    """Parameters for fa2_layout (ForceAtlas2) operation."""
    fa2_params: Dict[str, Any]
    circle_layout_params: Dict[str, Any]
    partition_key: str
    remove_self_edges: bool
    engine: str
    featurize: Dict[str, Any]


class GroupInABoxLayoutParams(TypedDict, total=False):
    """Parameters for group_in_a_box_layout operation."""
    partition_alg: str
    partition_params: Dict[str, Any]
    layout_alg: Any  # Can be string or callable
    layout_params: Dict[str, Any]
    x: Union[int, float]
    y: Union[int, float]
    w: Union[int, float]
    h: Union[int, float]
    encode_colors: bool
    colors: List[str]
    partition_key: str
    engine: Literal['auto', 'cpu', 'gpu', 'pandas', 'cudf']


class GetIndegreesParams(TypedDict, total=False):
    """Parameters for get_indegrees operation."""
    col: str


class GetOutdegreesParams(TypedDict, total=False):
    """Parameters for get_outdegrees operation."""
    col: str


class MaterializeNodesParams(TypedDict, total=False):
    """Parameters for materialize_nodes operation."""
    engine: str
    reuse: bool


class PruneSelfEdgesParams(TypedDict, total=False):
    """Parameters for prune_self_edges operation (no params)."""
    pass


class CollapseParams(TypedDict, total=False):
    """Parameters for collapse operation."""
    node: str
    attribute: str
    column: str
    self_edges: bool
    unwrap: bool
    verbose: bool


class DropNodesParams(TypedDict, total=False):
    """Parameters for drop_nodes operation."""
    nodes: Union[List[Any], Dict[str, Any]]  # Required in safelist


class KeepNodesParams(TypedDict, total=False):
    """Parameters for keep_nodes operation."""
    nodes: Union[List[Any], Dict[str, Any]]  # Required in safelist


class GetTopologicalLevelsParams(TypedDict, total=False):
    """Parameters for get_topological_levels operation."""
    level_col: str
    allow_cycles: bool
    warn_cycles: bool
    remove_self_loops: bool


class NameParams(TypedDict, total=False):
    """Parameters for name metadata operation."""
    name: str  # Required in safelist


class DescriptionParams(TypedDict, total=False):
    """Parameters for description metadata operation."""
    description: str  # Required in safelist


# Union type of all param types for generic usage
CallParams = Union[
    HopParams,
    UmapParams,
    HypergraphParams,
    GetDegreesParams,
    FilterEdgesByDictParams,
    FilterNodesByDictParams,
    ComputeCugraphParams,
    ComputeIgraphParams,
    EncodePointColorParams,
    EncodeEdgeColorParams,
    EncodePointSizeParams,
    EncodePointIconParams,
    LayoutIgraphParams,
    LayoutCugraphParams,
    LayoutGraphvizParams,
    Fa2LayoutParams,
    GroupInABoxLayoutParams,
    GetIndegreesParams,
    GetOutdegreesParams,
    MaterializeNodesParams,
    PruneSelfEdgesParams,
    CollapseParams,
    DropNodesParams,
    KeepNodesParams,
    GetTopologicalLevelsParams,
    NameParams,
    DescriptionParams,
]
