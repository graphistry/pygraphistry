from typing import TYPE_CHECKING, Any, Callable, Iterable, List, Optional, Union
import pandas as pd

try:
    from umap import UMAP
except:
    UMAP = Any

class Plottable(object):

    _edges : Any
    _nodes : Any
    _source : Optional[str]
    _destination : Optional[str]
    _node : Optional[str]
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

    _node_embedding : Optional[Any]
    _node_encoder : Optional[Any]
    _node_features : Optional[Any]
    _node_imputer : Optional[Any]
    _node_scaler : Optional[Any]
    _node_target : Optional[Any]
    _node_target_encoder : Optional[Any]

    _edge_embedding : Optional[Any]
    _edge_encoders : Optional[Any]
    _edge_features : Optional[Any]
    _edge_imputer : Optional[Any]
    _edge_scaler : Optional[Any]
    _edge_target : Optional[Any]
    _edge_target_encoder : Optional[Any]

    _weighted_adjacency_nodes : Optional[Any]
    _weighted_adjacency_edges : Optional[Any]
    _weighted_edges_df : Optional[Any]
    _weighted_edges_df_from_nodes : Optional[Any]
    _weighted_edges_df_from_edges : Optional[Any]

    _umap : Optional[UMAP]

    _adjacency : Optional[Any]
    _entity_to_index : dict
    _index_to_entity : dict


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
        self, edges: Union[Callable, Any], source: Optional[str] = None, destination: Optional[str] = None,
        *args, **kwargs
    ) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def pipe(self, graph_transform: Callable, *args, **kwargs) -> 'Plottable':
        if 1 + 1:
            raise RuntimeError('should not happen')
        return self

    def bind(self, source=None, destination=None, node=None,
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

    def materialize_nodes(self, reuse: bool = True) -> 'Plottable':
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
