from typing import Any, Iterable, List, Optional, Union
from typing_extensions import Protocol
import pandas as pd

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

    def __init__(self, *args, **kwargs):
        raise RuntimeError('should not happen')
        None

    def nodes(self, nodes: pd.DataFrame, node: Optional[str] = None) -> 'Plottable':
        raise RuntimeError('should not happen')
        return self

    def edges(self, nodes: pd.DataFrame, source: Optional[str] = None, destination: Optional[str] = None) -> 'Plottable':
        raise RuntimeError('should not happen')
        return self

    def bind(self, source=None, destination=None, node=None,
             edge_title=None, edge_label=None, edge_color=None, edge_weight=None, edge_size=None, edge_opacity=None, edge_icon=None,
             edge_source_color=None, edge_destination_color=None,
             point_title=None, point_label=None, point_color=None, point_weight=None, point_size=None, point_opacity=None, point_icon=None,
             point_x=None, point_y=None):
        raise RuntimeError('should not happen')
        return self
