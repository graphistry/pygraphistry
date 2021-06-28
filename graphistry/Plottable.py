from typing import Iterable, List, Optional, Union
from typing_extensions import Protocol
import pandas as pd

class Plottable(Protocol):

    def __init__(self, *args, **kwargs):
        raise RuntimeError('should not happen')
        None

    @property
    def _source(self) -> Optional[str]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _destination(self) -> Optional[str]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _node(self) -> Optional[str]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _point_title(self) -> Optional[str]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _point_label(self) -> Optional[str]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _nodes(self) -> Optional[pd.DataFrame]:
        raise RuntimeError('should not happen')
        return None

    @property
    def _edges(self) -> Optional[pd.DataFrame]:
        raise RuntimeError('should not happen')
        return None

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
