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

    def bind(self, **kwargs) -> 'Plottable':
        raise RuntimeError('should not happen')
        return self
