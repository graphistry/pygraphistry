from typing import Iterable, List, Optional, Union
from typing_extensions import Protocol
import pandas as pd

class Plottable(Protocol):
    @property
    def _point_title(self) -> Optional[str]:
        return None

    @property
    def _point_label(self) -> Optional[str]:
        return None

    @property
    def _nodes(self) -> Optional[pd.DataFrame]:
        return None

    @property
    def _edges(self) -> Optional[pd.DataFrame]:
        return None

    def nodes(self, nodes: pd.DataFrame, node: Optional[str]) -> 'Plottable':
        return self

    def edges(self, nodes: pd.DataFrame, source: Optional[str], destination: Optional[str]) -> 'Plottable':
        return self

    def bind(self, **kwargs) -> 'Plottable':
        return self
