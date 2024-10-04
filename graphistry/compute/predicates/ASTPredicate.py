from abc import abstractmethod
import pandas as pd
from typing import Any, TYPE_CHECKING

from graphistry.compute.ASTSerializable import ASTSerializable


if TYPE_CHECKING:
    SeriesT = pd.Series
else:
    SeriesT = Any


class ASTPredicate(ASTSerializable):
    """
    Internal, not intended for use outside of this module.
    These are fancy columnar predicates used in {k: v, ...} node/edge df matching when going beyond primitive equality
    """

    @abstractmethod
    def __call__(self, s: SeriesT) -> SeriesT:
        """
        Abstract method to apply the predicate to a pandas Series or compatible object.

        Args:
            s (SeriesT): The input pandas Series or compatible object.

        Returns:
            SeriesT: The resulting Series after applying the predicate.
        """
        raise NotImplementedError()
