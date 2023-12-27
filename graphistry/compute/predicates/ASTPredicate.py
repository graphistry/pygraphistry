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
        raise NotImplementedError()
