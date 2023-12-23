from abc import abstractmethod
import pandas as pd

from graphistry.compute.ASTSerializable import ASTSerializable


class ASTPredicate(ASTSerializable):
    """
    Internal, not intended for use outside of this module.
    These are fancy columnar predicates used in {k: v, ...} node/edge df matching when going beyond primitive equality
    """

    @abstractmethod
    def __call__(self, s: pd.Series) -> pd.Series:
        raise NotImplementedError()
