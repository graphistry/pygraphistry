from abc import abstractmethod
import pandas as pd


class ASTPredicate():
    """
    Internal, not intended for use outside of this module.
    These are fancy columnar predicates used in {k: v, ...} node/edge df matching when going beyond primitive equality
    """

    @abstractmethod
    def __call__(self, s: pd.Series) -> pd.Series:
        raise NotImplementedError()

    @abstractmethod
    def to_json(self, validate=True) -> dict:
        raise NotImplementedError()
    
    @classmethod
    def from_json(cls, d: dict) -> 'ASTPredicate':
        raise NotImplementedError()

    def validate(self) -> None:
        pass
