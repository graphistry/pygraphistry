from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd

from graphistry.utils.json import JSONVal, serialize_to_json_val


class ASTPredicate(ABC):
    """
    Internal, not intended for use outside of this module.
    These are fancy columnar predicates used in {k: v, ...} node/edge df matching when going beyond primitive equality
    """

    reserved_fields = ['type']

    @abstractmethod
    def __call__(self, s: pd.Series) -> pd.Series:
        raise NotImplementedError()

    def validate(self) -> None:
        pass

    def to_json(self, validate=True) -> Dict[str, JSONVal]:
        """
        Returns JSON-compatible dictionry {"type": "ClassName", "arg1": val1, ...}
        Emits all non-reserved instance fields
        """
        if validate:
            self.validate()
        data: Dict[str, JSONVal] = {'type': self.__class__.__name__}
        for key, value in self.__dict__.items():
            if key not in self.reserved_fields:
                data[key] = serialize_to_json_val(value)
        return data

    @classmethod
    def from_json(cls, d: Dict[str, JSONVal]) -> 'ASTPredicate':
        """
        Given c.to_json(), hydrate back c

        Corresponding c.__class__.__init__ must accept all non-reserved instance fields
        """
        constructor_args = {k: v for k, v in d.items() if k not in cls.reserved_fields}
        return cls(**constructor_args)
