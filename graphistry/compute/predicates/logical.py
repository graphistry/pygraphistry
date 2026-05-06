from __future__ import annotations

from typing import Any, Dict, List, Sequence

from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT
from graphistry.utils.json import JSONVal


class AllOf(ASTPredicate):
    def __init__(self, predicates: Sequence[ASTPredicate]) -> None:
        self.predicates = list(predicates)

    def __call__(self, s: SeriesT) -> SeriesT:
        if not self.predicates:
            raise ValueError("AllOf requires at least one predicate")
        result = self.predicates[0](s)
        for predicate in self.predicates[1:]:
            result = result & predicate(s)
        return result

    def _validate_fields(self) -> None:
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        if not isinstance(self.predicates, list):
            raise GFQLTypeError(
                ErrorCode.E201,
                "predicates must be a list",
                field="predicates",
                value=type(self.predicates).__name__,
            )
        if len(self.predicates) < 1:
            raise GFQLTypeError(
                ErrorCode.E201,
                "predicates must contain at least one predicate",
                field="predicates",
                value=len(self.predicates),
            )
        for i, predicate in enumerate(self.predicates):
            if not isinstance(predicate, ASTPredicate):
                raise GFQLTypeError(
                    ErrorCode.E201,
                    "predicates must contain only ASTPredicate values",
                    field=f"predicates.{i}",
                    value=type(predicate).__name__,
                )

    def _get_child_validators(self) -> Sequence[ASTSerializable]:
        return self.predicates

    def to_json(self, validate: bool = True) -> Dict[str, JSONVal]:
        if validate:
            self.validate()
        return {
            "type": "AllOf",
            "predicates": [predicate.to_json(validate=validate) for predicate in self.predicates],
        }

    @classmethod
    def from_json(cls, d: Dict[str, JSONVal], validate: bool = True) -> "AllOf":
        from graphistry.compute.predicates.from_json import from_json as predicate_from_json

        predicates_raw = d.get("predicates")
        if not isinstance(predicates_raw, list):
            raise ValueError("AllOf predicates must be a list")
        predicates: List[ASTPredicate] = []
        for item in predicates_raw:
            if not isinstance(item, dict):
                raise ValueError("AllOf predicate items must be dictionaries")
            predicates.append(predicate_from_json(item))
        out = cls(predicates)
        if validate:
            out.validate()
        return out


def all_of(*predicates: ASTPredicate) -> AllOf:
    return AllOf(predicates)
