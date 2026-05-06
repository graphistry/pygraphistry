import pandas as pd
import pytest

from graphistry.compute.exceptions import GFQLTypeError
from graphistry.compute.predicates.from_json import from_json
from graphistry.compute.predicates.logical import all_of
from graphistry.compute.predicates.str import contains


def test_all_of_accepts_single_predicate() -> None:
    pred = all_of(contains("alp", na=False))

    pred.validate()
    result = pred(pd.Series(["alpha", "beta", None]))

    assert result.tolist() == [True, False, False]


def test_all_of_from_json_accepts_single_predicate() -> None:
    pred = from_json(
        {
            "type": "AllOf",
            "predicates": [
                {
                    "type": "Contains",
                    "pat": "alp",
                    "case": True,
                    "flags": 0,
                    "na": False,
                    "regex": True,
                }
            ],
        }
    )

    result = pred(pd.Series(["alpha", "beta"]))

    assert result.tolist() == [True, False]


def test_all_of_rejects_empty_predicate_list() -> None:
    pred = all_of()

    with pytest.raises(GFQLTypeError, match="at least one predicate"):
        pred.validate()
