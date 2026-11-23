"""Shared observable probability-policy checks for decision providers; no network."""
import pandas as pd

from graphistry.compute.semantic.contract import DecisionProvider, ProbabilityKind
from graphistry.compute.semantic.ops import sem_decide
from graphistry.compute.semantic.rules import ProbabilityRequirement


QUESTION = {"id": "p", "kind": "predicate", "text": "Is the evidence positive?"}


def check_probability_policy(provider: DecisionProvider, requirement: ProbabilityRequirement,
                             expected_kind: ProbabilityKind) -> pd.DataFrame:
    output = sem_decide(pd.DataFrame({"text": ["positive"]}), questions=[QUESTION], provider=provider,
                        task_id="conformance", evidence_cols=["text"], require_probabilities=requirement)
    measured = expected_kind in ("native_decision", "token_logprob")
    unavailable = requirement is not False and (not measured or (
        requirement == "native" and expected_kind != "native_decision"))
    assert output["p__status"].tolist() == (["error"] if unavailable else ["ok"])
    if unavailable:
        assert output["p__error_code"].tolist() == ["probabilities_unavailable"]
        assert output["p"].isna().all() and output["p__p"].isna().all() and output["p__dist"].isna().all()
        assert output["p__probability_kind"].tolist() == ["none"]
    else:
        assert output["p"].tolist() == [True] and output["p__probability_kind"].tolist() == [expected_kind]
        assert output["p__p"].notna().all() if measured or expected_kind == "verbalized" else output["p__p"].isna().all()
    return output
