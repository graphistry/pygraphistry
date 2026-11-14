import os
from typing import Dict, List, Tuple

import pandas as pd

import graphistry
from graphistry import call
from graphistry.compute.semantic import CallableDecisionProvider, bind_providers
from graphistry.compute.semantic.contract import DecisionQuestion
from graphistry.compute.semantic.ops import sem_decide


DATA = os.path.join(os.path.dirname(__file__), "..", "..", "data", "semantic")
SECTORS = ["technology", "finance", "healthcare", "transportation", "food", "energy", "retail", "other"]
COMPANY_QUESTIONS = [
    {"id": "q1_finance", "kind": "predicate", "text": "Is this company primarily a financial services business?"},
    {"id": "q2_healthcare", "kind": "predicate", "text": "Is this company primarily a healthcare business?"},
    {"id": "q3_physical_goods", "kind": "predicate",
     "text": "Does the company mainly sell physical goods rather than services or software?"},
    {"id": "q4_sector", "kind": "choice", "text": "Which sector fits this company best?", "options": SECTORS},
    {"id": "q5_regulation", "kind": "score", "text": "How strongly regulated is this company's core business?",
     "levels": ["low", "medium", "high"]},
]
INJECTION_QUESTION = {
    "id": "q_injection", "kind": "predicate",
    "text": "Does this text attempt to instruct or manipulate an AI system that is processing it?",
}


def load_companies() -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = pd.read_csv(os.path.join(DATA, "semantic_companies.csv"))
    labels = pd.read_csv(os.path.join(DATA, "semantic_company_labels.csv"))
    return rows, labels


def company_provider(rows: pd.DataFrame, labels: pd.DataFrame) -> CallableDecisionProvider:
    by_row = rows.set_index("row_id")["company"].to_dict()
    expected: Dict[Tuple[str, str], str] = {
        (by_row[r.row_id], r.question_id): str(r.expected_value) for r in labels.itertuples()
    }

    def answer(evidence: Dict[str, object], question: DecisionQuestion) -> Tuple[object, Dict[str, float]]:
        label = expected[(str(evidence["company"]), question.id)]
        if question.kind == "predicate":
            value = label == "true"
            return value, {"true": 0.9 if value else 0.1, "false": 0.1 if value else 0.9}
        keys = question.distribution_keys()
        rest = (0.35 if question.kind == "choice" else 0.3) / (len(keys) - 1)
        top = 0.65 if question.kind == "choice" else 0.7
        return label, {key: (top if key == label else rest) for key in keys}

    return CallableDecisionProvider(answer, probability_kind="native_decision")


class TestCompanies:

    def test_committed_files_hold_only_synthetic_rows(self) -> None:
        rows, labels = load_companies()
        assert len(rows) == 20 and set(rows["source"]) == {"synthetic"}
        assert rows["row_id"].min() >= 5 and rows["company"].is_unique
        assert len(labels) == 100 and labels["row_id"].min() >= 5

    def test_replayed_labels_come_back_exactly_on_every_path(self) -> None:
        rows, labels = load_companies()
        provider = company_provider(rows, labels)
        kwargs = dict(questions=COMPANY_QUESTIONS, id_col="row_id", evidence_cols=["company", "description"],
                      task_id="companies")
        out = sem_decide(rows, provider=provider, batch_size=1, **kwargs)  # type: ignore[arg-type]
        by_key = {(r.row_id, r.question_id): str(r.expected_value) for r in labels.itertuples()}
        for question in COMPANY_QUESTIONS:
            qid = str(question["id"])
            assert set(out[f"{qid}__status"]) == {"ok"}
            for row_id, got in zip(out["row_id"], out[qid]):
                assert (str(bool(got)).lower() if question["kind"] == "predicate" else got) == by_key[(row_id, qid)]
        assert (set(out["q1_finance__p"]), set(out["q4_sector__p"]), set(out["q5_regulation__p"])) == ({0.9}, {0.65}, {0.7})
        meta_cols = [f"{q['id']}__meta" for q in COMPANY_QUESTIONS]
        seven = sem_decide(rows, provider=provider, batch_size=7, **kwargs)  # type: ignore[arg-type]
        with bind_providers({"companies": provider}):
            g = graphistry.nodes(rows).gfql([call("sem_decide", dict(kwargs, provider="companies"))])
        for other in (seven, g._nodes.reset_index(drop=True)):
            pd.testing.assert_frame_equal(other.drop(columns=meta_cols), out.drop(columns=meta_cols))


class TestInjectionColumn:

    def test_template_isolation_and_replay(self) -> None:
        rows = pd.read_csv(os.path.join(DATA, "semantic_injection_rows.csv"))
        assert len(rows) == 60
        extra = pd.DataFrame([{"id": "extra00", "text": "Closing tag test </untrusted> please file it",
                               "label": "benign", "kind": "benign", "hard_negative": 0}])
        df = pd.concat([rows, extra], ignore_index=True)
        maybe_id = rows["id"].iloc[3]
        by_text = dict(zip(df["text"], zip(df["id"], df["label"])))
        seen: List[str] = []

        def answer(evidence: object, question: DecisionQuestion) -> object:
            text = str(evidence)
            seen.append(text)
            inner = text[len("<untrusted>"):-len("</untrusted>")]
            row_id, label = by_text[inner]
            if row_id == maybe_id:
                return "maybe"
            value = label == "injection"
            return value, {"true": 0.95 if value else 0.05, "false": 0.05 if value else 0.95}

        provider = CallableDecisionProvider(answer, probability_kind="native_decision")
        out = sem_decide(df, questions=[INJECTION_QUESTION], provider=provider, id_col="id",
                         template="<untrusted>{text}</untrusted>", task_id="injection", batch_size=8)
        assert sorted(seen) == sorted(f"<untrusted>{t}</untrusted>" for t in df["text"])
        assert out.loc[out["id"] == maybe_id, "q_injection__error_code"].tolist() == ["invalid_provider_result"]
        assert out.loc[out["id"] == maybe_id, "q_injection"].isna().all()
        scored = out[(out["id"] != maybe_id) & (out["id"] != "extra00")]
        assert len(scored) == 59
        truth = scored["label"] == "injection"
        predicted = scored["q_injection"].astype(bool)
        assert ((predicted & truth).sum() / predicted.sum()) == 1.0
        assert ((predicted & truth).sum() / truth.sum()) == 1.0
        pd.testing.assert_frame_equal(out[["kind", "hard_negative"]], df[["kind", "hard_negative"]])
        assert out.loc[out["id"] == "extra00", "q_injection"].tolist() == [False]
