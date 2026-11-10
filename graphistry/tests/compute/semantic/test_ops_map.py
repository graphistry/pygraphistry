import json
import threading
from typing import Dict, List, Optional, Sequence

import pandas as pd
import pytest

from graphistry.compute.semantic.contract import Budget, GenerationRequest, GenerationResult, ProviderCapabilities
from graphistry.compute.semantic.ops import sem_map
from graphistry.compute.semantic.providers.callable import CallableGenerationProvider


class Recording(CallableGenerationProvider):
    def __init__(self, fn, **kwargs: object) -> None:  # type: ignore[no-untyped-def]
        super().__init__(fn, **kwargs)  # type: ignore[arg-type]
        self.calls: List[List[str]] = []
        self.evidence: List[object] = []
        self.lock = threading.Lock()

    def generate_batch(self, requests: Sequence[GenerationRequest], *, deadline_s: Optional[float],
                       budget: Optional[Budget]) -> List[GenerationResult]:
        with self.lock:
            self.calls.append([r.request_id for r in requests])
            self.evidence.extend(r.evidence for r in requests)
        return super().generate_batch(requests, deadline_s=deadline_s, budget=budget)


class Raw:
    name = "raw"
    model = "raw-m"

    def __init__(self, script) -> None:  # type: ignore[no-untyped-def]
        self.script = script
        self.count = 0

    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(False, 1, None, False, 1)

    def estimate_cost_usd(self, requests: Sequence[GenerationRequest]) -> Optional[float]:
        return 0.0

    def generate_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
        self.count += 1
        return self.script(self.count - 1, requests)


def frame() -> pd.DataFrame:
    return pd.DataFrame({"text": ["alpha", "beta", "gamma", "delta", None, "zeta"],
                         "other": [1, 2, 3, 4, None, 6]}, index=[10, 11, 12, 13, 14, 15])


def scripted(evidence: object, instruction: str) -> Optional[str]:
    text = str(evidence)
    if "gamma" in text:
        raise RuntimeError("boom")
    if "delta" in text:
        return None
    return text.upper()


class TestOutputs:

    def test_columns_statuses_and_identity(self) -> None:
        provider = Recording(scripted)
        out = sem_map(frame(), instruction="Shout", out_col="loud", provider=provider,
                      template="<t>{text}</t>", task_id="t", batch_size=2)
        assert list(out.columns) == ["text", "other", "loud", "loud__status", "loud__error_code", "loud__meta"]
        assert out.index.tolist() == [10, 11, 12, 13, 14, 15]
        assert out["loud__status"].tolist() == ["ok", "ok", "error", "uncertain", "skipped", "ok"]
        assert out["loud"].tolist()[:2] == ["<T>ALPHA</T>", "<T>BETA</T>"]
        assert out["loud__error_code"].iloc[2] == "callable_exception"
        reasons = [json.loads(m)["status_reason"] for m in out["loud__meta"]]
        assert reasons[3] == "null_answer" and reasons[4] == "null_evidence"
        assert all("confidence" not in json.loads(m) for m in out["loud__meta"])
        assert sorted(map(str, provider.evidence)) == sorted(
            ["<t>alpha</t>", "<t>beta</t>", "<t>gamma</t>", "<t>delta</t>", "<t>zeta</t>"])

    def test_batch_invariance(self) -> None:
        frames = []
        for size in (1, 4, None):
            out = sem_map(frame(), instruction="Shout", out_col="loud", provider=Recording(scripted),
                          evidence_cols=["text"], task_id="t", batch_size=size)
            frames.append(out.drop(columns=["loud__meta"]))
        pd.testing.assert_frame_equal(frames[0], frames[1])
        pd.testing.assert_frame_equal(frames[0], frames[2])


class TestFailuresAndLimits:

    def test_whole_call_raise_and_bad_return(self) -> None:
        def script(index: int, requests: Sequence[GenerationRequest]) -> List[GenerationResult]:
            if index == 0:
                raise RuntimeError("down")
            if index == 1:
                return [GenerationResult(request_id="nope", value="x", status="ok", error_code=None, model="m",
                                         provider="raw", task_version="1", input_hash="h", latency_ms=None,
                                         cost_usd=None, cache_hit=None, retry_count=None)]
            return [GenerationResult(request_id=r.request_id, value="ok", status="ok", error_code=None, model="m",
                                     provider="raw", task_version="1", input_hash="h", latency_ms=None,
                                     cost_usd=None, cache_hit=None, retry_count=None) for r in requests]

        df = pd.DataFrame({"text": list("abcdef")})
        out = sem_map(df, instruction="i", out_col="o", provider=Raw(script), evidence_cols=["text"], task_id="t",
                      batch_size=2, max_concurrency=1)
        assert out["o__error_code"].tolist()[:4] == ["provider_exception"] * 2 + ["invalid_provider_result"] * 2
        assert out["o__status"].tolist()[4:] == ["ok", "ok"]

    def test_max_rows_budget_deadline(self) -> None:
        df = pd.DataFrame({"text": [f"t{i}" for i in range(6)]})
        out = sem_map(df, instruction="i", out_col="o", provider=Recording(lambda e, i: "x"), evidence_cols=["text"],
                      task_id="t", max_rows=2)
        assert out["o__status"].tolist() == ["ok", "ok"] + ["skipped"] * 4
        out = sem_map(df, instruction="i", out_col="o", provider=Recording(lambda e, i: "x"), evidence_cols=["text"],
                      task_id="t", batch_size=1, budget={"max_requests": 3})
        assert out["o__status"].tolist() == ["ok"] * 3 + ["skipped"] * 3

        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def slow(evidence: Dict[str, object], instruction: str) -> str:
            if evidence["text"] in ("t3", "t4", "t5"):
                entered.set()
                release.wait(10)
                finished.set()
            return "x"

        try:
            out = sem_map(df, instruction="i", out_col="o", provider=Recording(slow), evidence_cols=["text"],
                          task_id="t", batch_size=1, max_concurrency=1, deadline_s=0.8)
            assert entered.is_set() and not finished.is_set()
            assert out["o__status"].tolist() == ["ok"] * 3 + ["skipped"] * 3
            assert {json.loads(m)["status_reason"] for m in out["o__meta"].iloc[3:]} == {"deadline"}
        finally:
            release.set()
        assert finished.wait(10)

    def test_validation_before_any_call(self) -> None:
        provider = Recording(lambda e, i: "x")
        for kwargs in (dict(out_col="text"), dict(out_col="__gfql_x__"), dict(instruction=""),
                       dict(max_output_tokens=0), dict(template="{0}")):
            params = dict(instruction="i", out_col="o", provider=provider, evidence_cols=["text"], task_id="t")
            params.update(kwargs)
            if "template" in kwargs:
                params.pop("evidence_cols")
            with pytest.raises(ValueError):
                sem_map(frame(), **params)  # type: ignore[arg-type]
        assert provider.calls == []

    def test_malformed_ids_metadata_and_estimates(self) -> None:
        def bad_ids(index: int, requests: Sequence[GenerationRequest]) -> List[GenerationResult]:
            return [GenerationResult(request_id=[r.request_id], value="x", status="ok", error_code=None,  # type: ignore[arg-type]
                                     model="m", provider="raw", task_version="1", input_hash="h", latency_ms=None,
                                     cost_usd=None, cache_hit=None, retry_count=None) for r in requests]

        out = sem_map(pd.DataFrame({"text": ["a", "b"]}), instruction="i", out_col="o", provider=Raw(bad_ids),
                      evidence_cols=["text"], task_id="t")
        assert out["o__error_code"].tolist() == ["invalid_provider_result"] * 2

        def bad_meta(index: int, requests: Sequence[GenerationRequest]) -> List[GenerationResult]:
            return [GenerationResult(request_id=r.request_id, value=5, status="ok", error_code=None,  # type: ignore[arg-type]
                                     model="m", provider="raw", task_version="1", input_hash="h",
                                     latency_ms=float("inf"), cost_usd=None, cache_hit=None, retry_count=-1)
                    for r in requests]

        out = sem_map(pd.DataFrame({"text": ["a"]}), instruction="i", out_col="o", provider=Raw(bad_meta),
                      evidence_cols=["text"], task_id="t")
        meta = json.loads(out["o__meta"].iloc[0])
        assert meta["latency_ms"] is None and meta["retry_count"] is None

        provider = Recording(lambda e, i: "x")
        provider.estimate_cost_usd = lambda requests: float("nan")  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="cannot estimate"):
            sem_map(pd.DataFrame({"text": ["a"]}), instruction="i", out_col="o", provider=provider,
                    evidence_cols=["text"], task_id="t", budget={"max_cost_usd": 1.0})
        assert provider.calls == []

    def test_non_string_value_is_error(self) -> None:
        out = sem_map(pd.DataFrame({"text": ["a"]}), instruction="i", out_col="o",
                      provider=Recording(lambda e, i: 5), evidence_cols=["text"], task_id="t")  # type: ignore[arg-type,return-value]
        assert out["o__error_code"].tolist() == ["invalid_provider_result"]
