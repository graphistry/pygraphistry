import json
import threading
from dataclasses import replace
from enum import Enum
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import pytest

from graphistry.compute.semantic.contract import GenerationRequest, GenerationResult, compute_generation_hash
from graphistry.compute.semantic.ops import sem_map
from graphistry.compute.semantic.providers.callable import CallableGenerationProvider
from graphistry.tests.compute.semantic.semantic_test_utils import TEXT_DTYPES, Recording, RecordingGenerator, caps


class ScriptedGenerator:
    name = "raw"
    model = "raw-m"

    def __init__(self, script) -> None:  # type: ignore[no-untyped-def]
        self.script = script
        self.count = 0

    def capabilities(self):  # type: ignore[no-untyped-def]
        return caps(concurrency_limit=1)

    def generate_batch(self, requests, *, deadline_s, budget):  # type: ignore[no-untyped-def]
        self.count += 1
        return self.script(self.count - 1, requests)


def generated(request: GenerationRequest, value: object = "ok", **overrides: object) -> GenerationResult:
    fields: Dict[str, object] = dict(request_id=request.request_id, value=value, status="ok", error_code=None, model="m",
                                     provider="raw", task_version="1", input_hash="provider-made", latency_ms=None,
                                     cost_usd=None, cache_hit=None, retry_count=None)
    fields.update(overrides)
    return GenerationResult(**fields)  # type: ignore[arg-type]


def frame() -> pd.DataFrame:
    return pd.DataFrame({"text": ["alpha", "beta", "gamma", "delta", None, "zeta"], "other": [1, 2, 3, 4, None, 6]},
                        index=[10, 11, 12, 13, 14, 15])


def shout(evidence: object, instruction: str) -> Optional[str]:
    text = str(evidence)
    if "gamma" in text:
        raise RuntimeError("boom")
    return None if "delta" in text else text.upper()


def run(df: pd.DataFrame, provider: object, **kwargs: object) -> pd.DataFrame:
    params: Dict[str, object] = dict(instruction="Shout", out_col="o", evidence_cols=["text"], task_id="t")
    params.update(kwargs)
    return sem_map(df, provider=provider, **params)  # type: ignore[arg-type]


class TestOutputs:

    def test_columns_statuses_identity_and_meta(self) -> None:
        provider = RecordingGenerator(shout)
        out = run(frame(), provider, template="<t>{text}</t>", evidence_cols=None, batch_size=2)
        assert list(out.columns) == ["text", "other", "o", "o__status", "o__error_code", "o__meta"]
        assert out.index.tolist() == [10, 11, 12, 13, 14, 15] and str(out["o"].dtype) in TEXT_DTYPES
        assert out["o__status"].tolist() == ["ok", "ok", "error", "uncertain", "skipped", "ok"]
        assert out["o"].tolist()[:2] == ["<T>ALPHA</T>", "<T>BETA</T>"] and out["o__error_code"].iloc[2] == "callable_exception"
        metas = [json.loads(m) for m in out["o__meta"]]
        assert [m["status_reason"] for m in metas][3:5] == ["null_answer", "null_evidence"]
        assert all("confidence" not in m for m in metas)
        assert metas[0]["input_hash"] == compute_generation_hash("<t>alpha</t>", "Shout", "1")
        assert sorted(r.evidence for call in provider.calls for r in call) == sorted(  # type: ignore[type-var]
            ["<t>alpha</t>", "<t>beta</t>", "<t>gamma</t>", "<t>delta</t>", "<t>zeta</t>"])
        assert {(r.instruction, r.task_id) for call in provider.calls for r in call} == {("Shout", "t")}

    def test_batch_invariance_and_defaults(self) -> None:
        frames = [run(frame(), RecordingGenerator(shout), batch_size=size).drop(columns=["o__meta"]) for size in (1, 4, None)]
        pd.testing.assert_frame_equal(frames[0], frames[1])
        pd.testing.assert_frame_equal(frames[0], frames[2])
        provider = RecordingGenerator()
        run(pd.DataFrame({"text": [f"t{i}" for i in range(40)]}), provider)
        assert sorted(len(call) for call in provider.calls) == [8, 16, 16]
        limited = RecordingGenerator(capabilities=caps(max_rows_per_request=3))
        run(pd.DataFrame({"text": [f"t{i}" for i in range(7)]}), limited, max_output_tokens=np.int64(12))
        assert sorted(len(call) for call in limited.calls) == [1, 3, 3]
        assert {r.max_output_tokens for call in limited.calls for r in call} == {12}
        assert {type(r.max_output_tokens) for call in limited.calls for r in call} == {int}


class TestFailures:

    def test_whole_call_faults_and_invalid_results(self) -> None:
        def script(index: int, requests: Sequence[GenerationRequest]) -> object:
            good = [generated(r) for r in requests]
            if index == 0:
                raise RuntimeError("down")
            return {1: [generated(replace(requests[0], request_id="nope"))], 2: good + good[:1],
                    3: [generated(r, 5) for r in requests], 4: [generated(r, request_id=[r.request_id]) for r in requests],
                    5: [generated(r, None, retry_count=-1, latency_ms=float("inf")) for r in requests]
                    }.get(index, good)

        out = run(pd.DataFrame({"text": list("abcdefghijklmn")}), ScriptedGenerator(script), batch_size=2)
        assert out["o__error_code"].tolist()[:12] == ["provider_exception"] * 2 + ["invalid_provider_result"] * 10
        assert out["o__status"].tolist()[12:] == ["ok", "ok"]
        meta = json.loads(out["o__meta"].iloc[10])
        assert (meta["retry_count"], meta["latency_ms"]) == (None, None)

    def test_non_string_value_from_a_callable_is_invalid(self) -> None:
        out = run(pd.DataFrame({"text": ["a"]}), RecordingGenerator(lambda e, i: 5))
        assert out["o__error_code"].tolist() == ["invalid_provider_result"]

    def test_callable_values_are_read_as_plain_python(self) -> None:
        out = run(pd.DataFrame({"text": ["a", "b"]}), RecordingGenerator(
            lambda e, i: float("nan") if e["text"] == "a" else np.str_("x")))  # type: ignore[arg-type]
        assert out["o__status"].tolist() == ["uncertain", "ok"] and out["o"].isna().tolist() == [True, False]
        assert type(out["o"].iloc[1]) is str

    def test_provider_values_that_raise_when_read_error_only_their_row(self) -> None:
        class Unreadable:
            def tolist(self) -> object:
                raise TypeError("cannot read")

        class FloatRaises(float):
            def __float__(self) -> float:
                raise RuntimeError("no floats")

        hostile = {"b": dict(value=Unreadable()),
                   "c": dict(value=None, status="uncertain", status_reason="tie", latency_ms=FloatRaises(1.0))}
        out = run(pd.DataFrame({"text": ["a", "b", "c", "d"]}), ScriptedGenerator(lambda i, reqs: [
            generated(r, **hostile.get(r.evidence["text"], {})) for r in reqs]))  # type: ignore[index]
        assert out["o__status"].tolist() == ["ok", "error", "error", "ok"]
        assert out["o__error_code"].tolist()[1:3] == ["invalid_provider_result"] * 2

    def test_answers_are_plain_strings_and_kernel_fields_win(self) -> None:
        class Label(str, Enum):
            B = "b"

        class FromResponse(GenerationResult):
            def __init__(self, base: GenerationResult) -> None:
                super().__init__(**vars(base))

        out = run(pd.DataFrame({"text": ["a"]}), ScriptedGenerator(lambda i, reqs: [
            FromResponse(generated(r, Label.B, task_version="999")) for r in reqs]), task_version="2")
        meta = json.loads(out["o__meta"].iloc[0])
        assert (out["o__status"].tolist(), out["o"].tolist(), type(out["o"].iloc[0])) == (["ok"], ["b"], str)
        assert (meta["task_version"], meta["input_hash"]) == ("2", compute_generation_hash({"text": "a"}, "Shout", "2"))

    def test_kernel_built_errors_carry_the_provider_name_and_model(self) -> None:
        meta = json.loads(run(pd.DataFrame({"text": ["a"]}), ScriptedGenerator(
            lambda i, reqs: [generated(r, 5) for r in reqs]))["o__meta"].iloc[0])
        assert (meta["provider"], meta["model"]) == ("raw", "raw-m")


class TestLimitsAndValidation:

    def test_remaining_deadline_and_budget_reach_the_provider(self) -> None:
        provider = RecordingGenerator(delay=0.05)
        run(pd.DataFrame({"text": ["a", "b", "c"]}), provider, batch_size=1, max_concurrency=1, deadline_s=5.0,
            budget={"max_requests": 3})
        deadlines = [deadline for deadline, _ in provider.limits]
        assert [budget.max_requests for _, budget in provider.limits if budget] == [3, 2, 1]
        assert deadlines[0] > deadlines[1] > deadlines[2] and deadlines[1] <= 4.95

    def test_max_rows_requests_and_deadline(self) -> None:
        df = pd.DataFrame({"text": [f"t{i}" for i in range(6)]})
        assert run(df, RecordingGenerator(), max_rows=2)["o__status"].tolist() == ["ok"] * 2 + ["skipped"] * 4
        out = run(df, RecordingGenerator(), batch_size=1, budget={"max_requests": 3})
        assert out["o__status"].tolist() == ["ok"] * 3 + ["skipped"] * 3
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()

        def slow(evidence: Dict[str, object], instruction: str) -> str:
            if evidence["text"] in ("t3", "t4", "t5"):
                entered.set()
                release.wait(10)
                finished.set()
            return "x"

        try:
            out = run(df, RecordingGenerator(slow), batch_size=1, max_concurrency=1, deadline_s=0.8)
            assert entered.is_set() and not finished.is_set()
            assert out["o__status"].tolist() == ["ok"] * 3 + ["skipped"] * 3
            assert {json.loads(m)["status_reason"] for m in out["o__meta"].iloc[3:]} == {"deadline"}
        finally:
            release.set()
        assert finished.wait(10)

    @pytest.mark.parametrize("kwargs", [
        dict(out_col="text"), dict(out_col="__gfql_x__"), dict(out_col=""), dict(instruction=""),
        dict(max_output_tokens=0), dict(template="{0}", evidence_cols=None), dict(deadline_s=0), dict(overwrite="yes"),
        dict(provider=Recording()), dict(batch_size=0), dict(max_rows=True), dict(task_id=""), dict(column_max_chars=0),
    ])
    def test_raises_before_any_call(self, kwargs: dict) -> None:
        provider = RecordingGenerator()
        params: Dict[str, object] = dict(provider=provider)
        params.update(kwargs)
        with pytest.raises(ValueError):
            run(frame(), **params)  # type: ignore[arg-type]
        assert provider.calls == []

    def test_existing_output_columns_need_overwrite_and_evidence_is_truncated(self) -> None:
        with pytest.raises(ValueError, match="overwrite"):
            run(frame().assign(o__meta="x"), RecordingGenerator())
        provider = RecordingGenerator()
        out = run(frame().assign(o__meta="x"), provider, overwrite=True, column_max_chars=2)
        assert out["o__meta"].iloc[0] != "x"
        assert sorted(r.evidence["text"] for call in provider.calls for r in call) == [  # type: ignore[index]
            "al", "be", "de", "ga", "ze"]


class TestGenerationContract:

    def test_checks_and_unanswered_results(self) -> None:
        from graphistry.compute.semantic.contract import (
            GenerationResult, align_results, check_generation_result, unanswered_generation_result)
        req = GenerationRequest(request_id="r0", evidence="e", instruction="i", task_id="t", task_version="1")
        good = generated(req)
        assert check_generation_result(req, good) == []
        for bad, fragment in ((generated(req, 3), "string or None"), (generated(req, request_id="x"), "request_id"),
                              (generated(req, None, status="uncertain", status_reason="tie"), "not allowed"),
                              (generated(req, None, status="uncertain", status_reason="min_p"), "not allowed"),
                              (generated(req, status="uncertain", status_reason="refusal"), "value None"),
                              (generated(req, status=pd.NA), "status must be a string")):
            problems = check_generation_result(req, bad)
            assert len(problems) == 1 and fragment in problems[0], problems
        failed = unanswered_generation_result(req, provider="p", model="m", error_code="x")
        skipped = unanswered_generation_result(req, provider="p", model="m", status_reason="deadline")
        assert check_generation_result(req, failed) == check_generation_result(req, skipped) == []
        assert check_generation_result(req, generated(req, None, status="uncertain")) == []
        with pytest.raises(ValueError, match="callable"):
            CallableGenerationProvider(5)  # type: ignore[arg-type]
        assert failed.input_hash == compute_generation_hash("e", "i", "1")
        assert align_results([req], [good], GenerationResult) == {"r0": {"": good}}
        assert align_results([req], [good, good], GenerationResult) is None
        assert align_results([req], [], GenerationResult) == {"r0": None}

    def test_generation_hash_golden_value(self) -> None:
        import hashlib
        expected = hashlib.sha256('{"evidence":"café","instruction":"Summarize","task_version":"2"}'.encode()).hexdigest()
        assert compute_generation_hash("café", "Summarize", "2") == expected
