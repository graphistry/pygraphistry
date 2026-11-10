import os
from typing import List, Tuple, Union
from unittest import mock

import pandas as pd
import pytest

import graphistry
from graphistry import call
from graphistry.compute.ast import ASTObject
from graphistry.compute.exceptions import ErrorCode, GFQLCallExecutionError
from graphistry.compute.semantic import (
    CallableDecisionProvider, CallableGenerationProvider, DecisionQuestion, bind_providers)
from graphistry.compute.semantic.ops import sem_decide, sem_map

pytestmark = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1",
)

QUESTIONS = [
    {"id": "p", "kind": "predicate", "text": "Is it a company?"},
    {"id": "c", "kind": "choice", "text": "Which?", "options": ["a", "b"]},
]
NODES = pd.DataFrame({"id": [3, 1, 2, 4], "name": ["acme inc", "bob", "zeta inc", "yolo"]}, index=[30, 10, 20, 40])
EDGES = pd.DataFrame({"s": [3, 1], "d": [1, 2]})


def provider() -> CallableDecisionProvider:
    def answer(evidence, question):  # type: ignore[no-untyped-def]
        flag = "inc" in str(evidence)
        if question.kind == "predicate":
            return flag, {"true": 0.8 if flag else 0.2, "false": 0.2 if flag else 0.8}
        return ("a" if flag else "b"), {"a": 0.7 if flag else 0.3, "b": 0.3 if flag else 0.7}
    return CallableDecisionProvider(answer, probability_kind="native_decision")


def normalized(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.drop(columns=["p__meta", "c__meta"]).astype(object)
    return values.where(values.notna(), None).reset_index(drop=True)


def test_kernel_matches_pandas_and_keeps_engine() -> None:
    import cudf
    expected = sem_decide(NODES, questions=QUESTIONS, provider=provider(), evidence_cols=["name"], task_id="t")
    got = sem_decide(cudf.from_pandas(NODES), questions=QUESTIONS, provider=provider(), evidence_cols=["name"],
                     task_id="t")
    assert isinstance(got, cudf.DataFrame)
    pd.testing.assert_frame_equal(normalized(got.to_pandas()), normalized(expected), check_dtype=False)
    host = got.to_pandas()
    assert host.index.tolist() == [30, 10, 20, 40] and host["p"].tolist() == [True, False, True, False]
    assert host["p__meta"].tolist() == expected["p__meta"].tolist()
    assert str(got["p"].dtype) == "bool"
    assert str(got["p__p"].dtype) == "float64" and str(got["c__p"].dtype) == "float64"
    for col in ("c", "p__status", "c__status", "p__dist", "p__probability_kind", "p__error_code", "p__meta"):
        assert str(got[col].dtype) == "object", (col, got[col].dtype)


def test_gfql_cudf_engine_and_sem_map() -> None:
    import cudf
    g = graphistry.nodes(cudf.from_pandas(NODES), "id")
    with bind_providers({"p": provider()}):
        out = g.gfql([call("sem_decide", {"questions": QUESTIONS, "provider": "p", "task_id": "t",
                                          "evidence_cols": ["name"]})], engine="cudf")
    assert isinstance(out._nodes, cudf.DataFrame)
    assert out._nodes.to_pandas()["p"].tolist() == [True, False, True, False]
    gen = CallableGenerationProvider(lambda e, i: str(e).upper())
    mapped = sem_map(cudf.from_pandas(NODES), instruction="i", out_col="o", provider=gen, evidence_cols=["name"],
                     task_id="t")
    assert isinstance(mapped, cudf.DataFrame) and mapped.to_pandas()["o__status"].tolist() == ["ok"] * 4
    assert mapped.to_pandas()["o"].tolist() == [str({"name": n}).upper() for n in NODES["name"]]


def remove_from_pandas_shim(monkeypatch: pytest.MonkeyPatch) -> None:
    import cudf
    # cudf 26.02 dropped the native from_pandas classmethods; graphistry's lazy cudf import re-adds them once run.
    for cls in (cudf.Series, cudf.DataFrame):
        if "from_pandas" in vars(cls) and not isinstance(vars(cls)["from_pandas"], classmethod):
            monkeypatch.delattr(cls, "from_pandas")


@pytest.mark.parametrize("pandas_compatible", [False, True])
def test_direct_calls_keep_null_and_error_rows(monkeypatch: pytest.MonkeyPatch, pandas_compatible: bool) -> None:
    import cudf
    remove_from_pandas_shim(monkeypatch)

    def answer(evidence: object, question: DecisionQuestion) -> Union[bool, str, None]:
        text = str(evidence)
        if "bob" in text:
            raise RuntimeError("down")
        if "yolo" in text:
            return None
        return "inc" in text if question.kind == "predicate" else "a"

    expected = sem_decide(NODES, questions=QUESTIONS, provider=CallableDecisionProvider(answer),
                          evidence_cols=["name"], task_id="t")
    text = [col for col in expected.columns if col not in NODES.columns and expected[col].dtype == object]
    with cudf.option_context("mode.pandas_compatible", pandas_compatible):
        got = sem_decide(cudf.from_pandas(NODES), questions=QUESTIONS, provider=CallableDecisionProvider(answer),
                         evidence_cols=["name"], task_id="t")
        mapped = sem_map(cudf.from_pandas(NODES), instruction="i", out_col="o",
                         provider=CallableGenerationProvider(lambda e, i: "x"), evidence_cols=["name"], task_id="t")
        host, mapped_host = got.to_pandas(), mapped.to_pandas()
        dtypes = {col: str(got[col].dtype) for col in text}
        dtypes.update({col: str(mapped[col].dtype) for col in ("o", "o__status", "o__error_code", "o__meta")})
    assert isinstance(got, cudf.DataFrame) and isinstance(mapped, cudf.DataFrame)
    assert set(dtypes.values()) == {"object"}, dtypes
    assert host["p__status"].tolist() == ["ok", "error", "ok", "uncertain"]
    assert host["p"].isna().tolist() == [False, True, False, True] and host["p__p"].isna().all()
    for col in text:
        if not col.endswith("__meta"):
            assert host[col].tolist() == expected[col].tolist(), col
    pd.testing.assert_frame_equal(normalized(host), normalized(expected), check_dtype=False)
    assert mapped_host["o__status"].tolist() == ["ok"] * 4 and mapped_host["o__error_code"].tolist() == [None] * 4


REVERSED = ["cni emca", "bob", "cni atez", "oloy"]


def auto_case() -> Tuple[graphistry.Plottable, List[ASTObject], CallableDecisionProvider, List[str]]:
    import cudf
    seen: List[str] = []

    def answer(evidence: object, question: DecisionQuestion) -> bool:
        seen.append(str(evidence))
        return "inc" in str(evidence)

    g = graphistry.nodes(cudf.from_pandas(NODES), "id").edges(cudf.from_pandas(EDGES), "s", "d")
    query: List[ASTObject] = [
        call("sem_decide", {"questions": QUESTIONS[:1], "provider": "p", "task_id": "t", "evidence_cols": ["name"]}),
        call("rows", {"table": "nodes"}), call("select", {"items": [["out", "reverse(name)"]]})]
    return g, query, CallableDecisionProvider(answer), seen


def test_auto_polars_gpu_route_refuses_rerun_after_provider_call() -> None:
    import cudf
    from graphistry.compute.gfql.lazy import polars_gpu_available
    if not polars_gpu_available():
        pytest.skip("needs the cudf-polars GPU target")
    g, query, counted, seen = auto_case()
    with bind_providers({"p": counted}):
        with pytest.raises(GFQLCallExecutionError) as info:
            g.gfql(query)
        assert info.value.code == ErrorCode.E306 and isinstance(info.value.__cause__, NotImplementedError)
        assert len(seen) == 4
        out = g.gfql(query, engine="cudf")
    assert len(seen) == 8 and isinstance(out._nodes, cudf.DataFrame)
    assert out._nodes.to_pandas()["out"].tolist() == REVERSED


def test_auto_without_polars_gpu_runs_once_on_cudf() -> None:
    import cudf
    import graphistry.compute.gfql_unified as gu
    g, query, counted, seen = auto_case()
    with mock.patch.object(gu, "_polars_gpu_probe", return_value=False), bind_providers({"p": counted}):
        out = g.gfql(query)
    assert len(seen) == 4 and isinstance(out._nodes, cudf.DataFrame)
    assert out._nodes.to_pandas()["out"].tolist() == REVERSED
