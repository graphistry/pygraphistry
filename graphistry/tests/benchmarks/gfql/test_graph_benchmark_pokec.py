from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import (
    TYPE_CHECKING,
    Callable,
    List,
    Literal,
    Optional,
    Protocol,
    runtime_checkable,
    Tuple,
    TypedDict,
    TypeVar,
)
import unittest

import pandas as pd

if TYPE_CHECKING:
    from graphistry.Engine import Engine


IndexPath = Literal["scan", "index"]
QueryKind = Literal["global", "seed"]
ExecutionTarget = Literal["cpu", "gpu", "mixed"]
FailureStatus = Literal["unsupported", "error", "oom"]
ResultT = TypeVar("ResultT")


class TimingReceipt(TypedDict):
    median_ms: float
    samples_ms: List[float]


class ExecutionReceipt(TypedDict, total=False):
    status: Literal["ok", "fallback"]
    execution_target: ExecutionTarget
    synchronization: Literal["none", "cuda-device"]
    fallback_reason: str


class FailureReceipt(TypedDict):
    status: FailureStatus
    unsupported: bool
    error_type: str
    error: str
    kind: QueryKind


@runtime_checkable
class PokecBenchmarkModule(Protocol):
    def _selected_engines(self, selection: str) -> Tuple[str, ...]:
        ...

    def _frame_engine(self, engine: str) -> "Engine":
        ...

    def _to_native_frames(
        self,
        nodes: pd.DataFrame,
        edges: pd.DataFrame,
        engine: str,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ...

    def _execution_metadata(
        self,
        engine: str,
        *,
        uses_native_index: bool,
        execution_paths: Tuple[IndexPath, ...] = (),
    ) -> ExecutionReceipt:
        ...

    def _time(
        self,
        fn: Callable[[], ResultT],
        runs: int,
        warmup: int,
        synchronize: Optional[Callable[[], None]] = None,
    ) -> Tuple[ResultT, TimingReceipt]:
        ...

    def _failure_result(
        self,
        exc: Exception,
        kind: QueryKind,
    ) -> FailureReceipt:
        ...


def _load_pokec_module() -> PokecBenchmarkModule:
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "benchmarks" / "gfql" / "graph_benchmark_pokec.py"
    spec = importlib.util.spec_from_file_location("graph_benchmark_pokec", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading module at {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not isinstance(module, PokecBenchmarkModule):
        raise RuntimeError(f"Benchmark module at {module_path} lacks its helper API")
    return module


_POKEC = _load_pokec_module()


class TestPokecBenchmarkHelpers(unittest.TestCase):
    def test_selected_engines(self) -> None:
        cases = [
            ("pandas", ("pandas",)),
            ("cudf", ("cudf",)),
            ("polars", ("polars",)),
            ("polars-gpu", ("polars-gpu",)),
            ("both", ("pandas", "cudf")),
            ("all", ("pandas", "cudf", "polars", "polars-gpu")),
        ]
        for selection, expected in cases:
            with self.subTest(selection=selection):
                self.assertEqual(_POKEC._selected_engines(selection), expected)

    def test_selected_engines_rejects_unknown(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "unsupported GFQL benchmark engine selection",
        ):
            _POKEC._selected_engines("gpu")

    def test_polars_gpu_uses_polars_resident_frames(self) -> None:
        from graphistry.Engine import Engine

        self.assertEqual(_POKEC._frame_engine("polars"), Engine.POLARS)
        self.assertEqual(_POKEC._frame_engine("polars-gpu"), Engine.POLARS)

    def test_pandas_frame_conversion_is_identity(self) -> None:
        nodes = pd.DataFrame({"id": [1, 2]})
        edges = pd.DataFrame({"src": [1], "dst": [2]})

        converted_nodes, converted_edges = _POKEC._to_native_frames(
            nodes,
            edges,
            "pandas",
        )

        self.assertIs(converted_nodes, nodes)
        self.assertIs(converted_edges, edges)

    def test_polars_gpu_index_execution_is_explicit_cpu_fallback(self) -> None:
        metadata = _POKEC._execution_metadata(
            "polars-gpu",
            uses_native_index=True,
            execution_paths=("index",),
        )

        self.assertEqual(metadata["status"], "fallback")
        self.assertEqual(metadata["execution_target"], "cpu")
        self.assertIn("not a GPU-only result", str(metadata["fallback_reason"]))

    def test_polars_gpu_cypher_execution_stays_gpu(self) -> None:
        metadata = _POKEC._execution_metadata(
            "polars-gpu",
            uses_native_index=False,
        )

        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["execution_target"], "gpu")
        self.assertNotIn("fallback_reason", metadata)

    def test_polars_gpu_scan_path_is_a_gpu_result(self) -> None:
        metadata = _POKEC._execution_metadata(
            "polars-gpu",
            uses_native_index=True,
            execution_paths=("scan",),
        )

        self.assertEqual(metadata["status"], "ok")
        self.assertEqual(metadata["execution_target"], "gpu")

    def test_polars_gpu_mixed_path_is_explicit_fallback(self) -> None:
        metadata = _POKEC._execution_metadata(
            "polars-gpu",
            uses_native_index=True,
            execution_paths=("index", "scan"),
        )

        self.assertEqual(metadata["status"], "fallback")
        self.assertEqual(metadata["execution_target"], "mixed")
        self.assertIn("mixed", str(metadata["fallback_reason"]))

    def test_time_retains_all_measured_samples(self) -> None:
        calls = 0

        def measured() -> int:
            nonlocal calls
            calls += 1
            return calls

        result, timing = _POKEC._time(measured, runs=3, warmup=2)

        self.assertEqual(result, 5)
        self.assertEqual(calls, 5)
        self.assertEqual(len(timing["samples_ms"]), 3)
        self.assertGreaterEqual(timing["median_ms"], 0.0)

    def test_time_synchronizes_gpu_boundaries(self) -> None:
        sync_calls = 0

        def synchronize() -> None:
            nonlocal sync_calls
            sync_calls += 1

        _POKEC._time(lambda: 1, runs=2, warmup=1, synchronize=synchronize)

        self.assertEqual(sync_calls, 5)

    def test_failure_statuses_are_distinct(self) -> None:
        from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

        unsupported = _POKEC._failure_result(NotImplementedError("no"), "seed")
        structured_unsupported = _POKEC._failure_result(
            GFQLValidationError(ErrorCode.E108, "not implemented"),
            "seed",
        )
        error = _POKEC._failure_result(ValueError("bad"), "seed")
        oom = _POKEC._failure_result(MemoryError("full"), "seed")

        self.assertEqual(unsupported["status"], "unsupported")
        self.assertEqual(structured_unsupported["status"], "unsupported")
        self.assertTrue(structured_unsupported["unsupported"])
        self.assertEqual(error["status"], "error")
        self.assertEqual(oom["status"], "oom")
