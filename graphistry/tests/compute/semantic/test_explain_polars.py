"""explain_sem_decide and explain_sem_map read a Polars table, as g.sem_decide reads the nodes of a Polars graph.

Gated on polars at module level, so it runs in the polars lane (bin/test-polars.sh) and is skipped elsewhere.
"""
import pandas as pd
import pytest

from graphistry.compute.semantic import explain_sem_decide, explain_sem_map
from graphistry.tests.compute.semantic.semantic_test_utils import P, Recording, RecordingGenerator, caps

pl = pytest.importorskip("polars")


def table() -> pd.DataFrame:
    return pd.DataFrame({"name": ["a", "b", "a"], "note": ["n0", "n1", "n2"]})


def serial() -> dict:
    return dict(capabilities=caps(native=False, concurrency_limit=1, max_rows_per_request=1))


def test_a_polars_table_plans_like_the_pandas_table() -> None:
    args = dict(questions=[P], provider=Recording(fingerprint="v1", **serial()), task_id="t", evidence_cols=["name"])
    from_pandas = explain_sem_decide(table(), **args)
    from_polars = explain_sem_decide(pl.from_pandas(table()), **args)
    lazy = explain_sem_decide(pl.from_pandas(table()).lazy(), **args)
    assert from_polars == from_pandas == lazy and from_polars["dedupe_saved_rows"] == 1


def test_sem_map_plans_a_polars_table() -> None:
    mapped = explain_sem_map(pl.from_pandas(table()), instruction="i", out_col="o", provider=RecordingGenerator(**serial()),
                             task_id="t", evidence_cols=["name"])
    assert mapped["total_rows"] == 3 and mapped["provider_rows"] == 2
