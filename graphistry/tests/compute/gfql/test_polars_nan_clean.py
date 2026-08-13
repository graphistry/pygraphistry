"""_pl_nan_to_null: NaN->null correctness, identity-stable, and FRESH per call.

The clean-verdict memo this file used to pin returned a stale "clean" after
polars' in-place mutation APIs and served wrong rows through the public API --
the undeclared ingest path now probes the frame's CURRENT content every call.
"""
import pytest

pl = pytest.importorskip("polars")

from graphistry.compute.gfql.lazy.engine.polars.nan_clean import _pl_nan_to_null  # noqa: E402


def test_no_float_cols_is_noop_same_object():
    df = pl.DataFrame({"a": [1, 2, 3], "s": ["x", "y", "z"]})
    out = _pl_nan_to_null(df)
    assert out is df


def test_nan_present_is_cleaned_to_null():
    df = pl.DataFrame({"w": [1.0, float("nan"), 3.0]})
    out = _pl_nan_to_null(df)
    # NaN -> null: null_count reflects the converted cell; no NaN remains.
    assert out.get_column("w").null_count() == 1
    assert not bool(out.get_column("w").is_nan().any())


def test_clean_float_frame_identity_stable():
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0]})  # float col, no NaN
    assert _pl_nan_to_null(df) is df           # clean -> unchanged (identity-stable)
    assert _pl_nan_to_null(df) is df


def test_inplace_nan_injection_is_seen_on_the_next_call():
    """The regression the memo caused: mark clean, mutate in place with a
    polars in-place API, and the next call must see the NaN -- not a stale
    "clean" verdict."""
    df = pl.DataFrame({"w": [1.0, 2.0, 3.0]})
    assert _pl_nan_to_null(df) is df  # verdict: clean
    df.replace_column(0, pl.Series("w", [1.0, float("nan"), 3.0]))
    out = _pl_nan_to_null(df)
    assert out.get_column("w").null_count() == 1, "stale clean verdict -- a memo is back"
    assert not bool(out.get_column("w").is_nan().any())


def test_inplace_nan_injection_end_to_end_rows():
    """agent-02 F-01 e2e recipe: the stale verdict changed WHERE rows through
    the public API. NaN must read as missing (pandas-oracle semantics) after an
    in-place injection on a graph that already answered once."""
    import graphistry

    nodes = pl.DataFrame({"id": [0, 1, 2, 3], "w": [1.0, 2.0, 3.0, 4.0]})
    edges = pl.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    q = "MATCH (a) WHERE a.w >= 2 RETURN a.id AS x ORDER BY x"
    first = g.gfql(q, engine="polars")._nodes.to_pandas()["x"].tolist()
    assert first == [1, 2, 3]
    nodes.replace_column(1, pl.Series("w", [1.0, float("nan"), 3.0, 4.0]))
    second = g.gfql(q, engine="polars")._nodes.to_pandas()["x"].tolist()
    assert second == [2, 3], "NaN injected in place must read as missing, not stale-clean"


def test_distinct_frames_do_not_cross_contaminate():
    clean = pl.DataFrame({"w": [1.0, 2.0]})
    _pl_nan_to_null(clean)
    dirty = pl.DataFrame({"w": [float("nan"), 2.0]})
    out = _pl_nan_to_null(dirty)
    assert out.get_column("w").null_count() == 1


def test_lazyframe_keeps_unconditional_rewrite():
    lf = pl.DataFrame({"w": [1.0, float("nan")]}).lazy()
    out = _pl_nan_to_null(lf)
    assert isinstance(out, pl.LazyFrame)
    assert out.collect().get_column("w").null_count() == 1
