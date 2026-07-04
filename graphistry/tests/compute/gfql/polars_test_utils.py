"""Shared helpers for the GFQL polars-engine test suite (conformance matrix, chain,
cypher-conformance, row-pipeline, call-modality, GPU lanes).

These were re-implemented per test file; ONE definition each here so the parity logic
cannot silently diverge between files. LOUD-FAILURE CONTRACTS (do not "simplify" away):

* ``run_status`` maps ONLY ``NotImplementedError`` to ``("nie",)``; every other
  exception surfaces as ``("err", <TypeName>)`` — mapping broader exceptions to "nie"
  would convert crashes into allowed declines and gut the conformance matrix.
* ``assert_parity_or_nie`` skips (not passes) when the PANDAS ORACLE itself errors;
  callers keep a pandas-oracle canary test so a global oracle break still fails loudly.
* ``typed_frame_sig`` compares typed cell VALUES (floats rounded, NA→None) — it is the
  STRONGEST signature tier. Files with weaker stringified signatures keep them
  deliberately (documented there); do not swap tiers silently.

NOT a pytest module (no ``test_`` prefix): importers do their own
``pytest.importorskip("polars")`` — nothing here imports polars at module scope.
"""
import pytest


def to_pandas_any(df):
    """Normalize any engine frame (polars / cudf / pandas) to pandas for comparison.
    None passes through (callers distinguish a missing frame from an empty one)."""
    if df is None:
        return None
    if "pandas" in type(df).__module__:
        return df
    if hasattr(df, "to_pandas"):  # polars.DataFrame, cudf.DataFrame
        return df.to_pandas()
    return df


def typed_frame_sig(df):
    """Canonical VALUE-level repr of a frame for cross-engine comparison: normalize to
    pandas, sort columns, round floats (FP tolerance across engines), sort rows
    (order-insensitive), NaN/NA -> None. Compares actual cell values, not just
    id-sets/shape."""
    df = to_pandas_any(df)
    if df is None:
        return None
    df = df.reindex(sorted(df.columns), axis=1).copy()
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(6)
    cols = tuple(df.columns)
    # rows as tuples (NaN/NA -> None), then sort the LIST of tuples (order-insensitive) with a
    # None-safe, type-safe key. Avoids per-row agg(join) which is fragile on empty/mixed frames.
    # astype(object) FIRST so pandas nullable-extension dtypes (cudf->pandas yields these) turn
    # pd.NA into a real None; .where(notna, None) on the original extension array would re-coerce
    # back to pd.NA, and a pd.NA in the signature makes `res == base` raise "bool of NA ambiguous".
    obj = df.astype(object).where(df.notna(), None)
    rows = [tuple(r) for r in obj.to_numpy().tolist()]
    rows.sort(key=lambda t: tuple((v is None, type(v).__name__, str(v)) for v in t))
    return (cols, tuple(rows))


def graph_sig(g):
    """Full value-level signature of a graph/row result (both frames, values compared)."""
    return (typed_frame_sig(g._nodes), typed_frame_sig(g._edges))


def run_status(g, query, engine):
    """('ok', sig) | ('nie',) | ('err', ExcTypeName). ONLY NotImplementedError is 'nie'."""
    try:
        return ("ok", graph_sig(g.gfql(query, engine=engine)))
    except NotImplementedError:
        return ("nie",)
    except Exception as ex:  # any non-NIE error is itself a conformance failure to surface
        return ("err", type(ex).__name__)


def available_nonpandas_engines():
    """polars always; cudf / polars-gpu when the GPU stack is importable (the dgx GPU lane)."""
    engines = ["polars"]
    try:
        import cudf  # noqa: F401
        engines.append("cudf")
    except Exception:
        pass
    import importlib.util
    if importlib.util.find_spec("cudf_polars") is not None:
        engines.append("polars-gpu")
    return engines


def assert_parity_or_nie(g, query, label, engines):
    """For EVERY engine in ``engines``: result == pandas oracle, OR honest NIE. Never a
    silent divergence / non-NIE crash. (Callers pass their module's engine list so the
    dgx GPU lane extends coverage without editing call sites.)"""
    base = run_status(g, query, "pandas")
    if base[0] == "err":
        pytest.skip(f"{label}: pandas oracle itself errored ({base[1]})")
    for eng in engines:
        res = run_status(g, query, eng)
        if res[0] == "nie":
            continue  # honest decline — allowed
        assert res[0] != "err", f"{label}[{eng}]: non-NIE {res[1]} where pandas={base[0]}"
        assert res == base, f"{label}[{eng}]: SILENT DIVERGENCE {eng}{res} != pandas{base}"


def assert_surfaces_agree(res_a, res_b, label):
    """Two surfaces (chain vs DAG / chain vs cypher) must AGREE: both honest-NIE, or both
    ok with the SAME value signature — never one ok and the other NIE (the silent-bridge
    bug class). STRICTER than the old inline pattern: an 'err' on EITHER surface fails
    here (the old ``if nie/elif ok`` let a chain 'err' fall through and pass)."""
    assert res_a[0] in ("ok", "nie"), f"{label}: first surface non-NIE error {res_a}"
    assert res_b[0] in ("ok", "nie"), f"{label}: second surface non-NIE error {res_b}"
    assert res_a[0] == res_b[0], f"{label}: surface divergence {res_a[0]} != {res_b[0]} (silent-bridge class)"
    if res_a[0] == "ok":
        assert res_a[1] == res_b[1], f"{label}: surfaces both ok but signatures differ"
