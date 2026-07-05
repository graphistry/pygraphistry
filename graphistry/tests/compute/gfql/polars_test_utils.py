"""Shared helpers for the GFQL polars-engine test suite — ONE definition each (previously
re-implemented per file) so the parity logic cannot silently diverge.

LOUD-FAILURE CONTRACTS (do not "simplify" away):
* ``run_status`` maps ONLY NotImplementedError to ``("nie",)``; every other exception surfaces
  as ``("err", <TypeName>)`` — a broader nie-mapping would turn crashes into allowed declines.
* ``assert_parity_or_nie`` skips (not passes) when the PANDAS ORACLE itself errors; callers
  keep a pandas-oracle canary test so a global oracle break still fails loudly.
* ``typed_frame_sig`` compares typed cell VALUES (floats rounded, NA→None) — the STRONGEST
  signature tier; files with weaker stringified sigs keep them deliberately (documented there).
NOT a pytest module (no ``test_`` prefix): importers do their own pytest.importorskip("polars")
— nothing here imports polars at module scope."""
import pandas as pd
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
    """Canonical VALUE-level frame repr for cross-engine compare: to-pandas, sorted columns,
    rounded floats (FP tolerance), sorted rows (order-insensitive), NaN/NA -> None — actual
    cell values, not just id-sets/shape."""
    df = to_pandas_any(df)
    if df is None:
        return None
    df = df.reindex(sorted(df.columns), axis=1).copy()
    import numpy as np
    for c in df.columns:
        if df[c].dtype.kind == "f":
            with np.errstate(over="ignore"):  # .round(6) on 1e308-scale cells warns benignly
                df[c] = df[c].round(6)
    cols = tuple(df.columns)
    # rows as tuples (NaN/NA -> None), sorted with a None-safe type-safe key (per-row agg(join)
    # is fragile on empty/mixed frames). astype(object) FIRST so nullable-extension dtypes
    # (cudf->pandas) turn pd.NA into a real None — .where on the extension array would re-coerce
    # to pd.NA, and a pd.NA in the sig makes `res == base` raise "bool of NA ambiguous".
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
    """EVERY engine in ``engines``: result == pandas oracle OR honest NIE — never a silent
    divergence / non-NIE crash. (Callers pass their module's engine list so the dgx GPU lane
    extends coverage without editing call sites.)"""
    base = run_status(g, query, "pandas")
    if base[0] == "err":
        pytest.skip(f"{label}: pandas oracle itself errored ({base[1]})")
    for eng in engines:
        res = run_status(g, query, eng)
        if res[0] == "nie":
            continue  # honest decline — allowed
        assert res[0] != "err", f"{label}[{eng}]: non-NIE {res[1]} where pandas={base[0]}"
        assert res == base, f"{label}[{eng}]: SILENT DIVERGENCE {eng}{res} != pandas{base}"


# --- graph-shape comparison helpers (chain/hop parity lanes). No try/except and no non-empty
# defaults inside: an exception-swallowing or `return set()` fallback would turn a parity
# assert into `set() == set()`. ---

def node_id_set(g):
    """Node-id set (weakest tier — pair with attr/multiset checks per the caller's table)."""
    df = g._nodes
    if df is None:
        return set()
    return set(to_pandas_any(df)[g._node].tolist())


def edge_pair_set(g):
    """(src, dst) endpoint-pair SET — blind to multiplicity; see edge_pair_multiset."""
    df = g._edges
    if df is None or len(df) == 0:
        return set()
    df = to_pandas_any(df)
    return set(zip(df[g._source].tolist(), df[g._destination].tolist()))


def edge_pair_multiset(g):
    """Edge MULTISET (Counter) — catches a dropped parallel/self-loop copy or a count-preserving
    edge SWAP, which len() and edge_pair_set both miss; the min_hops recompute-all combine
    (fuzz seeds 24/48) diverged exactly here."""
    from collections import Counter
    df = g._edges
    if df is None or len(df) == 0:
        return Counter()
    df = to_pandas_any(df)
    return Counter(zip(df[g._source].tolist(), df[g._destination].tolist()))


def node_attr_map(g):
    """Null-aware per-node ATTRIBUTE map — sees a node present in BOTH outputs with a different
    (or NULL) attr cell, invisible to node_id_set; the min_hops null-attr-on-source-side-endpoint
    rule (fuzz seed-48 n5/n7: kind=y carried as NaN, so a downstream kind=y filter rejects them)
    lives exactly here. Normalizes NaN/None→None and int/float→float (pandas upcasts an int col
    to float once a NaN-stub row is concatenated; polars keeps Int64+null — else spurious
    5 != 5.0). Excludes internal `__gfql_*` columns (pandas ASTEdge.execute leaks its auto
    hop-label into the min_hops result, polars doesn't) — impl detail, not a parity concern."""
    df = g._nodes
    if df is None:
        return {}
    df = to_pandas_any(df)
    key = g._node
    cols = sorted(c for c in df.columns if c != key and not c.startswith("__gfql_"))

    def norm(v):
        if v is None:
            return None
        if isinstance(v, float) and pd.isna(v):
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, (int, float)):
            return float(v)
        try:
            if pd.isna(v):
                return None
        except (TypeError, ValueError):
            pass
        return v

    return {row[key]: tuple(norm(row[c]) for c in cols) for _, row in df.iterrows()}


def named_flag_set(g, col):
    """Ids whose named-op flag column `col` is truthy (node table if present, else edges)."""
    df = g._nodes if col in (g._nodes.columns if g._nodes is not None else []) else g._edges
    if df is None or col not in df.columns:
        return None
    df = to_pandas_any(df)
    key = g._node if g._node in df.columns else g._source
    return set(df[df[col].fillna(False).astype(bool)][key].tolist())


def assert_surfaces_agree(res_a, res_b, label):
    """Two surfaces (chain vs DAG / chain vs cypher) must AGREE: both honest-NIE, or both ok
    with the SAME value signature — never one ok and one NIE (the silent-bridge class).
    STRICTER than the old inline ``if nie/elif ok``, which let an 'err' fall through and pass:
    a non-NIE error on EITHER surface fails here."""
    assert res_a[0] in ("ok", "nie"), f"{label}: first surface non-NIE error {res_a}"
    assert res_b[0] in ("ok", "nie"), f"{label}: second surface non-NIE error {res_b}"
    assert res_a[0] == res_b[0], f"{label}: surface divergence {res_a[0]} != {res_b[0]} (silent-bridge class)"
    if res_a[0] == "ok":
        assert res_a[1] == res_b[1], f"{label}: surfaces both ok but signatures differ"
