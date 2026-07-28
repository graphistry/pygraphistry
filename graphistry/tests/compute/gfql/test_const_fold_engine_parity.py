"""Cross-engine differential for plan-time constant folding (``expr_const_fold``).

THE PROPERTY UNDER TEST, stated once: **folding changes no answer on any engine.**
Every case below is run twice against the same graph and the same engine — once with
the pass live, once with it replaced by the identity — and the two answers must be
equal.  That is the only guarantee that lets a plan-time rewrite ship.

WHY THIS FILE EXISTS SEPARATELY FROM THE UNIT TESTS: the divergence the criterion is
built around is invisible without real engines.  pandas>=3 defaults to an Arrow-backed
``str`` dtype whose ``utf8_lower``/``utf8_upper`` are SIMPLE per-codepoint case
mappings, while polars' (and Python's) are FULL mappings (#1802).  On this graph
``toUpper(n.name) = 'STRASSE'`` genuinely answers differently on pandas and polars —
so a cross-engine equality assertion over non-ASCII input fails for reasons that have
nothing to do with this pass.  Non-ASCII cases therefore assert fold-vs-no-fold per
engine (which must hold, because the pass DECLINES them) and deliberately do NOT
assert cross-engine agreement; ASCII cases assert both.

ENGAGEMENT IS INSTRUMENTED, NOT ASSUMED.  Each case declares whether the pass is
expected to fire, and a spy asserts it — so an "identical answers" pass cannot come
from a rewrite that never ran (`folds_expected=False` rows are the negative control).
"""
from typing import Dict, List, Optional, Set

import importlib.util

import pandas as pd
import pytest

import graphistry  # noqa: F401  (registers the plottable methods)
from graphistry.compute.gfql.cypher import lowering as lowering_module


ENGINES = ["pandas", "polars", "cudf", "polars-gpu"]


def _unavailable_reason(engine: str) -> Optional[str]:
    """``None`` when the engine can run here, else a NAMED reason."""
    if engine == "pandas":
        return None
    if importlib.util.find_spec("polars") is None:
        return f"{engine}: polars not installed"
    if engine == "cudf" and importlib.util.find_spec("cudf") is None:
        return "cudf: RAPIDS cudf not installed"
    if engine == "polars-gpu" and importlib.util.find_spec("cudf_polars") is None:
        return "polars-gpu: cudf_polars not installed"
    return None


def _require_engine(engine: str) -> None:
    """Skip with a NAMED reason so an absent engine is visible in the report rather
    than silently counting as coverage."""
    reason = _unavailable_reason(engine)
    if reason is not None:
        pytest.skip(reason)


# node 4 is NULL, node 5 is the empty string; 9/10 are the German sharp-s pair whose
# case mapping is where SIMPLE and FULL implementations part company.
NAMES = ["Alice", "alice", "ALICE", None, "", "male", "MALE", "Male", "straße", "STRASSE"]


def _graph():
    nodes = pd.DataFrame({"node_id": list(range(1, 11)), "name": NAMES})
    edges = pd.DataFrame({"src": [1, 2, 3], "dst": [2, 3, 4]})
    return graphistry.nodes(nodes, "node_id").edges(edges, "src", "dst")


class Case:
    """One predicate spelling, with what it is expected to do to the plan."""

    def __init__(
        self,
        label: str,
        predicate: str,
        *,
        folds_expected: bool,
        ascii_stable: bool,
        pandas_ids: Optional[List[int]] = None,
    ) -> None:
        self.label = label
        self.predicate = predicate
        self.folds_expected = folds_expected   # does the pass fire on this text?
        self.ascii_stable = ascii_stable       # may every engine be required to agree?
        self.pandas_ids = pandas_ids           # literal oracle (ASCII cases only)

    def __repr__(self) -> str:
        return self.label


CASES = [
    # --- ASCII, one-sided: nothing to fold; the literal is compared AS WRITTEN -------
    Case("lower_one_sided_lc", "toLower(n.name) = 'alice'",
         folds_expected=False, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("lower_one_sided_UC", "toLower(n.name) = 'ALICE'",
         folds_expected=False, ascii_stable=True, pandas_ids=[]),
    Case("lower_one_sided_Mixed", "toLower(n.name) = 'Alice'",
         folds_expected=False, ascii_stable=True, pandas_ids=[]),
    Case("lower_one_sided_male", "toLower(n.name) = 'male'",
         folds_expected=False, ascii_stable=True, pandas_ids=[6, 7, 8]),
    Case("lower_one_sided_MALE", "toLower(n.name) = 'MALE'",
         folds_expected=False, ascii_stable=True, pandas_ids=[]),
    Case("lower_one_sided_Male", "toLower(n.name) = 'Male'",
         folds_expected=False, ascii_stable=True, pandas_ids=[]),
    Case("lower_one_sided_empty", "toLower(n.name) = ''",
         folds_expected=False, ascii_stable=True, pandas_ids=[5]),
    # --- ASCII, two-sided: THE FOLD FIRES; must answer exactly as it did unfolded ----
    Case("lower_two_sided_UC", "toLower(n.name) = toLower('ALICE')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("lower_two_sided_Mixed", "toLower(n.name) = toLower('Alice')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("lower_two_sided_MALE", "toLower(n.name) = toLower('MALE')",
         folds_expected=True, ascii_stable=True, pandas_ids=[6, 7, 8]),
    Case("lower_two_sided_empty", "toLower(n.name) = toLower('')",
         folds_expected=True, ascii_stable=True, pandas_ids=[5]),
    Case("upper_two_sided", "toUpper(n.name) = toUpper('alice')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("gql_lower_two_sided", "lower(n.name) = lower('ALICE')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("gql_upper_two_sided", "upper(n.name) = upper('alice')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("nested_substring", "toLower(n.name) = toLower(substring('ALICEXX', 0, 5))",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    # 'STRASSE' is pure ASCII, so it folds; the COLUMN side is what diverges (below).
    Case("lower_two_sided_STRASSE", "toLower(n.name) = toLower('STRASSE')",
         folds_expected=True, ascii_stable=True, pandas_ids=[10]),
    # --- #1802 territory: the COLUMN side diverges between engines. Fold-vs-no-fold
    # must still hold per engine; cross-engine agreement is NOT asserted. -------------
    Case("upper_one_sided_STRASSE", "toUpper(n.name) = 'STRASSE'",
         folds_expected=False, ascii_stable=False),
    Case("upper_two_sided_strasse", "toUpper(n.name) = toUpper('strasse')",
         folds_expected=True, ascii_stable=False),
    # --- non-ASCII literals: the pass DECLINES (negative control on engagement) ------
    Case("lower_two_sided_nonascii_sharp_s", "toLower(n.name) = toLower('STRAßE')",
         folds_expected=False, ascii_stable=False),
    Case("lower_two_sided_nonascii_turkish", "toLower(n.name) = toLower('İSTANBUL')",
         folds_expected=False, ascii_stable=False),
    Case("upper_two_sided_nonascii_dotless", "toUpper(n.name) = toUpper('ıstanbul')",
         folds_expected=False, ascii_stable=False),
    Case("lower_one_sided_nonascii", "toLower(n.name) = 'straße'",
         folds_expected=False, ascii_stable=False),
    Case("lower_two_sided_nonascii_greek", "toLower(n.name) = toLower('ΣΊΣΥΦΟΣ')",
         folds_expected=False, ascii_stable=False),
]


def _query(case: Case) -> str:
    return f"MATCH (n) WHERE {case.predicate} RETURN n.node_id AS id ORDER BY id"


def _ids(g, query: str, engine: str) -> List[int]:
    out = g.gfql(query, engine=engine)._nodes
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    return sorted(int(v) for v in out["id"].tolist())


def _install_fold_spy(monkeypatch) -> List[str]:
    """Record every node the pass actually REWRITES (not merely visits)."""
    real = lowering_module.fold_constants
    changed: List[str] = []

    def spy(node):
        out = real(node)
        if out != node:
            changed.append(repr(out))
        return out

    monkeypatch.setattr(lowering_module, "fold_constants", spy)
    return changed


def _disable_folding(monkeypatch) -> None:
    monkeypatch.setattr(lowering_module, "fold_constants", lambda node: node)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", CASES, ids=lambda c: c.label)
def test_folding_is_answer_preserving(engine, case, monkeypatch):
    """THE differential: same graph, same engine, folding on vs folding off."""
    _require_engine(engine)
    g = _graph()

    with monkeypatch.context() as mp:
        changed = _install_fold_spy(mp)
        folded_ids = _ids(g, _query(case), engine)
    assert bool(changed) == case.folds_expected, (
        f"{case.label}: fold engagement {bool(changed)} != expected "
        f"{case.folds_expected} (rewrites: {changed})"
    )

    with monkeypatch.context() as mp:
        _disable_folding(mp)
        unfolded_ids = _ids(g, _query(case), engine)

    assert folded_ids == unfolded_ids, (
        f"{case.label}[{engine}]: folding changed the answer "
        f"{unfolded_ids} -> {folded_ids}"
    )


@pytest.mark.parametrize("case", [c for c in CASES if c.pandas_ids is not None],
                         ids=lambda c: c.label)
def test_literal_oracle_pandas(case):
    """Explicit expected ids, so a uniformly-wrong fold/no-fold PAIR still fails.
    pandas only: it is the engine whose case behaviour is pinned by the fixture's
    ASCII values (see #1802 for why non-ASCII is excluded here)."""
    assert _ids(_graph(), _query(case), "pandas") == case.pandas_ids


@pytest.mark.parametrize("case", [c for c in CASES if c.ascii_stable], ids=lambda c: c.label)
def test_cross_engine_agreement_on_ascii(case):
    """Every available engine must agree on the ASCII cases, folded.

    Reports which engines actually ran: a run where only pandas is available proves
    nothing cross-engine, so it is skipped rather than passed."""
    ran: Dict[str, List[int]] = {}
    for engine in ENGINES:
        if _unavailable_reason(engine) is not None:
            continue
        ran[engine] = _ids(_graph(), _query(case), engine)
    if len(ran) < 2:
        pytest.skip(f"only {sorted(ran)} available; cross-engine agreement unprovable")
    distinct: Set[str] = {repr(v) for v in ran.values()}
    assert len(distinct) == 1, f"{case.label}: engines disagree {ran}"


@pytest.mark.parametrize("query", [
    "MATCH (n) WHERE toUpper(n.name) = 'STRASSE' RETURN n.node_id AS id ORDER BY id",
    "MATCH (n) WHERE toUpper(n.name) = toUpper('strasse') RETURN n.node_id AS id ORDER BY id",
])
def test_non_ascii_case_divergence_is_pre_existing_not_introduced(query, monkeypatch):
    """The #1802 divergence must be UNCHANGED by this PR: the same cross-engine picture
    with folding on and with folding off.  Asserting the DELTA rather than the value
    keeps this green whichever way #1802 is eventually fixed."""
    if _unavailable_reason("polars") is not None:
        pytest.skip("polars not installed")
    g = _graph()
    folded = {e: _ids(g, query, e) for e in ("pandas", "polars")}
    monkeypatch.setattr(lowering_module, "fold_constants", lambda node: node)
    unfolded = {e: _ids(g, query, e) for e in ("pandas", "polars")}
    assert folded == unfolded, (
        f"this PR changed the #1802 cross-engine picture: {unfolded} -> {folded}"
    )
