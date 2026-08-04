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

THE ENGINE IS ALSO THE ORACLE FOR THE DECLINE TAXONOMY (`TestDeclineWitnesses`).  A
function is declined for a stated MECHANISM, and two of those mechanisms are only
observable by asking an engine what the literal-only call actually answers: whether the
value is a type the driver's contract guard rejects (`float`/`bool`/`list`), and whether
an argument-closed aggregate depends on the row set rather than on its arguments.  The
groups with NO available witness are asserted to have none, so a preference cannot pass
itself off as a correctness claim.
"""
from typing import Any, Dict, List, Optional, Set

import importlib.util

import pandas as pd
import pytest

import graphistry  # noqa: F401  (registers the plottable methods)
from graphistry.compute.gfql.cypher import lowering as lowering_module
from graphistry.compute.gfql.expr_const_fold import (
    DENIED_BY_POLICY,
    DENIED_RESULT_TYPE,
    DENIED_UNVERIFIED,
)


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
# case mapping is where SIMPLE and FULL implementations part company; 11/12 are the
# single characters `head('Male')` and `head(reverse('Male'))` fold to, so a head() case
# can pin a NON-EMPTY id set instead of an uninformative [].  Neither matches any
# pre-existing case's literal ('m'/'M'/'e'/'E' appear in none of them).
NAMES = ["Alice", "alice", "ALICE", None, "", "male", "MALE", "Male", "straße", "STRASSE",
         "M", "e"]


def _graph():
    nodes = pd.DataFrame({"node_id": list(range(1, len(NAMES) + 1)), "name": NAMES})
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


@pytest.mark.parametrize("engine", ENGINES)
def test_folded_plan_is_not_reused_across_parameter_values(engine):
    """THE CLASSIC constant-folding-plus-plan-cache BUG, pinned.

    A `$param` is a literal by the time this pass runs, so `toLower($p)` folds the
    PARAMETER VALUE into the plan. That is only safe because the compiled-plan cache
    keys on the params as well as the query text. Same query text, three different
    parameter values, on the SAME graph object (which is what owns the cache):
    each must get its own answer."""
    _require_engine(engine)
    g = _graph()
    query = ("MATCH (n) WHERE toLower(n.name) = toLower($p) "
             "RETURN n.node_id AS id ORDER BY id")
    got = {}
    for value, expected in (("ALICE", [1, 2, 3]), ("MALE", [6, 7, 8]), ("nope", [])):
        out = g.gfql(query, params={"p": value}, engine=engine)._nodes
        out = out.to_pandas() if hasattr(out, "to_pandas") else out
        got[value] = sorted(int(v) for v in out["id"].tolist())
        assert got[value] == expected, f"{value}: {got[value]} != {expected} (got so far {got})"
    # and again in a different order, to catch a cache that keys only on the text
    out = g.gfql(query, params={"p": "ALICE"}, engine=engine)._nodes
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    assert sorted(int(v) for v in out["id"].tolist()) == [1, 2, 3]


# ================================================================================
# (c) VALUE IDENTITY for every FOLDABLE_FUNCTIONS entry
#
# The unit tests pin that `head('abc')` folds to `'a'`.  This pins the thing that
# actually matters: the folded plan and the unfolded plan ANSWER THE SAME on the same
# data, on each engine.  `size` and `substring` get their own cases here rather than
# only appearing nested inside a toLower case.
# ================================================================================

SCALAR_FOLD_CASES = [
    Case("size_two_sided", "size(n.name) = size('abcde')",
         folds_expected=True, ascii_stable=True, pandas_ids=[1, 2, 3]),
    Case("substring_literal", "n.name = substring('xAlicex', 1, 5)",
         folds_expected=True, ascii_stable=True, pandas_ids=[1]),
    Case("head_literal", "n.name = head('Male')",
         folds_expected=True, ascii_stable=True, pandas_ids=[11]),
    Case("tail_literal", "toLower(n.name) = tail('xmale')",
         folds_expected=True, ascii_stable=True, pandas_ids=[6, 7, 8]),
    Case("reverse_literal", "toLower(n.name) = reverse('elam')",
         folds_expected=True, ascii_stable=True, pandas_ids=[6, 7, 8]),
    Case("head_of_reverse", "n.name = head(reverse('Male'))",
         folds_expected=True, ascii_stable=True, pandas_ids=[12]),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case", SCALAR_FOLD_CASES, ids=lambda c: c.label)
def test_scalar_folds_are_value_identical_per_engine(engine, case, monkeypatch):
    """Folded vs unfolded, same engine, same graph — plus an explicit id oracle.

    ONE DISCLOSED ASYMMETRY.  For the `head`/`tail`/`reverse`/`substring` literals the
    polars engine has no native row-op lowering of the UNFOLDED spelling and raises
    `NotImplementedError`, while the folded spelling is a plain literal comparison it
    runs natively.  So on polars this pass does not merely rename a predicate, it WIDENS
    native coverage.  That is recorded here rather than papered over, and pandas — which
    can always run both arms — still has to produce identical answers, which is where
    the value identity is actually established.
    """
    _require_engine(engine)
    g = _graph()
    query = _query(case)

    with monkeypatch.context() as mp:
        changed = _install_fold_spy(mp)
        folded_ids = _ids(g, query, engine)
    assert changed, f"{case.label}: the pass never fired; equal answers prove nothing"
    assert folded_ids == case.pandas_ids, f"{case.label}[{engine}]: {folded_ids}"

    unfolded_ids: Optional[List[int]] = None
    with monkeypatch.context() as mp:
        _disable_folding(mp)
        try:
            unfolded_ids = _ids(g, query, engine)
        except NotImplementedError:
            unfolded_ids = None

    if unfolded_ids is None:
        assert engine != "pandas", (
            f"{case.label}: pandas must be able to run the UNFOLDED spelling; without it "
            "there is no engine left that establishes value identity"
        )
    else:
        assert unfolded_ids == folded_ids, (
            f"{case.label}[{engine}]: folding changed the answer "
            f"{unfolded_ids} -> {folded_ids}"
        )


# ================================================================================
# WITNESSES FOR THE DECLINE TAXONOMY
# ================================================================================

def _scalar(expr: str, engine: str = "pandas") -> Any:
    """The engine's own value for a literal-only call — the oracle, not a re-derivation."""
    out = _graph().gfql(
        f"MATCH (n) WHERE n.node_id = 1 RETURN {expr} AS v", engine=engine
    )._nodes
    out = out.to_pandas() if hasattr(out, "to_pandas") else out
    return out["v"].tolist()[0]


def _contract_guard_rejects(value: Any) -> bool:
    """The driver's contract guard, quoted from ``fold_constants``: a folder result that
    is not a non-bool ``str``/``int`` is refused."""
    return not isinstance(value, (str, int)) or isinstance(value, bool)


#: What a plain Python fold would produce for the witness-free declines.
PYTHON_ANSWER: Dict[str, Any] = {
    "abs": 3, "sign": -1, "coalesce": 1,      # DENIED_BY_POLICY
    "tostring": "1.5", "tointeger": 1,        # DENIED_UNVERIFIED
}


class TestDeclineWitnesses:
    @pytest.mark.parametrize("name,call", sorted(DENIED_RESULT_TYPE.items()),
                             ids=sorted(DENIED_RESULT_TYPE))
    def test_result_type_group_answers_a_type_the_contract_guard_rejects(self, name, call):
        """THE WITNESS for this bucket: these calls ARE argument-closed, so the argument
        guard never sees them — what stops them is that even a PERFECT folder could not
        return their value.  ``FoldedValue`` is ``str | int``; these answer ``float``,
        ``bool`` or ``list``.

        This is also where the `round` reasoning gets put in its place: the neo4j tie /
        JDK-6430675 argument is real, and it does no work HERE, because `round(1.5)` is
        `2.0` and the guard rejects it before any tie rule can matter."""
        value = _scalar(call)
        assert _contract_guard_rejects(value), (
            f"{name}: `{call}` answers {value!r} ({type(value).__name__}), which the "
            "contract guard ACCEPTS — so the result type is not what declines it. Refile "
            "it under the mechanism that does, or fold it."
        )

    @pytest.mark.parametrize("limit,expected", [(1, 1), (5, 5), (12, 12)])
    def test_aggregates_depend_on_the_row_set_not_on_their_arguments(self, limit, expected):
        """THE WITNESS for DENIED_AGGREGATE, and the only genuinely load-bearing deny-set
        here: `count(1)` is argument-closed and answers an `int`, so it passes every
        structural guard the driver has.  The same call answers differently on different
        row sets, so folding it to a literal would be WRONG — nothing but its absence
        from FOLDABLE_FUNCTIONS prevents that."""
        out = _graph().gfql(
            f"MATCH (n) WHERE n.node_id <= {limit} RETURN count(1) AS c", engine="pandas"
        )._nodes
        assert int(out["c"].tolist()[0]) == expected

    @pytest.mark.parametrize("name,call", sorted(DENIED_BY_POLICY.items()),
                             ids=sorted(DENIED_BY_POLICY))
    def test_policy_declines_have_NO_witness_and_say_so(self, name, call):
        """THE ABSENCE OF A WITNESS, asserted rather than assumed.  For these the engine's
        answer is guard-passing AND identical to the plain Python fold, so no expression
        exists where folding would change the answer.  They stay declined as a PERF call,
        not a correctness one.  If an engine ever stops agreeing with Python here, this
        fails and the entry earns a real criterion."""
        value = _scalar(call)
        assert not _contract_guard_rejects(value), (name, call, value)
        assert value == PYTHON_ANSWER[name] and type(value) is type(PYTHON_ANSWER[name]), (
            f"{name}: `{call}` answers {value!r}, not the Python answer "
            f"{PYTHON_ANSWER[name]!r} — that IS a witness; reclassify it"
        )

    @pytest.mark.parametrize("engine", ENGINES)
    @pytest.mark.parametrize("name,call", sorted(DENIED_UNVERIFIED.items()),
                             ids=sorted(DENIED_UNVERIFIED))
    def test_unverified_declines_agree_on_every_engine_this_lane_can_reach(
        self, engine, name, call
    ):
        """`toString`/`toInteger` are the ONLY declines where a real engine-divergence
        witness COULD exist: both answer guard-passing values, so nothing structural
        stops them.  The claimed divergence is cuDF-vs-pandas float->string formatting,
        and NO CI LANE HERE RUNS A GPU — so it is UNVERIFIED, not established.

        This asserts what IS reachable: pandas and polars agree with the plain Python
        answer.  If a GPU lane is ever added and cuDF disagrees, this fails — and that
        failure is the witness, at which point the entry is upgraded from UNVERIFIED to a
        real criterion.  If it never fails, the honest reading is that these belong in
        DENIED_BY_POLICY."""
        _require_engine(engine)
        try:
            value = _scalar(call, engine)
        except NotImplementedError:
            pytest.skip(f"{engine}: no native lowering for `RETURN {call}`")
        assert not _contract_guard_rejects(value), (name, call, value)
        assert value == PYTHON_ANSWER[name], (
            f"WITNESS FOUND — {name} on {engine}: `{call}` answers {value!r}, not "
            f"{PYTHON_ANSWER[name]!r}. Move it out of DENIED_UNVERIFIED and record this "
            "pair as its criterion."
        )
