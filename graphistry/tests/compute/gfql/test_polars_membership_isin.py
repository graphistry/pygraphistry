"""#2082: ``membership.is_in_ids`` — the ONE spelling of id-set membership for the polars engine.

polars < 1.28 rejects a List-typed (``implode()``) ``is_in`` RHS whose length differs from the
column (``ComputeError: shapes don't match: expected N elements in 'is_in' comparison, got 1``);
polars >= 1.28 deprecates the bare-``Series`` RHS. RAPIDS 25.02 ships polars 1.21, RAPIDS 26.02
ships 1.35, so the engine must be correct AND warning-free on both.

Oracles are the python-list form ``expr.is_in(ids.to_list())`` (never warned, never failed on
any 1.21..1.35 release in the #2082 per-release sweep) and hand-written literals.
Engine agreement is not used as evidence.
"""
import ast
import functools
import pathlib
import warnings
from typing import List

import pandas as pd
import pytest

import graphistry
from .polars_test_utils import edge_pair_set

pl = pytest.importorskip("polars")
from graphistry.compute.gfql.lazy.engine.polars import membership  # noqa: E402
from graphistry.compute.gfql.lazy.engine.polars.membership import (  # noqa: E402
    IMPLODED_RHS_FLOOR, imploded_rhs_supported, is_in_ids,
)


# --- the version predicate is a pure function of the version string -------------------------

@pytest.mark.parametrize("version,expected", [
    ("1.21.0", False), ("1.27.1", False), ("1.28.0rc1", False),
    ("1.28.0", True), ("1.28.1", True), ("1.29.0", True), ("1.35.2", True), ("2.0.0", True),
    ("1.31.0+cu12", True), ("1.30.0.dev0", True), ("1.28.0.dev0", False),
])
def test_imploded_rhs_supported_switches_exactly_at_1_28_0(version, expected):
    assert imploded_rhs_supported(version) is expected
    assert str(IMPLODED_RHS_FLOOR) == "1.28.0"


@pytest.mark.parametrize("version", ["not-a-version", "", "1.2.3.4.dev-nope"])
def test_imploded_rhs_supported_falls_back_to_the_bare_rhs_on_an_unparseable_version(version):
    """The bare RHS is correct on every release and only warns from 1.28; the imploded one is a
    hard ComputeError below it. So an unreadable version takes the bare side."""
    assert imploded_rhs_supported(version) is False


def test_installed_polars_branch_matches_the_predicate():
    assert membership._installed_polars_implodes() is imploded_rhs_supported(pl.__version__)


@pytest.mark.parametrize("implodes", [False, True], ids=["sub-1.28-bare", "1.28+-imploded"])
def test_id_set_spells_both_arms_whatever_polars_is_installed(monkeypatch, implodes):
    """The < 1.28 arm is what #2082 turns on, and no CI lane installs a polars that old: without
    this pin, collapsing id_set to a single spelling stays green everywhere and silently brings
    the RAPIDS 25.02 ComputeError back."""
    monkeypatch.setattr(membership, "_installed_polars_implodes", lambda: implodes)
    ids = pl.Series("id", [1, 2], dtype=pl.Int64)
    rhs = membership.id_set(ids)
    assert isinstance(rhs.dtype, pl.List) is implodes
    if implodes:
        assert rhs.len() == 1 and rhs.to_list() == [[1, 2]]
    else:
        assert rhs is ids


def _deprecations(caught):
    """polars raises its ``is_in`` deprecation from Rust: under a ``-W error`` filter it is PRINTED,
    not raised (verified against a mutant), so only record-and-assert can observe it.
    ``CategoricalRemappingWarning`` (local categoricals) is a perf hint, not part of the contract."""
    return [str(w.message) for w in caught
            if issubclass(w.category, DeprecationWarning) and "is_in" in str(w.message)]


# --- set semantics + no warning on the INSTALLED polars, every dtype/shape ------------------

def _scenarios():
    # (label, endpoint values, id values, dtype, expected kept mask from a hand oracle)
    yield "int/mixed", [1, 2, 9, None], [1, 2, 3], pl.Int64, [True, True, False, None]
    yield "int/empty-ids", [1, 2], [], pl.Int64, [False, False]
    yield "int/null-in-ids", [1, 2, None], [1, None], pl.Int64, [True, False, None]
    yield "str/mixed", ["a", "zz", None], ["a", "b"], pl.Utf8, [True, False, None]
    yield "str/empty-ids", ["a"], [], pl.Utf8, [False]
    yield "cat/mixed", ["a", "zz", None], ["a", "b"], pl.Categorical, [True, False, None]
    yield "cat/empty-ids", ["a"], [], pl.Categorical, [False]
    yield "float/mixed", [1.0, 2.5, None], [1.0, 3.0], pl.Float64, [True, False, None]


@pytest.mark.parametrize("label,vals,ids,dtype,expected", list(_scenarios()),
                         ids=[s[0] for s in _scenarios()])
def test_is_in_ids_is_set_membership_and_never_warns(label, vals, ids, dtype, expected):
    df = pl.DataFrame({"x": pl.Series(vals, dtype=dtype)})
    id_series = pl.Series("id", ids, dtype=dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = df.select(is_in_ids(pl.col("x"), id_series).alias("m")).get_column("m").to_list()
    oracle = df.select(pl.col("x").is_in(id_series.to_list()).alias("m")).get_column("m").to_list()
    assert got == expected, label
    assert got == oracle, label
    assert _deprecations(caught) == [], label


def test_is_in_ids_keeps_callers_fill_null_contract():
    """Sites chain ``.fill_null(False)``: a null endpoint must come out null BEFORE that fill."""
    df = pl.DataFrame({"x": pl.Series([None, 1], dtype=pl.Int64)})
    ids = pl.Series("id", [1], dtype=pl.Int64)
    raw = df.select(is_in_ids(pl.col("x"), ids).alias("m")).get_column("m").to_list()
    filled = df.select(is_in_ids(pl.col("x"), ids).fill_null(False).alias("m")).get_column("m").to_list()
    assert raw == [None, True] and filled == [False, True]


def test_is_in_ids_on_a_lazyframe_and_a_large_universe():
    """The RHS is a literal, not a same-length column: an id set far longer than the frame."""
    ids = pl.Series("id", list(range(0, 100_000, 2)), dtype=pl.Int64)
    lf = pl.LazyFrame({"x": pl.Series([0, 1, 99_998, 99_999, 250_000], dtype=pl.Int64)})
    got = lf.filter(is_in_ids(pl.col("x"), ids)).collect().get_column("x").to_list()
    assert got == [0, 99_998]


# --- the routed sites, end to end on the polars engine ---------------------------------------

_NODES = pd.DataFrame({"id": [0, 1, 2], "kind": ["a", "b", "a"]})
# every endpoint-miss shape: closed, dangling dst, dangling src, both missing, self-loop on missing
_EDGES = pd.DataFrame({"s": [0, 1, 7, 8, 9], "d": [1, 5, 2, 6, 9]})
_CLOSED = {(0, 1)}


def _ids(values, id_dtype):
    """Int64 literals re-typed as the id dtype under test (Categorical only casts from strings)."""
    e = pl.Series(list(values), dtype=pl.Int64)
    return e.cast(pl.Utf8).cast(id_dtype) if id_dtype in (pl.Utf8, pl.Categorical) else e.cast(id_dtype)


def _bind(nodes: pd.DataFrame, edges: pd.DataFrame, id_dtype):
    n_pl = pl.from_pandas(nodes).with_columns(_ids(nodes["id"], id_dtype).alias("id"))
    e_pl = pl.from_pandas(edges).with_columns(_ids(edges["s"], id_dtype).alias("s"), _ids(edges["d"], id_dtype).alias("d"))
    return graphistry.nodes(n_pl, "id").edges(e_pl, "s", "d")


def _pairs(pairs, id_dtype):
    return {tuple(_ids(p, id_dtype).to_list()) for p in pairs}


@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.Utf8, pl.Categorical], ids=["int", "str", "cat"])
def test_polars_hop_endpoint_gate_runs_on_the_installed_polars(direction, id_dtype):
    """Unseeded hop through ``_keep_edges_with_both_endpoints_resolvable``: only the closed edge
    survives, in every direction, for int / string / categorical ids (hand oracle).
    ``to_fixed_point`` keeps the hop OFF the single-bounded-hop lazy semi-join lane, so the
    eager ``is_in`` gate is the code that runs."""
    g = _bind(_NODES, _EDGES, id_dtype)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = g.hop(direction=direction, to_fixed_point=True, engine="polars")
    assert edge_pair_set(out) == _pairs(_CLOSED, id_dtype)
    assert _deprecations(caught) == []


@pytest.mark.parametrize("id_dtype", [pl.Int64, pl.Utf8], ids=["int", "str"])
def test_polars_seeded_multi_hop_gate_stops_at_a_dangling_endpoint(id_dtype):
    """Seeded 2-hop (hops=2 leaves the single-bounded-hop lane) from 2: the eager gate drops the
    dangling 3->4 edge before the BFS, so the second wave finds nothing. Without the gate the
    result would be {(2,3),(3,4)} -- the pin discriminates."""
    nodes = pd.DataFrame({"id": [0, 1, 2, 3]})
    edges = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4]})  # 3->4 dangles
    g = _bind(nodes, edges, id_dtype)
    seed = pl.DataFrame({"id": _ids([2], id_dtype)})
    out = g.hop(nodes=seed, hops=2, direction="forward", engine="polars")
    assert edge_pair_set(out) == _pairs({(2, 3)}, id_dtype)


def test_polars_hop_empty_node_table_over_edges_keeps_nothing():
    g = _bind(_NODES.iloc[0:0], _EDGES, pl.Int64)
    out = g.hop(direction="forward", to_fixed_point=True, engine="polars")
    assert out._edges.height == 0


@pytest.mark.parametrize("id_dtype", [pl.Utf8, pl.Categorical], ids=["str", "cat"])
def test_endpoint_gate_kernel_directly_on_string_and_categorical_ids(id_dtype):
    """``test_hop_kernel_contracts`` pins the kernel on Int64; #2082's failing corpus was
    categorical-heavy, so the same rule is pinned on string-typed ids here."""
    from graphistry.compute.gfql.lazy.engine.polars.hop_eager import (
        _keep_edges_with_both_endpoints_resolvable,
    )
    edges = pl.DataFrame({"s": _ids([0, 1, 7, 8], id_dtype), "d": _ids([1, 5, 2, 6], id_dtype)})
    for ids, expected in ((_ids([0, 1, 2], id_dtype), {(0, 1)}), (_ids([], id_dtype), set())):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kept = _keep_edges_with_both_endpoints_resolvable(edges, "s", "d", id_dtype, ids)
        assert set(zip(kept["s"].to_list(), kept["d"].to_list())) == _pairs(expected, id_dtype)
        assert _deprecations(caught) == []


# --- the routing itself, as a static lock ---------------------------------------------------
#
# The value-level pins above protect the helper, not the call sites. A future
# ``pl.col(x).is_in(ids.implode())`` written directly is correct and warning-free on every CI
# polars (>= 1.29) and raises ComputeError on the polars 1.21 that RAPIDS 25.02 pins, so no
# value test in any lane can catch it. This scan does.
#
# It follows a name bound by a plain, annotated or walrus assignment in the same module -- the
# spelling the reported site itself used (``universe = resolvable_ids.implode()`` then
# ``.is_in(universe)``) -- and walks the bound value, so an imploded set wrapped in another call
# (``.cast``, ``.alias``) still binds the name. That over-approximates, which fails loudly rather
# than silently. Binding through a tuple unpack, an attribute, or a helper's return value needs
# real dataflow and is NOT caught, nor is a bare-Series RHS: undecidable statically, and it only
# warns rather than failing.
#
# The lock lives in this polars-gated module, so it runs in the polars lane rather than every
# lane -- that lane fires on any python/gfql/core/infra change, which is where such a site
# could appear.

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
ROUTING_HELPER = REPO_ROOT / "graphistry" / "compute" / "gfql" / "lazy" / "engine" / "polars" / "membership.py"


def _implode_call(node: ast.AST) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == "implode")


def _imploded_is_in_linenos(tree: ast.AST) -> List[int]:
    """Lines in ``tree`` spelling ``<expr>.is_in(<imploded>)``, by argument or keyword, where
    ``<imploded>`` is either a direct ``....implode()`` or a name assigned one in the module."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AnnAssign, ast.NamedExpr)):  # `u: pl.Series = ...`, `(u := ...)`
            targets = [node.target]
        else:
            continue
        if node.value is not None and any(_implode_call(i) for i in ast.walk(node.value)):
            aliases |= {t.id for t in targets if isinstance(t, ast.Name)}
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "is_in"):
            continue
        for arg in [*node.args, *(kw.value for kw in node.keywords)]:
            if any(_implode_call(inner) or (isinstance(inner, ast.Name) and inner.id in aliases)
                   for inner in ast.walk(arg)):
                out.append(node.lineno)
                break
    return out


@functools.lru_cache(maxsize=1)
def _scan_for_imploded_is_in():
    """(offending ``path:line`` strings, paths actually parsed) over shipped graphistry code."""
    hits, scanned = [], []
    for path in sorted((REPO_ROOT / "graphistry").rglob("*.py")):
        # relative parts, so a checkout under a directory named "tests" cannot silently
        # disable the whole scan
        if "tests" in path.relative_to(REPO_ROOT).parts or path == ROUTING_HELPER:
            continue
        scanned.append(path)
        try:  # bytes, not read_text: source is UTF-8 by PEP 3120 whatever the locale is
            tree = ast.parse(path.read_bytes(), str(path))
        except SyntaxError as e:  # pragma: no cover - would mean the tree stopped parsing
            raise AssertionError(f"{path} does not parse under this python: {e}") from e
        hits += [f"{path.relative_to(REPO_ROOT)}:{line}" for line in _imploded_is_in_linenos(tree)]
    return tuple(hits), tuple(scanned)


def test_no_module_spells_an_imploded_is_in_outside_the_helper():
    hits, _ = _scan_for_imploded_is_in()
    assert hits == (), (
        "these sites spell is_in(...implode()) directly, which raises ComputeError on the "
        "polars 1.21 that RAPIDS 25.02 pins; route them through "
        f"graphistry.compute.gfql.lazy.engine.polars.membership.is_in_ids instead: {hits}"
    )


def test_the_routing_lock_scans_the_engine_and_exempts_a_file_that_exists():
    """A lock that silently scans nothing passes forever. Pin its inputs, as the lane-completeness
    and cache-registry locks pin theirs."""
    _, scanned = _scan_for_imploded_is_in()
    assert ROUTING_HELPER.is_file(), f"the exempt helper moved: {ROUTING_HELPER}"
    engine = REPO_ROOT / "graphistry" / "compute" / "gfql" / "lazy" / "engine" / "polars"
    # gfql_unified.py is routed too and lives OUTSIDE the engine subtree: without it, narrowing
    # the walk back to graphistry/compute/gfql would keep every other pin green.
    for must in (engine / "hop_eager.py", engine / "pattern_apply.py",
                 engine / "chain_specializations" / "hotpaths.py",
                 REPO_ROOT / "graphistry" / "compute" / "gfql_unified.py"):
        assert must in scanned, f"the routed module {must} was not scanned"
    assert ROUTING_HELPER in set((REPO_ROOT / "graphistry").rglob("*.py")), (
        "the helper must sit inside the walked tree, skipped by policy rather than by absence")
    assert ROUTING_HELPER not in scanned, "the exempt helper must be skipped, not parsed"
    # a floor far below the ~350 shipped modules, so ordinary tree growth never re-tunes it
    assert len(scanned) > 100, f"only {len(scanned)} files scanned; the walk is not reaching the tree"


@pytest.mark.parametrize("src,caught", [
    ("pl.col('a').is_in(ids.implode())", True),
    ("df.filter(pl.col('a').is_in(pl.lit(s).implode()))", True),
    ("col.is_in(\n    ids.implode(),\n)", True),
    ("col.is_in(other=ids.implode())", True),
    ("universe = ids.implode()\ncol.is_in(universe)", True),
    ("u: pl.Series = ids.implode()\ncol.is_in(u)", True),
    ("x = (u := ids.implode())\ncol.is_in(u)", True),
    ("pl.col('a').is_in(ids)", False),
    ("universe = id_set(ids)\ncol.is_in(universe)", False),
    ("u: pl.Series = id_set(ids)\ncol.is_in(u)", False),
    ("u = ids.implode().alias(a)\ncol.is_in(u)", True),
    ("u = ids.implode()\ncol.is_in([1])", False),
    ("x: int\nu = ids.implode()\ncol.is_in(u)", True),  # a valueless annotation must not abort the harvest
    ("pl.col('a').is_in([1, 2])", False),
    ("is_in_ids(pl.col('a'), ids)", False),
    ("df.with_columns(ids.implode()).filter(pl.col('a').is_in([1]))", False),
])
def test_the_routing_lock_detects_the_shape_it_guards(src, caught):
    """The matcher itself, against the spellings it must and must not flag."""
    assert bool(_imploded_is_in_linenos(ast.parse(src))) is caught
