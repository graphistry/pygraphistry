"""Boundary matrix for the two hop() contracts #1893 changes.

A. FILTER DOMAIN: ``source_node_match``/``destination_node_match`` are
   "kv-pairs to match nodes before/after hopping" -- they read the graph's NODE
   TABLE. Consequences pinned here: an id-only seed frame is a legal public
   shape at every hops value, and a seed frame carrying a same-named attribute
   column never overrides the node table.

B. UNDIRECTED FIXED-POINT WAVEFRONT: ``return_as_wave_front=True`` returns
   "only encountered nodes". Walking back along the edge you departed on is the
   trip home, not a discovery, so a SEED comes back only when some walk that
   REUSES NO EDGE reaches it -- which holds iff another seed shares its
   component (the shortest path between them is simple) or the seed lies on a
   cycle, intersected with what the traversal reached.

Every expected value below is hand-derived from those two rules on a graph
small enough to enumerate by eye, never read off another arm or the other
engine: two arms that agree can both be wrong.

Fixtures (node ``type`` is 'a' on even ids, 'b' on odd ids)::

    path5     0-1-2-3-4                     acyclic, diameter 4
    cycle3    0-1-2-0                       every node on a cycle
    selfloop  0-1, 1-1, 1-2                 cycle of length 1 at node 1
    parallel  0-1, 0-1, 1-2                 cycle of length 2 at nodes 0 and 1
    star      0-1, 0-2, 0-3                 acyclic, hub 0
    twocomp   0-1, 2-3                      two components
    isolated  1-2 (node 0 has no edges)     isolated node 0
"""
import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")
ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]

TOPOLOGIES = {
    "path5": ([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)]),
    "cycle3": ([0, 1, 2], [(0, 1), (1, 2), (2, 0)]),
    "selfloop": ([0, 1, 2], [(0, 1), (1, 1), (1, 2)]),
    "parallel": ([0, 1, 2], [(0, 1), (0, 1), (1, 2)]),
    "star": ([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)]),
    "twocomp": ([0, 1, 2, 3], [(0, 1), (2, 3)]),
    "isolated": ([0, 1, 2], [(1, 2)]),
}


def _graph(topology: str, engine: str):
    ids, edges = TOPOLOGIES[topology]
    ndf = pd.DataFrame({"id": ids, "type": ["a" if i % 2 == 0 else "b" for i in ids]})
    edf = pd.DataFrame({"s": [e[0] for e in edges], "d": [e[1] for e in edges]})
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(ndf), "id").edges(pl.from_pandas(edf), "s", "d")
    return graphistry.nodes(ndf, "id").edges(edf, "s", "d")


def _frame(engine: str, df: pd.DataFrame):
    return pl.from_pandas(df) if engine == "polars" else df


def node_ids(g):
    ndf = g._nodes
    ndf = ndf.to_pandas() if hasattr(ndf, "to_pandas") else ndf
    return sorted(ndf["id"].tolist())


def _hops_kwargs(hops):
    return {"to_fixed_point": True, "hops": 9} if hops == "tfp" else {"hops": hops}


# ======================================================================== A
# FILTER DOMAIN: the seed frame's own columns are never consulted.
#
# Three seed-frame SHAPES carry the same ids: id-only (the documented shape),
# id + the node table's true attr, and id + a STALE attr that contradicts the
# table. All three must produce the same answer at every hops value, in every
# direction, under every filter combination. Pre-#1892 the id-only shape raised
# at seeded hops==1 and the stale shape silently answered from the seed frame.

SEED_SETS = [[0], [2], [0, 2]]
DIRECTIONS = ["forward", "reverse", "undirected"]
FILTERS = [
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
    {"source_node_match": {"type": "a"}, "destination_node_match": {"type": "b"}},
]
HOPS_VALUES = [1, 2, "tfp"]


def _seed_frame_shapes(engine, seeds, topology):
    ids, _ = TOPOLOGIES[topology]
    true_types = {i: ("a" if i % 2 == 0 else "b") for i in ids}
    stale = {"a": "b", "b": "a"}
    return {
        "id_only": _frame(engine, pd.DataFrame({"id": seeds})),
        "true_attr": _frame(engine, pd.DataFrame(
            {"id": seeds, "type": [true_types[s] for s in seeds]})),
        "stale_attr": _frame(engine, pd.DataFrame(
            {"id": seeds, "type": [stale[true_types[s]] for s in seeds]})),
    }


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("topology", ["path5", "cycle3", "star"])
@pytest.mark.parametrize("seeds", SEED_SETS, ids=lambda s: "seeds" + "".join(map(str, s)))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("filt", FILTERS, ids=["src", "dst", "both"])
def test_filter_domain_ignores_seed_frame_columns(engine, topology, seeds, direction, filt):
    g = _graph(topology, engine)
    for hops in HOPS_VALUES:
        answers = {
            shape: node_ids(g.hop(nodes=frame, direction=direction, engine=engine,
                                  **_hops_kwargs(hops), **filt))
            for shape, frame in _seed_frame_shapes(engine, seeds, topology).items()
        }
        assert len(set(map(tuple, answers.values()))) == 1, (hops, answers)


# Hand-derived values on path5 (0a-1b-2a-3b-4a). A source filter gates the node
# a walk DEPARTS from; a destination filter gates the node it ARRIVES at; both
# read the node table. The answer is the subgraph of retained edges, so a cell
# that traverses nothing returns no nodes at all -- not the bare seed.
#
#   fwd  seeds[0] src a : 0 may depart -> edge (0,1). 1 is 'b' and may not
#                         depart, so hops 2/3/tfp add nothing.        [0, 1]
#   fwd  seeds[0] dst b : arrivals limited to {1, 3}; (0,1) qualifies, then
#                         (1,2) arrives at 'a' and is rejected.       [0, 1]
#   fwd  seeds[0] both  : the intersection of the two above.          [0, 1]
#   fwd  seeds[0,2] src a: 0 and 2 both depart -> (0,1) and (2,3); 1 and 3 are
#                         'b' and cannot continue.                 [0,1,2,3]
#   rev  seeds[4] src a : departing node is the seed 4 ('a') over (3,4); 3 is
#                         'b' and cannot depart.                        [3, 4]
#   rev  seeds[4] dst b : arrival is 3 ('b') over (3,4); next arrival would be
#                         2 ('a'), rejected.                            [3, 4]
#   undir seeds[2] src a: 2 departs both ways -> (1,2) and (2,3); 1 and 3 are
#                         'b'.                                       [1, 2, 3]
#   undir seeds[2] dst b: arrivals 1 and 3 are both 'b'; from them the only
#                         arrivals are 0, 2, 4, all 'a'.             [1, 2, 3]
#   undir seeds[0] src a: 0 departs over (0,1) only.                    [0, 1]
#   fwd  seeds[0] unfiltered: the control -- this one DOES grow with hops.

FILTER_DOMAIN_ORACLE = [
    ("forward", [0], {"source_node_match": {"type": "a"}}, [0, 1]),
    ("forward", [0], {"destination_node_match": {"type": "b"}}, [0, 1]),
    ("forward", [0], {"source_node_match": {"type": "a"},
                      "destination_node_match": {"type": "b"}}, [0, 1]),
    ("forward", [0, 2], {"source_node_match": {"type": "a"}}, [0, 1, 2, 3]),
    ("reverse", [4], {"source_node_match": {"type": "a"}}, [3, 4]),
    ("reverse", [4], {"destination_node_match": {"type": "b"}}, [3, 4]),
    ("undirected", [2], {"source_node_match": {"type": "a"}}, [1, 2, 3]),
    ("undirected", [2], {"destination_node_match": {"type": "b"}}, [1, 2, 3]),
    ("undirected", [0], {"source_node_match": {"type": "a"}}, [0, 1]),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", HOPS_VALUES + [3])
@pytest.mark.parametrize(
    "direction,seeds,filt,expected", FILTER_DOMAIN_ORACLE,
    ids=[f"{d}-{''.join(map(str, s))}-{'_'.join(sorted(f))}"
         for d, s, f, _ in FILTER_DOMAIN_ORACLE])
def test_filter_domain_hand_oracle(engine, hops, direction, seeds, filt, expected):
    g = _graph("path5", engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": seeds})), direction=direction,
              engine=engine, **_hops_kwargs(hops), **filt)
    assert node_ids(r) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops,expected", [
    (1, [0, 1]), (2, [0, 1, 2]), (3, [0, 1, 2, 3]), ("tfp", [0, 1, 2, 3, 4]),
])
def test_unfiltered_forward_grows_with_hops(engine, hops, expected):
    """Control: without a filter the answer DOES depend on hops, so the
    invariance asserted above is a property of the filter domain, not of a
    traversal that happens to saturate at hop 1."""
    g = _graph("path5", engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": [0]})), engine=engine,
              direction="forward", **_hops_kwargs(hops))
    assert node_ids(r) == expected


# NEGATIVE: a filter the node table cannot satisfy retains no edge, at every
# hops value -- and never falls back to reading the seed frame, which would
# make the 'zzz' cases answer from a seed column that does not exist there
# either, and would let seed 1 depart on its stale 'a'.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", HOPS_VALUES + [3])
@pytest.mark.parametrize("seeds,filt", [
    ([0], {"source_node_match": {"type": "zzz"}}),
    ([0], {"destination_node_match": {"type": "zzz"}}),
    ([1], {"source_node_match": {"type": "a"}}),          # seed 1 is 'b'
    ([1], {"destination_node_match": {"type": "zzz"}}),
], ids=["src-nomatch", "dst-nomatch", "seed-cannot-depart", "dst-nomatch-from-b"])
def test_filter_domain_negative_retains_nothing(engine, hops, seeds, filt):
    g = _graph("path5", engine)
    stale = pd.DataFrame({"id": seeds, "type": ["a" if i % 2 else "b" for i in seeds]})
    for frame in (pd.DataFrame({"id": seeds}), stale):
        r = g.hop(nodes=_frame(engine, frame), direction="forward", engine=engine,
                  **_hops_kwargs(hops), **filt)
        assert node_ids(r) == []


# Hop-window boundaries. hops=0 traverses no edge, so nothing is retained.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("filt", [{}] + FILTERS, ids=["none", "src", "dst", "both"])
def test_zero_hops_retains_nothing(engine, filt):
    g = _graph("path5", engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": [0]})), hops=0,
              direction="forward", engine=engine, **filt)
    assert node_ids(r) == []


def test_negative_hops_rejected_pandas():
    g = _graph("path5", "pandas")
    with pytest.raises(ValueError):
        g.hop(nodes=pd.DataFrame({"id": [0]}), hops=-1, engine="pandas")


@polars_only
def test_negative_hops_rejected_polars():
    g = _graph("path5", "polars")
    with pytest.raises(ValueError):
        g.hop(nodes=pl.from_pandas(pd.DataFrame({"id": [0]})), hops=-1, engine="polars")


def test_inverted_hop_window_rejected_pandas():
    g = _graph("path5", "pandas")
    with pytest.raises(ValueError):
        g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=2, max_hops=1, engine="pandas")


@pytest.mark.parametrize("engine", ENGINES)
def test_min_hops_window_filter_domain(engine):
    """min_hops>=2 leaves the seeded single-hop shortcut entirely, so the filter
    domain is unambiguously the node table: seed 0 departs, node 1 ('b') cannot,
    so the 2..3 window retains nothing."""
    g = _graph("path5", engine)
    if engine == "polars":
        pytest.importorskip("polars")
        with pytest.raises(NotImplementedError):
            g.hop(nodes=_frame(engine, pd.DataFrame({"id": [0]})), min_hops=2, max_hops=3,
                  source_node_match={"type": "a"}, engine=engine)
        return
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=2, max_hops=3,
              source_node_match={"type": "a"}, engine=engine)
    assert node_ids(r) == []


# ======================================================================== B
# UNDIRECTED FIXED-POINT WAVEFRONT: which seeds come back.
#
# Rule: a seed returns iff a walk that reuses no edge reaches it, i.e. another
# seed shares its component OR it lies on a cycle -- and it was reached.
# Hand derivation per cell:
#
#   path5    [0]    seed 0 alone on an acyclic path; the only way back is edge
#                   (0,1) twice.                                    [1,2,3,4]
#   path5    [2]    same, from the middle.                           [0,1,3,4]
#   path5    [0,4]  two seeds, one component: 4 reaches 0 along a simple path
#                   and vice versa.                                [0,1,2,3,4]
#   path5    [0,1]  two seeds, one component (adjacent).           [0,1,2,3,4]
#   cycle3   [0]    on a 3-cycle: 0-1-2-0 reuses no edge.               [0,1,2]
#   selfloop [1]    self-loop is a cycle of length 1.                   [0,1,2]
#   selfloop [0]    0 hangs off the cycle; no edge-disjoint return.        [1,2]
#   parallel [0]    two DISTINCT edges 0-1 form a cycle of length 2: out on
#                   one, back on the other.                             [0,1,2]
#   parallel [2]    2 hangs off that cycle by a single edge.              [0,1]
#   star     [0]    hub, acyclic, single seed.                          [1,2,3]
#   star     [1]    leaf, acyclic, single seed.                         [0,2,3]
#   star     [1,2]  two seeds sharing the component: 1-0-2 is simple. [0,1,2,3]
#   twocomp  [0,2]  two seeds in DIFFERENT components; neither has a partner
#                   or a cycle.                                            [1,3]
#   twocomp  [0]    single seed, acyclic.                                    [1]
#   isolated [0]    seed has no edges; nothing is reached at all.              []
#   isolated [0,1]  0 is isolated, so 1's component holds one seed; 0 is never
#                   reached either.                                          [2]

REDISCOVERY_ORACLE = [
    ("path5", [0], [1, 2, 3, 4]),
    ("path5", [2], [0, 1, 3, 4]),
    ("path5", [0, 4], [0, 1, 2, 3, 4]),
    ("path5", [0, 1], [0, 1, 2, 3, 4]),
    ("cycle3", [0], [0, 1, 2]),
    ("selfloop", [1], [0, 1, 2]),
    ("selfloop", [0], [1, 2]),
    ("parallel", [0], [0, 1, 2]),
    ("parallel", [2], [0, 1]),
    ("star", [0], [1, 2, 3]),
    ("star", [1], [0, 2, 3]),
    ("star", [1, 2], [0, 1, 2, 3]),
    ("twocomp", [0, 2], [1, 3]),
    ("twocomp", [0], [1]),
    ("isolated", [0], []),
    ("isolated", [0, 1], [2]),
]

# Both engines now agree with the hand oracle on every cell: pandas counts EDGE
# ROWS in the cycle helper (so parallel edges stay a length-2 cycle) and polars
# applies the same undirected-wavefront seed strip (#1918).
REDISCOVERY_XFAIL: set = set()

# to_fixed_point must equal the saturated bounded hop. The bounded arm applies the
# same edge-disjoint rediscovery rule as to_fixed_point (#1918), so no cell diverges.
TFP_EQUALS_BOUNDED_XFAIL: set = set()


def _rediscovery_params(xfail_set, reason):
    params = []
    for engine in ("pandas", "polars"):
        marks = [] if engine == "pandas" else [polars_only]
        for topology, seeds, expected in REDISCOVERY_ORACLE:
            cell_marks = list(marks)
            if (engine, topology, tuple(seeds)) in xfail_set:
                cell_marks.append(pytest.mark.xfail(
                    strict=True, raises=AssertionError, reason=reason))
            params.append(pytest.param(
                engine, topology, seeds, expected, marks=cell_marks,
                id=f"{engine}-{topology}-{''.join(map(str, seeds))}"))
    return params


@pytest.mark.parametrize(
    "engine,topology,seeds,expected",
    _rediscovery_params(
        REDISCOVERY_XFAIL,
        "#1918: pandas collapses parallel edges in the cycle helper; polars "
        "applies no undirected-wavefront seed strip"))
def test_undirected_tfp_wavefront_rediscovered_seeds(engine, topology, seeds, expected):
    g = _graph(topology, engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": seeds})), direction="undirected",
              return_as_wave_front=True, to_fixed_point=True, hops=9, engine=engine)
    assert node_ids(r) == expected


@pytest.mark.parametrize(
    "engine,topology,seeds,expected",
    _rediscovery_params(
        TFP_EQUALS_BOUNDED_XFAIL,
        "#1918: the bounded arm re-enters a seed over its own departure edge, "
        "so it keeps seeds to_fixed_point correctly drops"))
def test_undirected_tfp_equals_saturated_bounded(engine, topology, seeds, expected):
    g = _graph(topology, engine)
    kw = dict(nodes=_frame(engine, pd.DataFrame({"id": seeds})), direction="undirected",
              return_as_wave_front=True, hops=9, engine=engine)
    assert node_ids(g.hop(to_fixed_point=True, **kw)) == \
        node_ids(g.hop(to_fixed_point=False, **kw))


# The cells above are unfiltered, where topology alone decides. A filter can
# stop the traversal from ever reaching a seed the topology would rediscover --
# a seed on a cycle whose way home the filter forbids is NOT re-encountered.
# Hand derivation (undirected, wavefront; 'a' on even ids, 'b' on odd):
#
#   path5 [0,4] src a : 0 and 4 depart to 1 and 3; both are 'b' and stop there.
#                       Neither seed is reachable from the other.       [1, 3]
#   path5 [0,2] src a : 0 -> 1; 2 -> 1 and 3. Same dead end at 'b'.     [1, 3]
#   path5 [0,4] dst b : arrivals limited to 1 and 3; from them the only
#                       arrivals are 'a'.                               [1, 3]
#   cycle3 [0]  dst b : 1 is the only legal arrival; the way round the cycle
#                       back to 0 would have to arrive at 2 ('a'), so the seed
#                       stays out even though it sits on a cycle.          [1]
#   cycle3 [0,1] dst b: same traversal; seed 1 IS rediscovered, over the single
#                       edge (0,1), while seed 0 is not.                   [1]
#   star [1,2]  src b : only 1 may depart, reaching hub 0; 0 is 'a' and stops.
#                       Seed 2 is never reached at all.                    [0]
#   star [1,2]  dst b : both seeds may only arrive at 'b'; the hub is 'a', so
#                       no edge is retained.                                []
#   twocomp [0,2] src a: one seed per component, each stopping at 'b'.   [1, 3]
#   parallel [0,2] src a: 0 and 2 reach 1; 1 is 'b' so the parallel pair
#                       cannot carry 0 home.                               [1]
#   selfloop [1,0] src a: 1 is 'b' and cannot depart, so its self-loop is
#                       never traversed; 0 and 2 reach 1.                   [1]
#   path5 [0,1] dst a : only 1 may depart usefully, arriving at 0 and 2; seed 0
#                       is rediscovered over edge (0,1).                [0, 2]

FILTERED_REDISCOVERY_ORACLE = [
    ("path5", [0, 4], {"source_node_match": {"type": "a"}}, [1, 3]),
    ("path5", [0, 2], {"source_node_match": {"type": "a"}}, [1, 3]),
    ("path5", [0, 4], {"destination_node_match": {"type": "b"}}, [1, 3]),
    ("cycle3", [0], {"destination_node_match": {"type": "b"}}, [1]),
    ("cycle3", [0, 1], {"destination_node_match": {"type": "b"}}, [1]),
    ("star", [1, 2], {"source_node_match": {"type": "b"}}, [0]),
    ("star", [1, 2], {"destination_node_match": {"type": "b"}}, []),
    ("twocomp", [0, 2], {"source_node_match": {"type": "a"}}, [1, 3]),
    ("parallel", [0, 2], {"source_node_match": {"type": "a"}}, [1]),
    ("selfloop", [1, 0], {"source_node_match": {"type": "a"}}, [1]),
    ("path5", [0, 1], {"destination_node_match": {"type": "a"}}, [0, 2]),
]

FILTERED_IDS = [f"{t}-{''.join(map(str, s))}-{list(f)[0][:3]}"
                for t, s, f, _ in FILTERED_REDISCOVERY_ORACLE]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("topology,seeds,filt,expected", FILTERED_REDISCOVERY_ORACLE,
                         ids=FILTERED_IDS)
def test_undirected_tfp_wavefront_filtered_rediscovered_seeds(
        engine, topology, seeds, filt, expected):
    g = _graph(topology, engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": seeds})), direction="undirected",
              return_as_wave_front=True, to_fixed_point=True, hops=9, engine=engine, **filt)
    assert node_ids(r) == expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("topology,seeds,filt,expected", FILTERED_REDISCOVERY_ORACLE,
                         ids=FILTERED_IDS)
def test_undirected_tfp_equals_saturated_bounded_filtered(
        engine, topology, seeds, filt, expected):
    g = _graph(topology, engine)
    kw = dict(nodes=_frame(engine, pd.DataFrame({"id": seeds})), direction="undirected",
              return_as_wave_front=True, hops=9, engine=engine, **filt)
    assert node_ids(g.hop(to_fixed_point=True, **kw)) == \
        node_ids(g.hop(to_fixed_point=False, **kw)) == expected
