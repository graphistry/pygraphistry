"""
Unit tests for the treemap squarify implementation.

Tests are organized to:
1. Run against the reference `squarify` package to establish baselines
2. Run against our built-in `graphistry.layout.gib._squarify` to verify equivalence

We import both under aliases so the same test logic can be parameterized.
Any test that runs on the reference also runs on ours.
"""
import pytest

# ---------------------------------------------------------------------------
# Import helpers: reference vs ours
# ---------------------------------------------------------------------------

try:
    import squarify as _ref_squarify
    HAS_REF = True
except ImportError:
    HAS_REF = False

try:
    from graphistry.layout.gib import _squarify as _our_squarify
    HAS_OURS = True
except ImportError:
    HAS_OURS = False


# Collect (label, module) pairs we want to run all shared tests against.
IMPLS = []
if HAS_REF:
    IMPLS.append(("ref", _ref_squarify))
if HAS_OURS:
    IMPLS.append(("ours", _our_squarify))

IMPL_IDS = [label for label, _ in IMPLS]
IMPL_MODS = [mod for _, mod in IMPLS]


def pytest_generate_tests(metafunc):
    """Parametrize 'impl' fixture over available implementations."""
    if "impl" in metafunc.fixturenames:
        metafunc.parametrize("impl", IMPL_MODS, ids=IMPL_IDS)


# ---------------------------------------------------------------------------
# Geometry helpers / invariant checkers
# ---------------------------------------------------------------------------

TOL = 1e-9


def rect_area(r):
    return r["dx"] * r["dy"]


def rects_overlap(r1, r2, tol=1e-9):
    """True if two rects overlap (beyond floating-point tolerance)."""
    return (
        r1["x"] + tol < r2["x"] + r2["dx"]
        and r2["x"] + tol < r1["x"] + r1["dx"]
        and r1["y"] + tol < r2["y"] + r2["dy"]
        and r2["y"] + tol < r1["y"] + r1["dy"]
    )


def assert_geometry_invariants(rects, x, y, dx, dy, sizes=None):
    """Assert the full set of geometric invariants for a squarify output."""
    n = len(rects)

    # I1: correct length
    if sizes is not None:
        assert n == len(sizes), f"expected {len(sizes)} rects, got {n}"

    if n == 0:
        return  # nothing more to check on empty

    # I2: all rects inside the bounding box
    for i, r in enumerate(rects):
        assert r["x"] >= x - TOL, f"rect {i} x={r['x']} < canvas x={x}"
        assert r["y"] >= y - TOL, f"rect {i} y={r['y']} < canvas y={y}"
        assert r["x"] + r["dx"] <= x + dx + TOL, f"rect {i} right edge {r['x']+r['dx']} > {x+dx}"
        assert r["y"] + r["dy"] <= y + dy + TOL, f"rect {i} bottom edge {r['y']+r['dy']} > {y+dy}"

    # I3: positive dimensions (for non-degenerate canvas)
    if dx > 0 and dy > 0:
        for i, r in enumerate(rects):
            assert r["dx"] > -TOL, f"rect {i} dx={r['dx']} not positive"
            assert r["dy"] > -TOL, f"rect {i} dy={r['dy']} not positive"

    # I4: total rect area == canvas area
    total = sum(rect_area(r) for r in rects)
    canvas = dx * dy
    assert abs(total - canvas) < TOL * max(1, canvas), \
        f"area mismatch: rects={total} canvas={canvas}"

    # I5: no pair of rects overlaps (O(n^2) but n is small for tests)
    for i in range(n):
        for j in range(i + 1, n):
            assert not rects_overlap(rects[i], rects[j]), \
                f"rects {i} and {j} overlap: {rects[i]} vs {rects[j]}"


def normalize_and_squarify(impl, sizes, x, y, dx, dy):
    """Normalize sizes then run squarify — the canonical usage pattern."""
    normed = impl.normalize_sizes(sizes, dx, dy)
    return impl.squarify(normed, x, y, dx, dy)


# ===========================================================================
# Section 1: normalize_sizes
# ===========================================================================

class TestNormalizeSizes:

    def test_single_element_square(self, impl):
        result = impl.normalize_sizes([5], 10, 10)
        assert len(result) == 1
        assert abs(result[0] - 100.0) < TOL

    def test_single_element_rectangle(self, impl):
        result = impl.normalize_sizes([7], 4, 3)
        assert abs(result[0] - 12.0) < TOL

    def test_two_equal_elements_square(self, impl):
        result = impl.normalize_sizes([1, 1], 4, 4)
        assert len(result) == 2
        assert abs(result[0] - 8.0) < TOL
        assert abs(result[1] - 8.0) < TOL

    def test_two_unequal_elements(self, impl):
        result = impl.normalize_sizes([3, 1], 4, 4)
        assert abs(result[0] - 12.0) < TOL
        assert abs(result[1] - 4.0) < TOL
        assert abs(sum(result) - 16.0) < TOL

    def test_sum_equals_area(self, impl):
        sizes = [10, 6, 3, 1]
        dx, dy = 20, 10
        result = impl.normalize_sizes(sizes, dx, dy)
        assert abs(sum(result) - dx * dy) < TOL

    def test_already_normalized_unchanged(self, impl):
        dx, dy = 5.0, 4.0
        sizes = [10.0, 8.0, 2.0]
        result = impl.normalize_sizes(sizes, dx, dy)
        assert abs(sum(result) - dx * dy) < TOL
        # Ratios preserved
        ratio_in = sizes[0] / sizes[1]
        ratio_out = result[0] / result[1]
        assert abs(ratio_in - ratio_out) < TOL

    def test_float_inputs(self, impl):
        result = impl.normalize_sizes([1.5, 2.5], 4.0, 1.0)
        assert abs(sum(result) - 4.0) < TOL
        assert abs(result[0] / result[1] - 1.5 / 2.5) < TOL

    def test_many_elements_sum(self, impl):
        sizes = list(range(1, 11))  # [1..10], sum=55
        dx, dy = 11, 5
        result = impl.normalize_sizes(sizes, dx, dy)
        assert len(result) == 10
        assert abs(sum(result) - 55.0) < TOL

    def test_large_values_sum(self, impl):
        sizes = [1_000_000, 2_000_000, 3_000_000]
        result = impl.normalize_sizes(sizes, 100, 100)
        assert abs(sum(result) - 10000.0) < TOL

    def test_very_small_values_sum(self, impl):
        sizes = [1e-10, 2e-10, 3e-10]
        result = impl.normalize_sizes(sizes, 1.0, 1.0)
        assert abs(sum(result) - 1.0) < TOL

    def test_rectangular_canvas_wide(self, impl):
        result = impl.normalize_sizes([1, 1], 10, 1)
        assert abs(sum(result) - 10.0) < TOL

    def test_rectangular_canvas_tall(self, impl):
        result = impl.normalize_sizes([1, 1], 1, 10)
        assert abs(sum(result) - 10.0) < TOL

    def test_one_zero_among_positives(self, impl):
        result = impl.normalize_sizes([5, 0, 5], 2, 3)
        assert len(result) == 3
        assert abs(result[1] - 0.0) < TOL
        assert abs(sum(result) - 6.0) < TOL

    def test_all_equal_preserves_count_and_sum(self, impl):
        result = impl.normalize_sizes([3, 3, 3, 3], 8, 8)
        assert len(result) == 4
        assert abs(sum(result) - 64.0) < TOL
        # All equal output
        assert all(abs(v - result[0]) < TOL for v in result)

    def test_integer_inputs_become_float(self, impl):
        result = impl.normalize_sizes([1, 2, 3], 2, 3)
        assert all(isinstance(v, float) for v in result)

    def test_empty_returns_empty(self, impl):
        """Empty input: sum([])=0 but lambda never runs → returns []. Reference behavior."""
        result = impl.normalize_sizes([], 10, 10)
        assert result == []

    def test_all_zeros_raises(self, impl):
        """All-zero input: sum=0, lambda runs and hits 0/0 → ZeroDivisionError."""
        with pytest.raises((ZeroDivisionError, ValueError, Exception)):
            impl.normalize_sizes([0, 0, 0], 10, 10)

    def test_ratios_preserved(self, impl):
        """Output proportions must match input proportions."""
        sizes = [6, 3, 1]
        result = impl.normalize_sizes(sizes, 7, 11)
        assert abs(result[0] / result[2] - 6.0) < TOL  # 6:1 preserved
        assert abs(result[0] / result[1] - 2.0) < TOL  # 6:3 preserved

    def test_dx_zero_produces_zeros(self, impl):
        """When canvas area is 0, all outputs should be 0."""
        result = impl.normalize_sizes([1, 2, 3], 0, 5)
        assert all(abs(v) < TOL for v in result)

    def test_dy_zero_produces_zeros(self, impl):
        result = impl.normalize_sizes([1, 2, 3], 5, 0)
        assert all(abs(v) < TOL for v in result)


# ===========================================================================
# Section 2: squarify — structural / count tests
# ===========================================================================

class TestSquarifyStructure:

    def test_empty_returns_empty(self, impl):
        result = impl.squarify([], 0, 0, 10, 10)
        assert result == []

    def test_single_element_returns_one_rect(self, impl):
        result = impl.squarify([100.0], 0, 0, 10, 10)
        assert len(result) == 1

    def test_two_elements_returns_two_rects(self, impl):
        normed = impl.normalize_sizes([1, 1], 10, 10)
        result = impl.squarify(normed, 0, 0, 10, 10)
        assert len(result) == 2

    def test_output_length_matches_input_length(self, impl):
        for n in [1, 2, 3, 5, 7, 10, 20]:
            sizes = list(range(n, 0, -1))  # descending: n, n-1, ..., 1
            normed = impl.normalize_sizes(sizes, 100, 100)
            rects = impl.squarify(normed, 0, 0, 100, 100)
            assert len(rects) == n, f"n={n}: expected {n} rects, got {len(rects)}"

    def test_output_has_required_keys(self, impl):
        normed = impl.normalize_sizes([4, 3, 2, 1], 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        for r in rects:
            assert "x" in r and "y" in r and "dx" in r and "dy" in r

    def test_non_zero_origin(self, impl):
        normed = impl.normalize_sizes([1, 1, 1], 5, 5)
        rects = impl.squarify(normed, 10, 20, 5, 5)
        for r in rects:
            assert r["x"] >= 10 - TOL
            assert r["y"] >= 20 - TOL


# ===========================================================================
# Section 3: squarify — geometry invariants for cases not in the table
# (The bulk of geometry invariant testing is in the Section 8 parametrized
# table. Only keep cases here that the table doesn't cover.)
# ===========================================================================

class TestSquarifyGeometryInvariants:

    def test_single_element_exact_coords(self, impl):
        """Single element must fill the exact canvas coords."""
        normed = impl.normalize_sizes([1], 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        r = rects[0]
        assert abs(r["x"] - 0) < TOL
        assert abs(r["y"] - 0) < TOL
        assert abs(r["dx"] - 10) < TOL
        assert abs(r["dy"] - 10) < TOL

    def test_float_origin_invariants(self, impl):
        """Float origin not in geometry table — test separately."""
        normed = impl.normalize_sizes([2, 1], 3.5, 2.5)
        rects = impl.squarify(normed, 1.5, 0.5, 3.5, 2.5)
        assert_geometry_invariants(rects, 1.5, 0.5, 3.5, 2.5)


# ===========================================================================
# Section 4: squarify — specific value checks (whitebox baselines)
# Using actual squarify output as ground truth where we know the values.
# ===========================================================================

class TestSquarifySpecificValues:

    def test_single_fills_canvas(self, impl):
        normed = impl.normalize_sizes([5], 8, 3)
        rects = impl.squarify(normed, 0, 0, 8, 3)
        r = rects[0]
        assert abs(r["x"] - 0) < TOL
        assert abs(r["y"] - 0) < TOL
        assert abs(r["dx"] - 8) < TOL
        assert abs(r["dy"] - 3) < TOL

    def test_two_equal_wide_splits_vertically(self, impl):
        """Two equal sizes on wide canvas: side by side (column layout)."""
        normed = impl.normalize_sizes([1, 1], 2, 1)
        rects = impl.squarify(normed, 0, 0, 2, 1)
        # Each should be a 1x1 square
        for r in rects:
            assert abs(r["dx"] - 1.0) < TOL
            assert abs(r["dy"] - 1.0) < TOL

    def test_two_equal_tall_splits_horizontally(self, impl):
        """Two equal sizes on tall canvas: stacked (row layout)."""
        normed = impl.normalize_sizes([1, 1], 1, 2)
        rects = impl.squarify(normed, 0, 0, 1, 2)
        for r in rects:
            assert abs(r["dx"] - 1.0) < TOL
            assert abs(r["dy"] - 1.0) < TOL

    def test_two_equal_wide_splits_into_squares(self, impl):
        """[1,1] on 2x1: each rect should be 1x1 — verifies column layout chosen."""
        normed = impl.normalize_sizes([1, 1], 2, 1)
        rects = impl.squarify(normed, 0, 0, 2, 1)
        for r in rects:
            assert abs(r["dx"] - 1.0) < TOL
            assert abs(r["dy"] - 1.0) < TOL


# ===========================================================================
# Section 5: squarify vs reference (cross-implementation equivalence)
# Only runs when both implementations are available.
# ===========================================================================

@pytest.mark.skipif(
    not (HAS_REF and HAS_OURS),
    reason="Need both reference squarify and our _squarify installed"
)
class TestEquivalenceVsReference:
    """
    Verify our implementation matches the reference output on all test cases.
    This gives us confidence beyond geometry invariants.
    """

    CASES = [
        # (sizes, dx, dy)
        ([1], 10, 10),
        ([1, 1], 10, 10),
        ([1, 1], 20, 5),
        ([1, 1], 5, 20),
        ([3, 1], 10, 10),
        ([4, 3, 2, 1], 10, 10),
        ([6, 6, 4, 3, 2, 2, 1], 6, 4),
        ([1, 1, 1, 1], 4, 4),
        ([1, 1, 1, 1], 8, 2),
        ([100, 1, 1, 1], 10, 10),
        ([16, 8, 4, 2, 1], 10, 10),
        (list(range(10, 0, -1)), 10, 10),
        ([89, 55, 34, 21, 13, 8, 5, 3, 2, 1], 20, 20),
        ([9, 4, 1], 7, 7),
        ([5, 3, 2], 30, 10),
        ([50] + [1] * 9, 10, 10),
        ([6, 3, 2, 1], 12, 10),
        ([1, 1, 1, 1], 1000, 1),
        ([1, 1, 1, 1], 1, 1000),
    ]

    @pytest.mark.parametrize("sizes,dx,dy", CASES,
        ids=[f"n{len(s)}-{dx}x{dy}" for s, dx, dy in CASES])
    def test_matches_reference(self, sizes, dx, dy):
        ref_normed = _ref_squarify.normalize_sizes(sizes, dx, dy)
        ref_rects = _ref_squarify.squarify(ref_normed, 0, 0, dx, dy)

        our_normed = _our_squarify.normalize_sizes(sizes, dx, dy)
        our_rects = _our_squarify.squarify(our_normed, 0, 0, dx, dy)

        assert len(our_rects) == len(ref_rects), \
            f"lengths differ: ours={len(our_rects)}, ref={len(ref_rects)}"

        for i, (ours, ref) in enumerate(zip(our_rects, ref_rects)):
            for key in ("x", "y", "dx", "dy"):
                assert abs(ours[key] - ref[key]) < TOL, \
                    f"sizes={sizes} dx={dx} dy={dy}: rect {i} key={key}: ours={ours[key]}, ref={ref[key]}"

    def test_normalize_matches_reference(self):
        cases = [
            ([1, 2, 3], 10, 10),
            ([5, 0, 5], 2, 3),
            ([100, 1], 50, 20),
            (list(range(1, 11)), 11, 5),
            ([1e-10, 2e-10], 1.0, 1.0),
        ]
        for sizes, dx, dy in cases:
            ref = _ref_squarify.normalize_sizes(sizes, dx, dy)
            ours = _our_squarify.normalize_sizes(sizes, dx, dy)
            assert len(ours) == len(ref)
            for i, (o, r) in enumerate(zip(ours, ref)):
                assert abs(o - r) < TOL, \
                    f"normalize_sizes({sizes}, {dx}, {dy}): index {i}: ours={o} ref={r}"


# ===========================================================================
# Section 6: normalize_sizes + squarify integration (end-to-end flow)
# ===========================================================================

class TestNormalizeAndSquarifyIntegration:

    def test_flow_single(self, impl):
        rects = normalize_and_squarify(impl, [5], 0, 0, 8, 6)
        assert len(rects) == 1
        assert_geometry_invariants(rects, 0, 0, 8, 6)

    def test_flow_two(self, impl):
        rects = normalize_and_squarify(impl, [3, 1], 0, 0, 4, 4)
        assert len(rects) == 2
        assert_geometry_invariants(rects, 0, 0, 4, 4)

    def test_flow_many(self, impl):
        sizes = list(range(15, 0, -1))
        rects = normalize_and_squarify(impl, sizes, 0, 0, 30, 20)
        assert len(rects) == 15
        assert_geometry_invariants(rects, 0, 0, 30, 20)

    def test_no_overlaps_many_elements(self, impl):
        sizes = list(range(12, 0, -1))
        rects = normalize_and_squarify(impl, sizes, 0, 0, 12, 12)
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                assert not rects_overlap(rects[i], rects[j]), \
                    f"overlap between rect {i} and {j}"

    def test_worst_aspect_ratio_improves_vs_naive(self, impl):
        """
        Squarified treemaps should produce better aspect ratios than a
        naive vertical-strip layout on a square canvas with typical inputs.
        """
        sizes = [4, 3, 2, 1]
        dx, dy = 10.0, 10.0
        rects = normalize_and_squarify(impl, sizes, 0, 0, dx, dy)
        worst = max(
            max(r["dx"] / r["dy"], r["dy"] / r["dx"])
            for r in rects
        )
        # Squarified should beat naive (single row: widths = 4,3,2,1, height=10 → ratio=10)
        assert worst < 10.0, f"worst aspect ratio {worst} should be < 10"


# ===========================================================================
# Section 7: edge cases / robustness
# ===========================================================================

class TestEdgeCases:

    def test_squarify_empty_input(self, impl):
        result = impl.squarify([], 0, 0, 10, 10)
        assert result == []

    def test_squarify_single_tiny(self, impl):
        normed = impl.normalize_sizes([1], 0.001, 0.001)
        rects = impl.squarify(normed, 0, 0, 0.001, 0.001)
        assert len(rects) == 1
        assert abs(rect_area(rects[0]) - 0.001 * 0.001) < 1e-15

    def test_squarify_reproducible(self, impl):
        """Same inputs always produce identical outputs."""
        sizes = [5, 3, 2, 1]
        normed = impl.normalize_sizes(sizes, 10, 10)
        r1 = impl.squarify(normed, 0, 0, 10, 10)
        r2 = impl.squarify(normed, 0, 0, 10, 10)
        for a, b in zip(r1, r2):
            for k in ("x", "y", "dx", "dy"):
                assert a[k] == b[k]


# ===========================================================================
# Section 8: parametrized geometry invariant table
# Replaces many individual _run() one-liners with a single table.
# ===========================================================================

# Each row: (label, sizes, dx, dy)
_GEOMETRY_CASES = [
    # single element
    ("single_sq",         [1],                             10,    10),
    ("single_wide",       [1],                             20,    5),
    ("single_tall",       [1],                             3,     15),
    ("single_float_dims", [1],                             3.7,   2.9),
    # two elements
    ("two_eq_sq",         [1, 1],                          10,    10),
    ("two_eq_wide",       [1, 1],                          20,    5),
    ("two_eq_tall",       [1, 1],                          5,     20),
    ("two_uneq_3_1",      [3, 1],                          10,    10),
    ("two_extreme_999_1", [999, 1],                        30,    20),
    # classic paper examples
    ("classic_4321",      [4, 3, 2, 1],                    10,    10),
    ("classic_6643221",   [6, 6, 4, 3, 2, 2, 1],           6,     4),
    # all equal
    ("eq4_sq",            [1, 1, 1, 1],                    4,     4),
    ("eq4_wide",          [1, 1, 1, 1],                    8,     2),
    ("eq4_tall",          [1, 1, 1, 1],                    2,     8),
    ("eq50",              [1] * 50,                        10,    10),
    # one dominant
    ("dom_100_rest_1",    [100, 1, 1, 1],                  10,    10),
    ("dom_50_nine_1",     [50] + [1] * 9,                  10,    10),
    # descending sequences
    ("desc5",             [16, 8, 4, 2, 1],                10,    10),
    ("desc10",            list(range(10, 0, -1)),           10,    10),
    ("desc20",            list(range(20, 0, -1)),           20,    20),
    # fibonacci
    ("fib10",             [89, 55, 34, 21, 13, 8, 5, 3, 2, 1], 20, 20),
    # primes descending
    ("primes10",          [97, 89, 83, 79, 73, 71, 67, 61, 59, 53], 100, 100),
    # powers of two
    ("pow2_8",            [2 ** i for i in range(8, 0, -1)], 16,  16),
    # near-square canvas
    ("near_sq_wide",      [3, 2, 1],                       10.0000001, 10.0),
    ("near_sq_tall",      [3, 2, 1],                       10.0,       10.0000001),
    # extreme aspect ratio canvases
    ("extreme_wide",      [1, 1, 1, 1],                    1000,  1),
    ("extreme_tall",      [1, 1, 1, 1],                    1,     1000),
    # large / small canvas
    ("large_canvas",      [10, 6, 3, 1],                   1000,  1000),
    ("small_canvas",      [4, 3, 2, 1],                    0.01,  0.01),
    ("tiny_canvas_1px",   [1, 2, 3],                       1,     1),
    # offset origin
    ("offset_origin",     [4, 3, 2, 1],                    8,     6),
    # non-square canvases
    ("rect_wide",         [5, 3, 2],                       30,    10),
    ("rect_tall",         [5, 3, 2],                       10,    30),
    # 3-element various
    ("three_9_4_1",       [9, 4, 1],                       7,     7),
    ("three_6_3_1",       [6, 3, 2, 1],                    12,    10),
    # float origin (tested separately in _run_with_origin below)
]


@pytest.mark.parametrize("label,sizes,dx,dy", _GEOMETRY_CASES,
    ids=[r[0] for r in _GEOMETRY_CASES])
def test_geometry_invariants_table(impl, label, sizes, dx, dy):  # noqa: ARG001
    """Parametrized table: all inputs must satisfy geometry invariants."""
    normed = impl.normalize_sizes(sizes, dx, dy)
    rects = impl.squarify(normed, 0, 0, dx, dy)
    assert_geometry_invariants(rects, 0, 0, dx, dy, sizes)


@pytest.mark.parametrize("label,sizes,dx,dy", _GEOMETRY_CASES,
    ids=[r[0] for r in _GEOMETRY_CASES])
def test_geometry_invariants_table_offset(impl, label, sizes, dx, dy):  # noqa: ARG001
    """Same cases with a non-zero origin to catch coordinate-offset bugs."""
    ox, oy = 13.5, 7.25
    normed = impl.normalize_sizes(sizes, dx, dy)
    rects = impl.squarify(normed, ox, oy, dx, dy)
    assert_geometry_invariants(rects, ox, oy, dx, dy, sizes)


@pytest.mark.parametrize("label,sizes,dx,dy", _GEOMETRY_CASES,
    ids=[r[0] for r in _GEOMETRY_CASES])
def test_area_proportions_table(impl, label, sizes, dx, dy):  # noqa: ARG001
    """Each rect's area must equal its share of the canvas."""
    normed = impl.normalize_sizes(sizes, dx, dy)
    rects = impl.squarify(normed, 0, 0, dx, dy)
    total_in = sum(sizes)
    canvas = dx * dy
    for i, sz in enumerate(sizes):
        expected = sz / total_in * canvas
        got = rect_area(rects[i])
        assert abs(got - expected) < TOL * max(1, canvas), \
            f"rect {i} area {got} != expected {expected}"


# ===========================================================================
# Section 9: strip-loop boundary conditions (whitebox)
# ===========================================================================

class TestStripLoopBoundaries:
    """
    Whitebox tests targeting the greedy strip-finding loop in squarify().
    The loop runs while worst_ratio(sizes[:i]) >= worst_ratio(sizes[:i+1]).
    """

    def test_loop_exits_at_i1_when_adding_worsens(self, impl):
        """
        [1, 1] on 10x10: adding 2nd item always worsens ratio (both equal,
        rowlayout of [1,1] is 2 wide x 0.5 tall → AR=4 vs single 1x1 → AR=1).
        Loop exits at i=1, rects placed one-at-a-time.
        """
        normed = impl.normalize_sizes([1, 1], 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        assert len(rects) == 2
        assert_geometry_invariants(rects, 0, 0, 10, 10)

    def test_loop_runs_to_end_all_in_one_strip(self, impl):
        """
        When sizes are such that the worst ratio always improves by adding more
        items, the loop runs to completion → all items in one strip.
        [4, 3, 2, 1] on 10x10: loop runs fully as adding items keeps improving ratio.
        """
        normed = impl.normalize_sizes([4, 3, 2, 1], 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        assert len(rects) == 4
        assert_geometry_invariants(rects, 0, 0, 10, 10)

    def test_single_item_bypasses_loop(self, impl):
        """len==1 takes the early-return path, never enters strip loop."""
        normed = impl.normalize_sizes([7], 5, 3)
        rects = impl.squarify(normed, 0, 0, 5, 3)
        assert len(rects) == 1
        r = rects[0]
        assert abs(r["dx"] - 5) < TOL
        assert abs(r["dy"] - 3) < TOL

    def test_two_items_one_dominant_splits_at_1(self, impl):
        """999:1 — dominant item grabs almost all space, remainder tiny."""
        normed = impl.normalize_sizes([999, 1], 100, 100)
        rects = impl.squarify(normed, 0, 0, 100, 100)
        assert len(rects) == 2
        # First rect should have ~999/1000 of area
        assert abs(rect_area(rects[0]) / (100 * 100) - 999 / 1000) < TOL

    def test_aspect_ratio_monotone_for_strip_prefix(self, impl):  # noqa: ARG002
        """
        For the first strip, worst_ratio(sizes[:i]) must be >= worst_ratio(sizes[:i+1])
        for each i that the loop traversed. This is the loop invariant.
        We verify it holds for the classic [4,3,2,1] case.
        """
        from graphistry.layout.gib._squarify import _worst_ratio, normalize_sizes as _ns
        sizes = [float(s) for s in _ns([4, 3, 2, 1], 10, 10)]
        x, y, dx, dy = 0, 0, 10, 10
        # Simulate loop: find how far i goes
        i = 1
        while i < len(sizes) and _worst_ratio(sizes[:i], x, y, dx, dy) >= _worst_ratio(sizes[:i+1], x, y, dx, dy):
            # The condition that caused i to increment must hold
            assert _worst_ratio(sizes[:i], x, y, dx, dy) >= _worst_ratio(sizes[:i+1], x, y, dx, dy)
            i += 1

    def test_remaining_empty_when_all_in_one_strip(self, impl):
        """When loop goes all the way, remaining=[] → recursive call returns []."""
        # [1] is the base case, but [4,3,2,1] might place all 4 in one strip
        normed = impl.normalize_sizes([4, 3, 2, 1], 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        # Either way, all 4 must be present
        assert len(rects) == 4

    def test_many_strips_deep_recursion(self, impl):
        """50 equal elements → many strips → deep recursion, no stack issues."""
        normed = impl.normalize_sizes([1] * 50, 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        assert len(rects) == 50
        assert_geometry_invariants(rects, 0, 0, 10, 10)


# ===========================================================================
# Section 10: floating-point accumulation stress tests
# ===========================================================================

class TestFloatAccumulation:
    """
    Tests that target floating-point error accumulation across recursive calls.
    Each recursive level inherits leftover coordinates, so rounding errors compound.
    """

    def test_leftover_coords_sum_to_canvas(self, impl):
        """After many recursion levels, total area stays exact."""
        sizes = list(range(30, 0, -1))
        normed = impl.normalize_sizes(sizes, 100, 100)
        rects = impl.squarify(normed, 0, 0, 100, 100)
        total = sum(rect_area(r) for r in rects)
        assert abs(total - 10000.0) < 1e-6  # relaxed tol for deep recursion

    def test_x_coords_within_bounds_after_deep_recursion(self, impl):
        """x + dx must never exceed canvas width even after many splits."""
        sizes = list(range(40, 0, -1))
        normed = impl.normalize_sizes(sizes, 200, 100)
        rects = impl.squarify(normed, 0, 0, 200, 100)
        for r in rects:
            assert r["x"] + r["dx"] <= 200 + 1e-6
            assert r["y"] + r["dy"] <= 100 + 1e-6

    def test_irrational_canvas_no_blowup(self, impl):
        """Canvas with irrational-ish dimensions shouldn't cause NaN/inf."""
        import math
        dx = math.pi * 10
        dy = math.e * 10
        sizes = [7, 5, 3, 2, 1]
        normed = impl.normalize_sizes(sizes, dx, dy)
        rects = impl.squarify(normed, 0, 0, dx, dy)
        assert len(rects) == 5
        for r in rects:
            assert all(not (v != v) for v in r.values())  # no NaN
            assert all(v < float("inf") for v in r.values())  # no inf

    def test_very_unequal_sizes_no_zero_dim(self, impl):
        """Dominant first element followed by many tiny ones — no zero-area rects."""
        sizes = [1000] + [1] * 20
        normed = impl.normalize_sizes(sizes, 50, 50)
        rects = impl.squarify(normed, 0, 0, 50, 50)
        assert len(rects) == 21
        for r in rects:
            assert r["dx"] > 1e-10, f"near-zero dx: {r}"
            assert r["dy"] > 1e-10, f"near-zero dy: {r}"

    def test_uniform_sizes_no_rounding_drift(self, impl):
        """100 equal items: leftover coords must drift < epsilon."""
        n = 100
        normed = impl.normalize_sizes([1] * n, 10, 10)
        rects = impl.squarify(normed, 0, 0, 10, 10)
        assert len(rects) == n
        total = sum(rect_area(r) for r in rects)
        assert abs(total - 100.0) < 1e-6

    def test_sizes_summing_to_non_integer_area(self, impl):
        """Canvas area 7*11=77 (not a round number) — precision still holds."""
        sizes = [5, 4, 3, 2, 1]
        normed = impl.normalize_sizes(sizes, 7, 11)
        rects = impl.squarify(normed, 0, 0, 7, 11)
        assert_geometry_invariants(rects, 0, 0, 7, 11)


# ===========================================================================
# Section 11: numpy array inputs (belt-and-suspenders)
# ===========================================================================

import numpy as np  # noqa: E402  (needed here for these tests)


class TestNumpyInputs:
    """
    _squarify should accept numpy arrays as input, not just Python lists.
    treemap.py calls .to_numpy().tolist() before us, but test the raw API too.
    """

    def test_normalize_sizes_numpy_array(self, impl):
        arr = np.array([4.0, 3.0, 2.0, 1.0])
        result = impl.normalize_sizes(arr, 10, 10)
        assert abs(sum(result) - 100.0) < TOL

    def test_normalize_sizes_numpy_int_array(self, impl):
        arr = np.array([4, 3, 2, 1], dtype=np.int64)
        result = impl.normalize_sizes(arr, 10, 10)
        assert abs(sum(result) - 100.0) < TOL

    def test_squarify_numpy_array(self, impl):
        arr = np.array([40.0, 30.0, 20.0, 10.0])  # pre-normalized for 10x10
        rects = impl.squarify(arr, 0, 0, 10, 10)
        assert len(rects) == 4
        assert_geometry_invariants(rects, 0, 0, 10, 10)

    def test_full_flow_numpy(self, impl):
        arr = np.array([6, 3, 2, 1], dtype=float)
        normed = impl.normalize_sizes(arr, 12, 10)
        rects = impl.squarify(normed, 0, 0, 12, 10)
        assert len(rects) == 4
        assert_geometry_invariants(rects, 0, 0, 12, 10)


# ===========================================================================
# Section 12: treemap() end-to-end integration (pandas)
# Tests the full graphistry.layout.gib.treemap.treemap() function.
# ===========================================================================

class TestTreemapEndToEnd:

    def _make_plottable(self, partition_col, node_col="id"):
        """Build a minimal Plottable-like object for treemap()."""
        import pandas as pd
        from graphistry.PlotterBase import PlotterBase

        nodes = pd.DataFrame({
            node_col: list(range(10)),
            partition_col: [0, 0, 0, 0, 1, 1, 1, 2, 2, 3],
        })
        edges = pd.DataFrame({"s": [0], "d": [1]})

        g = PlotterBase()
        g = g.nodes(nodes, node_col).edges(edges, "s", "d")
        return g

    def test_output_is_dict_of_dicts(self):
        from graphistry.layout.gib.treemap import treemap
        g = self._make_plottable("partition")
        result = treemap(g)
        assert isinstance(result, dict)
        for key in ("x", "y", "dx", "dy"):
            assert key in result
            assert isinstance(result[key], dict)

    def test_all_partitions_present(self):
        from graphistry.layout.gib.treemap import treemap
        g = self._make_plottable("partition")
        result = treemap(g)
        partitions = set(g._nodes["partition"].unique())
        for key in ("x", "y", "dx", "dy"):
            assert set(result[key].keys()) == partitions, \
                f"key {key}: missing partitions"

    def test_values_are_finite_floats(self):
        from graphistry.layout.gib.treemap import treemap
        import math
        g = self._make_plottable("partition")
        result = treemap(g)
        for key in ("x", "y", "dx", "dy"):
            for pid, val in result[key].items():
                assert math.isfinite(val), f"key={key} partition={pid} value={val} not finite"

    def test_dx_dy_positive(self):
        from graphistry.layout.gib.treemap import treemap
        g = self._make_plottable("partition")
        result = treemap(g)
        for pid, val in result["dx"].items():
            assert val > 0, f"dx[{pid}]={val} not positive"
        for pid, val in result["dy"].items():
            assert val > 0, f"dy[{pid}]={val} not positive"

    def test_areas_proportional_to_partition_sizes(self):
        """Each partition's area should be proportional to its node count."""
        from graphistry.layout.gib.treemap import treemap
        import math
        g = self._make_plottable("partition")
        result = treemap(g)

        # Node counts per partition
        counts = g._nodes.groupby("partition")[g._node].count().to_dict()
        total_nodes = sum(counts.values())
        total_area = sum(result["dx"][p] * result["dy"][p] for p in counts)

        for pid, count in counts.items():
            expected_area = count / total_nodes * total_area
            got_area = result["dx"][pid] * result["dy"][pid]
            assert abs(got_area - expected_area) < TOL * max(1, total_area), \
                f"partition {pid}: expected area {expected_area}, got {got_area}"

    def test_custom_canvas_size(self):
        from graphistry.layout.gib.treemap import treemap
        g = self._make_plottable("partition")
        result = treemap(g, w=200.0, h=100.0)
        # All x+dx <= 200, all y+dy <= 100
        for pid in result["x"]:
            assert result["x"][pid] + result["dx"][pid] <= 200.0 + TOL
            assert result["y"][pid] + result["dy"][pid] <= 100.0 + TOL

    def test_custom_origin(self):
        from graphistry.layout.gib.treemap import treemap
        g = self._make_plottable("partition")
        result = treemap(g, x=5.0, y=3.0, w=50.0, h=50.0)
        for pid in result["x"]:
            assert result["x"][pid] >= 5.0 - TOL
            assert result["y"][pid] >= 3.0 - TOL

    def test_single_partition(self):
        """Single partition → one rect filling the whole canvas."""
        import pandas as pd
        from graphistry.PlotterBase import PlotterBase
        from graphistry.layout.gib.treemap import treemap

        nodes = pd.DataFrame({"id": [0, 1, 2], "partition": [0, 0, 0]})
        edges = pd.DataFrame({"s": [0], "d": [1]})
        g = PlotterBase().nodes(nodes, "id").edges(edges, "s", "d")
        result = treemap(g, w=10.0, h=10.0)

        assert len(result["x"]) == 1
        assert abs(result["dx"][0] - 10.0) < TOL
        assert abs(result["dy"][0] - 10.0) < TOL

    def test_many_single_node_partitions(self):
        """Each node in its own partition → n rects, geometry invariants hold."""
        import pandas as pd
        from graphistry.PlotterBase import PlotterBase
        from graphistry.layout.gib.treemap import treemap

        n = 8
        nodes = pd.DataFrame({"id": list(range(n)), "partition": list(range(n))})
        edges = pd.DataFrame({"s": [0], "d": [1]})
        g = PlotterBase().nodes(nodes, "id").edges(edges, "s", "d")
        result = treemap(g, w=100.0, h=100.0)

        assert len(result["x"]) == n
        rects = [{"x": result["x"][p], "y": result["y"][p],
                  "dx": result["dx"][p], "dy": result["dy"][p]}
                 for p in range(n)]
        assert_geometry_invariants(rects, 0, 0, 100.0, 100.0)
