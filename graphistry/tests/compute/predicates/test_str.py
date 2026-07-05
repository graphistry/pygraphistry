import pandas as pd
import pytest
import re

from graphistry.compute.predicates.str import (
    contains,
    startswith,
    endswith,
    match,
    fullmatch,
    IsUpper, isupper
)
# Helper to check if cuDF is available and functional (requires GPU)
def has_cudf():
    try:
        import cudf
        # Test actual GPU operation - import alone doesn't guarantee GPU works
        _ = cudf.Series([1, 2, 3])
        return True
    except (ImportError, Exception):
        # ImportError if cudf not installed
        # Other exceptions (CUDARuntimeError) if GPU not available
        return False


# Cache result to avoid repeated GPU checks
_cudf_available = None


def cudf_available():
    global _cudf_available
    if _cudf_available is None:
        _cudf_available = has_cudf()
    return _cudf_available


# Skip tests that require cuDF when it's not available or GPU not working
requires_cudf = pytest.mark.skipif(
    not cudf_available(),
    reason="cudf not installed or GPU not available"
)


def test_is_upper():
    d = isupper()
    assert isinstance(d, IsUpper)

    o = d.to_json()
    assert isinstance(o, dict)
    assert o['type'] == 'IsUpper'

    d2 = IsUpper.from_json(o)
    assert isinstance(d2, IsUpper)



def test_contains_pandas_basic():
    s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('og')
    result = predicate(s)
    expected = pd.Series([False, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_regex():
    s = pd.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('house|dog', regex=True)
    result = predicate(s)
    expected = pd.Series([False, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_case_insensitive():
    s = pd.Series(['Mouse', 'dog', 'HOUSE', 'house'])
    predicate = contains('house', case=False)
    result = predicate(s)
    expected = pd.Series([False, False, True, True])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_na_default():
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og')
    result = predicate(s)
    assert result[0] is False
    assert result[1] is True
    assert pd.isna(result[2])
    assert result[3] is False


def test_contains_pandas_na_false():
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og', na=False)
    result = predicate(s)
    expected = pd.Series([False, True, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_contains_pandas_na_true():
    s = pd.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og', na=True)
    result = predicate(s)
    expected = pd.Series([False, True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_contains_cudf_basic():
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house and parrot', '23'])
    predicate = contains('og')
    result = predicate(s)
    expected = cudf.Series([False, True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_cudf_case_insensitive():
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'HOUSE', 'house'])
    predicate = contains('house', case=False)
    result = predicate(s)
    expected = cudf.Series([False, False, True, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_cudf_na_handling():
    import cudf

    # Test default NA behavior
    s = cudf.Series(['Mouse', 'dog', None, 'house'])
    predicate = contains('og')
    result = predicate(s).to_pandas()
    assert result[0] is False
    assert result[1] is True
    assert pd.isna(result[2])
    assert result[3] is False

    # Test NA=False
    predicate = contains('og', na=False)
    result = predicate(s)
    expected = cudf.Series([False, True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())

    # Test NA=True
    predicate = contains('og', na=True)
    result = predicate(s)
    expected = cudf.Series([False, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_contains_pandas_cudf_parity():
    import cudf

    # Create identical data
    data = ['Mouse', 'dog', None, 'HOUSE', 'house and parrot']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)

    # Test case-sensitive
    predicate = contains('house')
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test case-insensitive
    predicate = contains('house', case=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test with na=False
    predicate = contains('house', na=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)



BACKENDS = [
    "pandas",
    pytest.param("cudf", marks=requires_cudf),
]

BOUNDARY_PREDICATES = [
    (
        "startswith",
        startswith,
        "ho",
        ['Mouse', 'dog', 'house', 'Home'],
        [False, False, True, False],
    ),
    (
        "endswith",
        endswith,
        "se",
        ['Mouse', 'dog', 'house', 'Home'],
        [True, False, True, False],
    ),
]


def _series_for_backend(backend, data):
    if backend == "cudf":
        import cudf
        return cudf.Series(data)
    return pd.Series(data)


def _to_pandas(result):
    return result.to_pandas() if hasattr(result, "to_pandas") else result


def _assert_result_values(result, expected):
    result = _to_pandas(result)
    assert len(result) == len(expected)
    for actual, expected_value in zip(result, expected):
        if expected_value is None:
            assert pd.isna(actual)
        else:
            assert bool(actual) is expected_value


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,predicate_factory,pat,data,expected",
    BOUNDARY_PREDICATES,
)
def test_boundary_string_basic(
    backend,
    name,
    predicate_factory,
    pat,
    data,
    expected,
):
    s = _series_for_backend(backend, data)
    result = predicate_factory(pat)(s)
    _assert_result_values(result, expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,predicate_factory,pat,data,expected",
    [
        (
            "startswith",
            startswith,
            "john",
            ['John', 'john', 'JOHN', 'Jane'],
            [True, True, True, False],
        ),
        (
            "endswith",
            endswith,
            ".com",
            ['test.com', 'test.COM', 'test.Com', 'test.org'],
            [True, True, True, False],
        ),
    ],
)
def test_boundary_string_case_insensitive(
    backend,
    name,
    predicate_factory,
    pat,
    data,
    expected,
):
    s = _series_for_backend(backend, data)
    result = predicate_factory(pat, case=False)(s)
    _assert_result_values(result, expected)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,predicate_factory,pat,expected_default,expected_false,expected_true",
    [
        (
            "startswith",
            startswith,
            "ho",
            [False, None, True],
            [False, False, True],
            [False, True, True],
        ),
        (
            "endswith",
            endswith,
            "se",
            [True, None, True],
            [True, False, True],
            [True, True, True],
        ),
    ],
)
def test_boundary_string_na_handling(
    backend,
    name,
    predicate_factory,
    pat,
    expected_default,
    expected_false,
    expected_true,
):
    s = _series_for_backend(backend, ['Mouse', None, 'house'])
    _assert_result_values(predicate_factory(pat)(s), expected_default)
    _assert_result_values(predicate_factory(pat, na=False)(s), expected_false)
    _assert_result_values(predicate_factory(pat, na=True)(s), expected_true)



def test_match_pandas_basic():
    s = pd.Series(['Mouse', 'dog', 'house', '123'])
    predicate = match(r'\d+')
    result = predicate(s)
    expected = pd.Series([False, False, False, True])
    pd.testing.assert_series_equal(result, expected)


def test_match_pandas_case_insensitive():
    s = pd.Series(['Mouse', 'mouse', 'MOUSE', 'dog'])
    predicate = match(r'mouse', case=False)
    result = predicate(s)
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_match_pandas_case_insensitive_with_flags():
    s = pd.Series(['Mouse', 'mouse', 'MOUSE', 'dog', None])
    predicate = match(r'mouse', case=False, flags=re.IGNORECASE)
    result = predicate(s)
    expected = pd.Series([True, True, True, False, None], dtype=object)
    pd.testing.assert_series_equal(result, expected)


def test_match_pandas_na_handling():
    s = pd.Series(['123', None, 'abc'])
    predicate = match(r'\d+')
    result = predicate(s)
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False

    # Test with na=False
    predicate = match(r'\d+', na=False)
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)

    # Test with na=True
    predicate = match(r'\d+', na=True)
    result = predicate(s)
    expected = pd.Series([True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_match_cudf_basic():
    import cudf
    s = cudf.Series(['Mouse', 'dog', 'house', '123'])
    predicate = match(r'\d+')
    result = predicate(s)
    expected = cudf.Series([False, False, False, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_cudf_case_insensitive():
    import cudf
    s = cudf.Series(['Mouse', 'mouse', 'MOUSE', 'dog'])
    predicate = match(r'mouse', case=False)
    result = predicate(s)
    expected = cudf.Series([True, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_cudf_na_handling():
    import cudf
    s = cudf.Series(['123', None, 'abc'])

    # Default NA handling
    predicate = match(r'\d+')
    result = predicate(s).to_pandas()
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False

    # NA=False
    predicate = match(r'\d+', na=False)
    result = predicate(s)
    expected = cudf.Series([True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())

    # NA=True
    predicate = match(r'\d+', na=True)
    result = predicate(s)
    expected = cudf.Series([True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_match_pandas_cudf_parity():
    import cudf

    # Create identical data
    data = ['Mouse', '123', None, 'MOUSE', 'dog123']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)

    # Test case-sensitive
    predicate = match(r'\d+')
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test case-insensitive
    predicate = match(r'mouse', case=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test with na=False
    predicate = match(r'\d+', na=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)



def test_fullmatch_pandas_basic():
    s = pd.Series(['123', '123abc', 'abc123', 'abc'])
    predicate = fullmatch(r'\d+')
    result = predicate(s)
    # Only '123' matches entirely
    expected = pd.Series([True, False, False, False])
    pd.testing.assert_series_equal(result, expected)


def test_fullmatch_pandas_case_insensitive():
    s = pd.Series(['ABC', 'abc', 'AbC', 'abcd'])
    predicate = fullmatch(r'abc', case=False)
    result = predicate(s)
    # 'abcd' has extra char
    expected = pd.Series([True, True, True, False])
    pd.testing.assert_series_equal(result, expected)


def test_fullmatch_pandas_vs_match():
    s = pd.Series(['123', '123abc', 'abc123'])

    # match() matches from start
    match_result = match(r'\d+')(s)
    expected_match = pd.Series([True, True, False])
    pd.testing.assert_series_equal(match_result, expected_match)

    # fullmatch() requires entire string
    fullmatch_result = fullmatch(r'\d+')(s)
    expected_fullmatch = pd.Series([True, False, False])
    pd.testing.assert_series_equal(fullmatch_result, expected_fullmatch)


def test_fullmatch_pandas_na_handling():
    s = pd.Series(['123', None, 'abc'])
    predicate = fullmatch(r'\d+')
    result = predicate(s)
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False

    # Test with na=False
    predicate = fullmatch(r'\d+', na=False)
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)

    # Test with na=True
    predicate = fullmatch(r'\d+', na=True)
    result = predicate(s)
    expected = pd.Series([True, True, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_fullmatch_cudf_basic():
    import cudf
    s = cudf.Series(['123', '123abc', 'abc123', 'abc'])
    predicate = fullmatch(r'\d+')
    result = predicate(s)
    expected = cudf.Series([True, False, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_fullmatch_cudf_case_insensitive():
    import cudf
    s = cudf.Series(['ABC', 'abc', 'AbC', 'abcd'])
    predicate = fullmatch(r'abc', case=False)
    result = predicate(s)
    expected = cudf.Series([True, True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_fullmatch_cudf_na_handling():
    import cudf
    s = cudf.Series(['123', None, 'abc'])

    # Default NA handling
    predicate = fullmatch(r'\d+')
    result = predicate(s).to_pandas()
    assert result[0] is True
    assert pd.isna(result[1])
    assert result[2] is False

    # NA=False
    predicate = fullmatch(r'\d+', na=False)
    result = predicate(s)
    expected = cudf.Series([True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())

    # NA=True
    predicate = fullmatch(r'\d+', na=True)
    result = predicate(s)
    expected = cudf.Series([True, True, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_fullmatch_pandas_cudf_parity():
    import cudf

    # Create identical data
    data = ['123', '123abc', None, 'ABC', 'abc']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)

    # Test case-sensitive
    predicate = fullmatch(r'\d+')
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test case-insensitive
    predicate = fullmatch(r'abc', case=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)

    # Test with na=False
    predicate = fullmatch(r'\d+', na=False)
    result_pandas = predicate(s_pandas)
    result_cudf = predicate(s_cudf).to_pandas()
    pd.testing.assert_series_equal(result_pandas, result_cudf)



def test_edge_cases_pandas():
    # Empty strings
    s = pd.Series(['', 'test', ''])
    predicate = contains('')
    result = predicate(s)
    # Empty pattern matches everything
    expected = pd.Series([True, True, True])
    pd.testing.assert_series_equal(result, expected)

    # Special regex characters
    s = pd.Series(['test.txt', 'test_txt', 'test?txt'])
    predicate = contains(r'test\.txt', regex=True)
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)

    # Unicode
    s = pd.Series(['café', 'naïve', 'test'])
    predicate = contains('café')
    result = predicate(s)
    expected = pd.Series([True, False, False])
    pd.testing.assert_series_equal(result, expected)


@requires_cudf
def test_edge_cases_cudf():
    import cudf

    # Empty strings
    s = cudf.Series(['', 'test', ''])
    predicate = contains('')
    result = predicate(s)
    expected = cudf.Series([True, True, True])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())

    # Special regex characters
    s = cudf.Series(['test.txt', 'test_txt', 'test?txt'])
    predicate = contains(r'test\.txt', regex=True)
    result = predicate(s)
    expected = cudf.Series([True, False, False])
    pd.testing.assert_series_equal(result.to_pandas(), expected.to_pandas())


@requires_cudf
def test_all_predicates_pandas_cudf_parity():
    import cudf

    # Test data with various edge cases
    data = ['Test', 'test', None, '', '123', 'Test123', 'END']
    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)

    # Test all predicates with various parameters
    predicates = [
        contains('est'),
        contains('EST', case=False),
        contains('est', na=False),
        startswith('Test'),
        startswith('Test', na=False),
        endswith('END'),
        endswith('end', na=True),
        match(r'\d+'),
        match(r'test', case=False),
        match(r'\d+', na=False),
        fullmatch(r'\d+'),
        fullmatch(r'test', case=False),
        fullmatch(r'Test123')
    ]

    for predicate in predicates:
        result_pandas = predicate(s_pandas)
        result_cudf = predicate(s_cudf).to_pandas()

        # Check if results are identical
        try:
            pd.testing.assert_series_equal(result_pandas, result_cudf)
        except AssertionError as e:
            pytest.fail(
                f"Parity check failed for "
                f"{predicate.__class__.__name__}: {e}"
            )



TUPLE_CASES = [
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['apple', 'banana', 'apricot', 'orange', None],
        {},
        [True, True, False, False, None],
    ),
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['Apple', 'BANANA', 'apricot', 'Orange', None],
        {"case": False},
        [True, True, False, False, None],
    ),
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['apple', None, 'banana', 'orange'],
        {},
        [True, None, True, False],
    ),
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['apple', None, 'banana', 'orange'],
        {"na": False},
        [True, False, True, False],
    ),
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['apple', None, 'banana', 'orange'],
        {"na": True},
        [True, True, True, False],
    ),
    (
        "startswith",
        startswith,
        ('app', 'ban'),
        ['APPLE', None, 'Banana', 'orange'],
        {"case": False, "na": False},
        [True, False, True, False],
    ),
    (
        "startswith",
        startswith,
        ('app',),
        ['apple', 'apricot', 'banana'],
        {},
        [True, False, False],
    ),
    (
        "startswith",
        startswith,
        (),
        ['apple', 'banana', 'orange'],
        {},
        [False, False, False],
    ),
    (
        "startswith",
        startswith,
        (),
        ['apple', None, 'orange'],
        {},
        [False, None, False],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.txt', 'data.csv', 'config.txt', 'image.png', None],
        {},
        [True, True, True, False, None],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.TXT', 'data.CSV', 'config.txt', 'image.PNG', None],
        {"case": False},
        [True, True, True, False, None],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.txt', None, 'data.csv', 'image.png'],
        {},
        [True, None, True, False],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.txt', None, 'data.csv', 'image.png'],
        {"na": False},
        [True, False, True, False],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.txt', None, 'data.csv', 'image.png'],
        {"na": True},
        [True, True, True, False],
    ),
    (
        "endswith",
        endswith,
        ('.txt', '.csv'),
        ['test.TXT', None, 'data.CSV', 'image.png'],
        {"case": False, "na": False},
        [True, False, True, False],
    ),
    (
        "endswith",
        endswith,
        ('.txt',),
        ['test.txt', 'data.csv', 'config.txt'],
        {},
        [True, False, True],
    ),
    (
        "endswith",
        endswith,
        (),
        ['test.txt', 'data.csv', 'image.png'],
        {},
        [False, False, False],
    ),
    (
        "endswith",
        endswith,
        (),
        ['test.txt', None, 'image.png'],
        {},
        [False, None, False],
    ),
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize(
    "name,predicate_factory,pat,data,kwargs,expected",
    TUPLE_CASES,
)
def test_boundary_string_tuple_cases(
    backend,
    name,
    predicate_factory,
    pat,
    data,
    kwargs,
    expected,
):
    s = _series_for_backend(backend, data)
    result = predicate_factory(pat, **kwargs)(s)
    _assert_result_values(result, expected)


@requires_cudf
@pytest.mark.parametrize(
    "name,predicate_factory,data,pat",
    [
        (
            "startswith",
            startswith,
            ['Apple', 'banana', None, 'application', 'orange', 'BANANA'],
            ('app', 'ban'),
        ),
        (
            "endswith",
            endswith,
            ['test.TXT', 'data.csv', None, 'config.txt', 'image.png', 'doc.CSV'],
            ('.txt', '.csv'),
        ),
    ],
)
def test_boundary_string_tuple_pandas_cudf_parity(
    name,
    predicate_factory,
    data,
    pat,
):
    import cudf

    s_pandas = pd.Series(data)
    s_cudf = cudf.Series(data)

    test_cases = [
        predicate_factory(pat),
        predicate_factory(pat, case=False),
        predicate_factory(pat, na=False),
        predicate_factory(pat, na=True),
        predicate_factory(pat, case=False, na=False),
        predicate_factory(pat, case=False, na=True),
        predicate_factory((pat[0],)),
        predicate_factory(()),
    ]

    for predicate in test_cases:
        result_pandas = predicate(s_pandas)
        result_cudf = predicate(s_cudf).to_pandas()
        try:
            pd.testing.assert_series_equal(result_pandas, result_cudf)
        except AssertionError as e:
            pytest.fail(f"Parity check failed for {name} {predicate}: {e}")


class TestCudfRegexPrep:
    """_cudf_regex_prep is a pure pattern transform (no cuDF needed): libcudf rejects
    inline flag groups, so a leading (?i) folds to the case=False path and any other
    flag declines honestly. Direct CPU coverage of every branch (viz-filter #1673)."""

    def test_non_string_passthrough(self):
        from graphistry.compute.predicates.str import _cudf_regex_prep
        assert _cudf_regex_prep(123, True) == (123, True)
        assert _cudf_regex_prep(None, False) == (None, False)

    def test_no_inline_flags_passthrough(self):
        from graphistry.compute.predicates.str import _cudf_regex_prep
        assert _cudf_regex_prep("al.*", True) == ("al.*", True)
        assert _cudf_regex_prep("a(?:b|c)d", True) == ("a(?:b|c)d", True)  # (?: not a flag group

    def test_leading_case_flag_folds(self):
        from graphistry.compute.predicates.str import _cudf_regex_prep
        assert _cudf_regex_prep("(?i)Al.*", True) == ("Al.*", False)
        assert _cudf_regex_prep("(?i)x", False) == ("x", False)
        assert _cudf_regex_prep("(?ii)x", True) == ("x", False)  # repeated i still i-only

    def test_other_inline_flags_decline(self):
        import pytest as _pytest
        from graphistry.compute.predicates.str import _cudf_regex_prep
        for pat in ["(?m)^a", "(?s).*", "(?im)a", "(?x) a b"]:
            with _pytest.raises(NotImplementedError):
                _cudf_regex_prep(pat, True)

    def test_lookaround_and_backrefs_decline(self):
        """#1675 wave-1: libcudf rejects lookaround/backrefs at kernel-compile time —
        decline honestly instead of a raw non-NIE RuntimeError (dgx-repro'd)."""
        import pytest as _pytest
        from graphistry.compute.predicates.str import _cudf_regex_prep
        for pat in ["(?=a)b", "(?!a)b", "(?<=a)b", "(?<!a)b", r"(a)\1"]:
            with _pytest.raises(NotImplementedError):
                _cudf_regex_prep(pat, True)


class TestCudfCasefoldOrDecline:
    """The cuDF case-insensitive workaround lowercases DATA + PATTERN; .lower() turns
    \\D into \\d (and \\W/\\S/\\B alike), silently INVERTING the predicate (wave-1,
    dgx-repro'd); case-crossing ranges + non-ASCII are also unsound (wave-2). Lowercase
    escapes are no-ops and must keep folding."""

    def test_plain_patterns_fold(self):
        from graphistry.compute.predicates.str import _cudf_casefold_or_decline
        assert _cudf_casefold_or_decline("NODE.1") == "node.1"
        assert _cudf_casefold_or_decline("Ab|Cd") == "ab|cd"
        assert _cudf_casefold_or_decline("[A-Z]+") == "[a-z]+"

    def test_lowercase_escapes_still_fold(self):
        """Wave-2: .lower() no-ops on \\d/\\./\\w — these worked at base and must not
        regress to NIE (a blanket backslash decline did exactly that)."""
        from graphistry.compute.predicates.str import _cudf_casefold_or_decline
        assert _cudf_casefold_or_decline(r"\d+") == r"\d+"
        assert _cudf_casefold_or_decline(r"NODE\.1") == r"node\.1"
        assert _cudf_casefold_or_decline(r"A\w+") == r"a\w+"

    def test_uppercase_escapes_decline(self):
        import pytest as _pytest
        from graphistry.compute.predicates.str import _cudf_casefold_or_decline
        for pat in [r"\D+", r"a\Wb", r"\S*", r"x\By"]:
            with _pytest.raises(NotImplementedError):
                _cudf_casefold_or_decline(pat)

    def test_case_crossing_ranges_and_nonascii_decline(self):
        """Wave-2: (?i)[A-z] silently narrows on fold; [X-b] folds to the INVALID
        [x-b]; non-ASCII folds can diverge from libcudf's lowercasing (Istanbul-I)."""
        import pytest as _pytest
        from graphistry.compute.predicates.str import _cudf_casefold_or_decline
        for pat in ["[A-z]+", "[X-b]", "\u0130stanbul",
                    "[?-Z]", "[Z-~]", "[X-^]", "[\\x41-Z]"]:  # wave-3: mixed letter/non-letter ranges
            with _pytest.raises(NotImplementedError):
                _cudf_casefold_or_decline(pat)
        assert _cudf_casefold_or_decline("[a-c]") == "[a-c]"  # same-case range folds fine
        assert _cudf_casefold_or_decline("[0-9]+") == "[0-9]+"  # non-letter range folds fine
        assert _cudf_casefold_or_decline("e-MAIL") == "e-mail"  # class-free literal hyphen folds fine
