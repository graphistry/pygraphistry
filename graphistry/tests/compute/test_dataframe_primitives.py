"""
Tests for DataFrame type coercion primitives (safe_concat, safe_merge).

These tests demonstrate the problem: raw pandas/cuDF operations fail when
mixing DataFrame types. The primitives should handle type conversion automatically.
"""
import os
import unittest
import pandas as pd
import pytest

_CUDF = pytest.mark.skipif(
    not (os.environ.get("TEST_CUDF") == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestMixedTypeFailures(unittest.TestCase):
    """
    Demonstrate that raw operations fail with mixed pandas/cuDF types.
    These tests should PASS (expect failure) before primitives exist.
    """

    def test_mixed_concat_fails_with_pandas_and_cudf(self):
        """
        Raw pd.concat fails when given mixed pandas/cuDF DataFrames.

        This is the core problem we're solving with safe_concat().
        """
        pandas_df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

        try:
            import cudf
            cudf_df = cudf.DataFrame({'a': [5, 6], 'b': [7, 8]})

            # This should raise TypeError or ValueError
            with pytest.raises((TypeError, ValueError, AttributeError)):
                pd.concat([pandas_df, cudf_df])

        except ImportError:
            # cuDF not available - skip this test
            pytest.skip("cuDF not available")

    def test_mixed_merge_fails_with_pandas_and_cudf(self):
        """
        Raw DataFrame.merge fails when left is pandas and right is cuDF.

        This is the core problem we're solving with safe_merge().
        """
        pandas_df = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})

        try:
            import cudf
            cudf_df = cudf.DataFrame({'id': [2, 3, 4], 'score': [10, 20, 30]})

            # This should raise TypeError or ValueError
            with pytest.raises((TypeError, ValueError, AttributeError)):
                pandas_df.merge(cudf_df, on='id')

        except ImportError:
            # cuDF not available - skip this test
            pytest.skip("cuDF not available")


class TestSafeConcatPandas(unittest.TestCase):
    """
    Test safe_concat with pandas-only DataFrames (should work without cuDF).
    """

    def test_safe_concat_pandas_basic(self):
        """Basic concat of two pandas DataFrames."""
        from graphistry.Engine import safe_concat
        from graphistry.Engine import EngineAbstract

        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        result = safe_concat([df1, df2], engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ['a', 'b']

    def test_safe_concat_pandas_with_ignore_index(self):
        """Concat with ignore_index=True."""
        from graphistry.Engine import safe_concat
        from graphistry.Engine import EngineAbstract

        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})

        result = safe_concat([df1, df2], engine=EngineAbstract.PANDAS, ignore_index=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.index) == [0, 1, 2, 3]

    def test_safe_concat_empty_list(self):
        """Concat of empty list should return empty DataFrame."""
        from graphistry.Engine import safe_concat
        from graphistry.Engine import EngineAbstract

        result = safe_concat([], engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


class TestSafeMergePandas(unittest.TestCase):
    """
    Test safe_merge with pandas-only DataFrames (should work without cuDF).
    """

    def test_safe_merge_pandas_basic(self):
        """Basic merge of two pandas DataFrames."""
        from graphistry.Engine import safe_merge
        from graphistry.Engine import EngineAbstract

        left = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'id': [2, 3, 4], 'score': [10, 20, 30]})

        result = safe_merge(left, right, on='id', engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # inner join
        assert list(result.columns) == ['id', 'val', 'score']

    def test_safe_merge_pandas_left_join(self):
        """Left merge of two pandas DataFrames."""
        from graphistry.Engine import safe_merge
        from graphistry.Engine import EngineAbstract

        left = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'id': [2, 3, 4], 'score': [10, 20, 30]})

        result = safe_merge(left, right, on='id', how='left', engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # all left rows
        assert pd.isna(result.loc[result['id'] == 1, 'score'].iloc[0])

    def test_safe_merge_pandas_different_columns(self):
        """Merge using left_on and right_on."""
        from graphistry.Engine import safe_merge
        from graphistry.Engine import EngineAbstract

        left = pd.DataFrame({'left_id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'right_id': [2, 3, 4], 'score': [10, 20, 30]})

        result = safe_merge(
            left, right,
            left_on='left_id',
            right_on='right_id',
            engine=EngineAbstract.PANDAS
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2


class TestSafeMergeNoMutation(unittest.TestCase):
    """#892: safe_merge must not mutate the caller's right DataFrame."""

    def test_right_not_mutated_same_dtype(self):
        """safe_merge does not mutate right DataFrame when dtypes already match."""
        from graphistry.Engine import safe_merge
        right = pd.DataFrame({'id': [1, 2], 'val': [10, 20]})
        right_dtype_before = right['id'].dtype
        left = pd.DataFrame({'id': [1, 2], 'x': ['a', 'b']})
        safe_merge(left, right, on='id')
        assert right['id'].dtype == right_dtype_before, "safe_merge mutated right DataFrame dtype"

    def test_right_not_mutated_on_dtype_mismatch(self):
        """safe_merge does not mutate right DataFrame even when dtype coercion is needed."""
        from graphistry.Engine import safe_merge
        left = pd.DataFrame({'id': pd.array([1, 2], dtype='int32'), 'x': ['a', 'b']})
        right = pd.DataFrame({'id': pd.array([1, 2], dtype='int64'), 'val': [10, 20]})
        right_dtype_before = right['id'].dtype
        right_id_values_before = right['id'].tolist()
        safe_merge(left, right, on='id')
        assert right['id'].dtype == right_dtype_before, "safe_merge mutated right DataFrame dtype"
        assert right['id'].tolist() == right_id_values_before, "safe_merge mutated right DataFrame values"

    @_CUDF
    def test_cudf_right_not_mutated_same_dtype(self):
        """cuDF: safe_merge does not mutate right DataFrame when dtypes already match (#892)."""
        import cudf
        from graphistry.Engine import safe_merge
        left = cudf.DataFrame({'id': [1, 2], 'x': ['a', 'b']})
        right = cudf.DataFrame({'id': [1, 2], 'val': [10, 20]})
        right_dtype_before = right['id'].dtype
        safe_merge(left, right, on='id')
        assert right['id'].dtype == right_dtype_before, "safe_merge mutated cuDF right DataFrame dtype"

    @_CUDF
    def test_cudf_right_not_mutated_on_dtype_mismatch(self):
        """cuDF: safe_merge does not mutate right DataFrame on dtype mismatch (#892 original bug)."""
        import cudf
        from graphistry.Engine import safe_merge
        # int32 left vs int64 right — the original trigger for the mutation bug
        left = cudf.DataFrame({'id': cudf.Series([1, 2], dtype='int32'), 'x': ['a', 'b']})
        right = cudf.DataFrame({'id': cudf.Series([1, 2], dtype='int64'), 'val': [10, 20]})
        right_dtype_before = right['id'].dtype
        right_id_values_before = right['id'].to_pandas().tolist()
        safe_merge(left, right, on='id')
        assert right['id'].dtype == right_dtype_before, "safe_merge mutated cuDF right DataFrame dtype"
        assert right['id'].to_pandas().tolist() == right_id_values_before, "safe_merge mutated cuDF right DataFrame values"

    @_CUDF
    def test_cudf_empty_right_not_mutated(self):
        """cuDF: safe_merge handles empty cuDF right DataFrame without mutation (#892 float64 inference case)."""
        import cudf
        from graphistry.Engine import safe_merge
        # Empty cuDF DataFrames infer float64 — the original dtype-mismatch trigger
        left = cudf.DataFrame({'id': cudf.Series([1, 2], dtype='int64'), 'x': ['a', 'b']})
        right = cudf.DataFrame({'id': cudf.Series([], dtype='float64'), 'val': cudf.Series([], dtype='float64')})
        right_dtype_before = right['id'].dtype
        result = safe_merge(left, right, on='id', how='left')
        assert right['id'].dtype == right_dtype_before, "safe_merge mutated empty cuDF right DataFrame dtype"
        assert len(result) == 2  # left join: all left rows preserved
