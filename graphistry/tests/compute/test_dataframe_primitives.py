"""
Tests for DataFrame type coercion primitives (safe_concat, safe_merge).

These tests demonstrate the problem: raw pandas/cuDF operations fail when
mixing DataFrame types. The primitives should handle type conversion automatically.
"""
import unittest
import pandas as pd
import pytest


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
        from graphistry.compute.primitives import safe_concat
        from graphistry.Engine import EngineAbstract

        df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})

        result = safe_concat([df1, df2], engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.columns) == ['a', 'b']

    def test_safe_concat_pandas_with_ignore_index(self):
        """Concat with ignore_index=True."""
        from graphistry.compute.primitives import safe_concat
        from graphistry.Engine import EngineAbstract

        df1 = pd.DataFrame({'a': [1, 2]})
        df2 = pd.DataFrame({'a': [3, 4]})

        result = safe_concat([df1, df2], engine=EngineAbstract.PANDAS, ignore_index=True)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4
        assert list(result.index) == [0, 1, 2, 3]

    def test_safe_concat_empty_list(self):
        """Concat of empty list should return empty DataFrame."""
        from graphistry.compute.primitives import safe_concat
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
        from graphistry.compute.primitives import safe_merge
        from graphistry.Engine import EngineAbstract

        left = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'id': [2, 3, 4], 'score': [10, 20, 30]})

        result = safe_merge(left, right, on='id', engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2  # inner join
        assert list(result.columns) == ['id', 'val', 'score']

    def test_safe_merge_pandas_left_join(self):
        """Left merge of two pandas DataFrames."""
        from graphistry.compute.primitives import safe_merge
        from graphistry.Engine import EngineAbstract

        left = pd.DataFrame({'id': [1, 2, 3], 'val': ['a', 'b', 'c']})
        right = pd.DataFrame({'id': [2, 3, 4], 'score': [10, 20, 30]})

        result = safe_merge(left, right, on='id', how='left', engine=EngineAbstract.PANDAS)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # all left rows
        assert pd.isna(result.loc[result['id'] == 1, 'score'].iloc[0])

    def test_safe_merge_pandas_different_columns(self):
        """Merge using left_on and right_on."""
        from graphistry.compute.primitives import safe_merge
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
