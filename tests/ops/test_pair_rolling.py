import numpy as np
import pandas as pd

from qlib.data.base import Expression
from qlib.data.ops import Corr, If


class StaticExpression(Expression):
    def __init__(self, series: pd.Series):
        self.series = series

    def load(self, instrument, start_index, end_index, *args):
        return self.series

    def _load_internal(self, instrument, start_index, end_index, *args):
        return self.series

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


def test_corr_aligns_zero_std_guard_for_partial_series():
    left = pd.Series([1.0, 2.0, 3.0], index=pd.RangeIndex(2, 5))
    right = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=pd.RangeIndex(0, 5))

    result = Corr(StaticExpression(left), StaticExpression(right), 2)._load_internal("partial", 0, 4)

    expected = left.rolling(2, min_periods=1).corr(right)
    pd.testing.assert_series_equal(result, expected)


def test_corr_preserves_zero_variance_nan_on_aligned_index():
    index = pd.RangeIndex(0, 4)
    left = pd.Series([1.0, 1.0, 2.0, 3.0], index=index)
    right = pd.Series([3.0, 4.0, 5.0, 6.0], index=index)

    result = Corr(StaticExpression(left), StaticExpression(right), 2)._load_internal("constant", 0, 3)

    assert np.isnan(result.loc[1])
    assert np.isfinite(result.loc[2:]).all()


def test_corr_computes_zero_variance_guard_after_index_alignment():
    left = pd.Series([100.0, 1.0, 1.000001], index=[0, 2, 3])
    right = pd.Series([0.0, 0.0, 1.0, 2.0], index=[0, 1, 2, 3])

    result = Corr(StaticExpression(left), StaticExpression(right), 3)._load_internal("gapped", 0, 3)

    assert np.isnan(result.loc[3])


def test_if_aligns_expression_branches_to_condition_index():
    condition = pd.Series([True, False, True], index=pd.RangeIndex(2, 5))
    left = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=pd.RangeIndex(0, 5))
    right = pd.Series([20.0, 21.0, 22.0, 23.0], index=pd.RangeIndex(1, 5))

    result = If(
        StaticExpression(condition),
        StaticExpression(left),
        StaticExpression(right),
    )._load_internal("partial", 0, 4)

    expected = pd.Series([12.0, 22.0, 14.0], index=condition.index)
    pd.testing.assert_series_equal(result, expected)
