# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import unittest
import numpy as np
import pandas as pd
from qlib.data import D
from qlib.tests import TestAutoData
from qlib.data.dataset.processor import MinMaxNorm, ZScoreNorm, CSZScoreNorm, CSZFillna


class TestProcessor(TestAutoData):
    TEST_INST = "SH600519"

    def test_MinMaxNorm(self):
        def normalize(df):
            min_val = np.nanmin(df.values, axis=0)
            max_val = np.nanmax(df.values, axis=0)
            ignore = min_val == max_val
            for _i, _con in enumerate(ignore):
                if _con:
                    max_val[_i] = 1
                    min_val[_i] = 0
            df[df.columns] = (df.values - min_val) / (max_val - min_val)
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        mmn = MinMaxNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        mmn.fit(df)
        mmn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_ZScoreNorm(self):
        def normalize(df):
            mean_train = np.nanmean(df.values, axis=0)
            std_train = np.nanstd(df.values, axis=0)
            ignore = std_train == 0
            for _i, _con in enumerate(ignore):
                if _con:
                    std_train[_i] = 1
                    mean_train[_i] = 0
            df[df.columns] = (df.values - mean_train) / std_train
            return df

        origin_df = D.features([self.TEST_INST], ["$high", "$open", "$low", "$close"]).tail(10)
        origin_df["test"] = 0
        df = origin_df.copy()
        zsn = ZScoreNorm(fields_group=None, fit_start_time="2021-05-31", fit_end_time="2021-06-11")
        zsn.fit(df)
        zsn.__call__(df)
        origin_df = normalize(origin_df)
        assert (df == origin_df).all().all()

    def test_CSZFillna(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[97:99])[228:238]
        df = origin_df.copy()
        CSZFillna(fields_group=None).__call__(df)
        assert ~df[1:2].isna().all().all() and origin_df[1:2].isna().all().all()

    def test_CSZScoreNorm(self):
        origin_df = D.features(D.instruments(market="csi300"), fields=["$high", "$open", "$low", "$close"])
        origin_df = origin_df.groupby("datetime", group_keys=False).apply(lambda x: x[10:12])[50:60]
        df = origin_df.copy()
        CSZScoreNorm(fields_group=None).__call__(df)
        # If we use the formula directly on the original data, we cannot get the correct result,
        # because the original data is processed by `groupby`, so we use the method of slicing,
        # taking the 2nd group of data from the original data, to calculate and compare.
        assert (df[2:4] == ((origin_df[2:4] - origin_df[2:4].mean()).div(origin_df[2:4].std()))).all().all()


class TestProcessorDataFrameCompatibility(unittest.TestCase):
    def _mixed_frame(self):
        index = pd.MultiIndex.from_product(
            [pd.date_range("2024-01-01", periods=4), ["SH600000"]],
            names=["datetime", "instrument"],
        )
        columns = pd.MultiIndex.from_tuples(
            [
                ("feature", "float32"),
                ("feature", "integer"),
                ("feature", "constant_integer"),
                ("feature", "binary_integer"),
                ("other", "sentinel"),
            ]
        )
        frame = pd.DataFrame(
            [
                [1.0, 1, 7, 0, "a"],
                [np.nan, 2, 7, 0, "b"],
                [3.0, 5, 7, 1, "c"],
                [5.0, 9, 7, 1, "d"],
            ],
            index=index,
            columns=columns,
        )
        frame[("feature", "float32")] = frame[("feature", "float32")].astype("float32")
        frame[("feature", "integer")] = frame[("feature", "integer")].astype("int64")
        frame[("feature", "constant_integer")] = frame[("feature", "constant_integer")].astype("int64")
        frame[("feature", "binary_integer")] = frame[("feature", "binary_integer")].astype("int64")
        return frame

    def test_column_replacement_preserves_frame_contract(self):
        for processor_cls in (MinMaxNorm, ZScoreNorm):
            with self.subTest(processor=processor_cls.__name__):
                frame = self._mixed_frame()
                original = frame.copy(deep=True)
                processor = processor_cls(
                    fields_group="feature",
                    fit_start_time="2024-01-01",
                    fit_end_time="2024-01-04",
                )
                processor.fit(frame)

                selected = original["feature"].to_numpy()
                if processor_cls is MinMaxNorm:
                    minimum = np.nanmin(selected, axis=0)
                    maximum = np.nanmax(selected, axis=0)
                    constant = minimum == maximum
                    minimum[constant] = 0
                    maximum[constant] = 1
                    expected = (selected - minimum) / (maximum - minimum)
                else:
                    mean = np.nanmean(selected, axis=0)
                    standard_deviation = np.nanstd(selected, axis=0)
                    constant = standard_deviation == 0
                    mean[constant] = 0
                    standard_deviation[constant] = 1
                    expected = (selected - mean) / standard_deviation

                result = processor(frame)

                self.assertIs(result, frame)
                pd.testing.assert_index_equal(result.index, original.index, exact=True)
                pd.testing.assert_index_equal(result.columns, original.columns, exact=True)
                pd.testing.assert_series_equal(result[("other", "sentinel")], original[("other", "sentinel")])
                expected_float_dtype = "float32" if processor_cls is MinMaxNorm else "float64"
                self.assertEqual(result[("feature", "float32")].dtype, np.dtype(expected_float_dtype))
                self.assertEqual(result[("feature", "integer")].dtype, np.dtype("float64"))
                self.assertEqual(result[("feature", "constant_integer")].dtype, np.dtype("int64"))
                self.assertEqual(result[("feature", "binary_integer")].dtype, np.dtype("int64"))
                pd.testing.assert_series_equal(
                    result[("feature", "constant_integer")], original[("feature", "constant_integer")]
                )
                np.testing.assert_array_equal(np.isnan(result["feature"].to_numpy()), np.isnan(expected))
                np.testing.assert_allclose(result["feature"].to_numpy(), expected, equal_nan=True)

    def test_empty_frame_preserves_schema(self):
        for processor_cls in (MinMaxNorm, ZScoreNorm):
            with self.subTest(processor=processor_cls.__name__):
                training_frame = self._mixed_frame()
                empty_frame = training_frame.iloc[0:0].copy()
                processor = processor_cls(
                    fields_group="feature",
                    fit_start_time="2024-01-01",
                    fit_end_time="2024-01-04",
                )
                processor.fit(training_frame)

                result = processor(empty_frame)

                self.assertIs(result, empty_frame)
                pd.testing.assert_frame_equal(result, training_frame.iloc[0:0])


if __name__ == "__main__":
    unittest.main()
